"""
Unit tests for Phase 2 & 2.5:
- Detection-head node exclusion (Tail-60 Rule)
- 6-format model matrix (FP32, INT8 U8S8 Full, INT8 U8S8 7bit, INT8 U8U8, INT8 Tier-S, OpenVINO IR)
- Tier-S superiority verification: Higher Cosine Similarity and Box IoU vs unmitigated INT8 Full
- Support for both YOLO11 (One-to-Many + NMS) and YOLO26 (End-to-End One-to-One NMS-Free) formats
- Pre-allocated IOBinding execution under BenchmarkProfiler
- Strict disk constraint verification (< 50MB for all 6 exported artifacts)
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.exporter import export_model_matrix, get_head_nodes_to_exclude
from src.inference import (
    TRAFFIC_CLASSES,
    OnnxRuntimeEngine,
    OpenVINOEngine,
    PyTorchEngine,
    compute_anchor_box_iou,
    compute_cosine_similarity,
    postprocess_yolo,
)
from src.profiler import BenchmarkProfiler
from src.video_loader import SyntheticVideoGenerator, preprocess_frame


@pytest.fixture(scope="session")
def exported_models_320():
    """Session-scoped fixture to ensure the 6-format matrix at imgsz=320 is exported."""
    pt_path = Path("yolo11n.pt")
    if not pt_path.exists():
        from ultralytics import YOLO
        YOLO("yolo11n.pt")

    imgsz = 320
    target_dir = Path("models") / f"yolo11n_{imgsz}"
    artifacts = {
        "fp32": target_dir / "fp32.onnx",
        "int8_u8s8_full": target_dir / "int8_u8s8_full.onnx",
        "int8_u8s8_7bit": target_dir / "int8_u8s8_7bit.onnx",
        "int8_u8u8": target_dir / "int8_u8u8.onnx",
        "int8_u8s8_tier_s": target_dir / "int8_u8s8_tier_s.onnx",
        "openvino_fp32": target_dir / "openvino_fp32",
    }

    # If any artifact is missing, export complete matrix
    all_exist = all(p.exists() for p in artifacts.values())
    if not all_exist:
        artifacts = export_model_matrix(
            model_name_or_path=pt_path,
            imgsz=imgsz,
            base_output_dir="models",
            num_calibration_frames=10,
            include_tier_s=True,
        )

    return artifacts


def test_get_head_nodes_to_exclude(exported_models_320):
    """Verify get_head_nodes_to_exclude inspects graph and extracts up to 60 tail/decode nodes."""
    fp32_path = exported_models_320["fp32"]
    nodes = get_head_nodes_to_exclude(fp32_path, max_tail_nodes=60)
    assert isinstance(nodes, list)
    assert 10 <= len(nodes) <= 60
    # Ensure head / DFL operations are identified
    assert any("/model." in n or "dfl" in n.lower() or "concat" in n.lower() for n in nodes)


def test_exported_artifacts_and_disk_constraint(exported_models_320):
    """Verify all 6 artifacts exist and total directory size is strictly under 50 MB."""
    target_dir = Path("models") / "yolo11n_320"
    assert target_dir.exists()

    expected_keys = [
        "fp32",
        "int8_u8s8_full",
        "int8_u8s8_7bit",
        "int8_u8u8",
        "int8_u8s8_tier_s",
        "openvino_fp32",
    ]
    for format_key in expected_keys:
        assert format_key in exported_models_320
        artifact_path = exported_models_320[format_key]
        assert artifact_path.exists(), f"Artifact missing: {format_key} -> {artifact_path}"

    total_bytes = sum(
        f.stat().st_size
        for f in target_dir.rglob("*")
        if f.is_file()
    )
    total_mb = total_bytes / (1024 * 1024)
    print(f"\n[Disk Check] Total size of all 6 yolo11n_320 artifacts: {total_mb:.2f} MB")
    assert total_mb < 50.0, f"Exported models exceeded 50 MB budget: {total_mb:.2f} MB"


def test_inference_engines_load_and_forward(exported_models_320):
    """Verify PyTorch, ONNX Runtime (with IOBinding), and OpenVINO engines load and run forward pass."""
    dummy_input = np.zeros((1, 3, 320, 320), dtype=np.float32)

    # 1. PyTorch
    pt_engine = PyTorchEngine("yolo11n.pt", num_threads=2)
    pt_engine.load()
    pt_out = pt_engine.forward(dummy_input)
    assert pt_out.ndim in (2, 3)

    # 2. ONNX Runtime FP32 with IOBinding
    ort_fp32 = OnnxRuntimeEngine(exported_models_320["fp32"], num_threads=2, use_io_binding=True)
    ort_fp32.load()
    assert ort_fp32.io_binding is not None
    fp32_out = ort_fp32.forward(dummy_input)
    assert fp32_out.shape == pt_out.shape

    # 3. ONNX Runtime Tier-S with IOBinding
    ort_tier_s = OnnxRuntimeEngine(exported_models_320["int8_u8s8_tier_s"], num_threads=2, use_io_binding=True)
    ort_tier_s.load()
    tier_s_out = ort_tier_s.forward(dummy_input)
    assert tier_s_out.shape == fp32_out.shape

    # 4. OpenVINO FP32
    ov_engine = OpenVINOEngine(exported_models_320["openvino_fp32"], num_threads=2)
    ov_engine.load()
    ov_out = ov_engine.forward(dummy_input)
    assert ov_out.shape == fp32_out.shape


def test_tier_s_superiority_cosine_and_box_iou(exported_models_320):
    """
    Verifies that Tier-S Zen 3 INT8 achieves higher Cosine Similarity and
    Bounding-Box IoU agreement against FP32 compared to unmitigated Full INT8.
    """
    sample_img = Path("data/coco128/images/train2017/000000000009.jpg")
    if sample_img.exists():
        raw_frame = cv2.imread(str(sample_img))
    else:
        synth = SyntheticVideoGenerator(total_frames=1, width=320, height=320, stride=1, seed=123)
        _, raw_frame = next(iter(synth))
    input_tensor = preprocess_frame(raw_frame, target_shape=(320, 320), normalize=True)

    ort_fp32 = OnnxRuntimeEngine(exported_models_320["fp32"], num_threads=2)
    ort_u8s8_full = OnnxRuntimeEngine(exported_models_320["int8_u8s8_full"], num_threads=2)
    ort_tier_s = OnnxRuntimeEngine(exported_models_320["int8_u8s8_tier_s"], num_threads=2)

    out_fp32 = ort_fp32.forward(input_tensor)
    out_full = ort_u8s8_full.forward(input_tensor)
    out_tier_s = ort_tier_s.forward(input_tensor)

    # 1. Cosine similarity
    sim_full = compute_cosine_similarity(out_fp32, out_full)
    sim_tier_s = compute_cosine_similarity(out_fp32, out_tier_s)

    # 2. Box IoU on top-scoring anchor candidates
    scores = np.max(out_fp32[0, 4:, :], axis=0)
    top_k = np.argsort(scores)[-20:]
    b_fp32 = out_fp32[0, :4, top_k].T
    b_full = out_full[0, :4, top_k].T
    b_tier_s = out_tier_s[0, :4, top_k].T

    iou_full = compute_anchor_box_iou(b_fp32, b_full)
    iou_tier_s = compute_anchor_box_iou(b_fp32, b_tier_s)

    print(f"\n[Tier-S vs Full INT8 Quality Comparison]")
    print(f"  Cosine Similarity:")
    print(f"    - Full INT8:   {sim_full:.6f}")
    print(f"    - Tier-S INT8: {sim_tier_s:.6f}")
    print(f"  Top-20 Box IoU Agreement:")
    print(f"    - Full INT8:   {iou_full:.6f}")
    print(f"    - Tier-S INT8: {iou_tier_s:.6f}")

    # Tier-S must beat Full INT8 in both metrics
    assert sim_tier_s > sim_full, f"Tier-S cosine similarity {sim_tier_s} did not beat Full {sim_full}"
    assert iou_tier_s > iou_full, f"Tier-S box IoU {iou_tier_s} did not beat Full {iou_full}"


def test_postprocess_yolo_traffic_and_nms_free():
    """Verify postprocess_yolo supports both YOLO11 (One-to-Many) and YOLO26 (NMS-Free)."""
    # 1. Standard One-to-Many format: (1, 84, 3)
    raw_o2m = np.zeros((1, 84, 3), dtype=np.float32)
    raw_o2m[0, 0:4, 0] = [100.0, 100.0, 50.0, 50.0]
    raw_o2m[0, 4 + 2, 0] = 0.90  # car
    raw_o2m[0, 0:4, 1] = [200.0, 200.0, 40.0, 40.0]
    raw_o2m[0, 4 + 9, 1] = 0.95  # non-traffic class (traffic light)
    raw_o2m[0, 0:4, 2] = [102.0, 102.0, 48.0, 48.0]
    raw_o2m[0, 4 + 2, 2] = 0.70  # overlapping car -> suppressed by NMS

    dets_o2m = postprocess_yolo(raw_o2m, conf_threshold=0.25, iou_threshold=0.45, filter_traffic_classes=True)
    assert len(dets_o2m) == 1
    assert dets_o2m[0].class_id == 2
    assert dets_o2m[0].class_name == "car"

    # 2. YOLO26 End-to-End One-to-One NMS-Free format: (1, 300, 6) -> [x1, y1, x2, y2, score, class_id]
    raw_o2o = np.zeros((1, 300, 6), dtype=np.float32)
    # Detection 1: car (class 2) with score 0.85
    raw_o2o[0, 0] = [50.0, 50.0, 150.0, 150.0, 0.85, 2.0]
    # Detection 2: bus (class 5) with score 0.75
    raw_o2o[0, 1] = [160.0, 60.0, 260.0, 180.0, 0.75, 5.0]
    # Detection 3: dog (class 16, non-traffic) with score 0.90 -> should be filtered out
    raw_o2o[0, 2] = [10.0, 10.0, 40.0, 40.0, 0.90, 16.0]
    # Detection 4: person (class 0) with low score 0.10 -> below conf_threshold
    raw_o2o[0, 3] = [30.0, 30.0, 60.0, 90.0, 0.10, 0.0]

    dets_o2o = postprocess_yolo(raw_o2o, conf_threshold=0.25, filter_traffic_classes=True)
    assert len(dets_o2o) == 2
    cnames = {d.class_name for d in dets_o2o}
    assert cnames == {"car", "bus"}
    assert dets_o2o[0].box == (50.0, 50.0, 150.0, 150.0)


def test_profiler_integration_with_tier_s_and_io_binding(exported_models_320):
    """Verify that OnnxRuntimeEngine running Tier-S with IOBinding integrates with BenchmarkProfiler."""
    profiler = BenchmarkProfiler(
        profile_name="edge_2c",
        warmup_frames=2,
    )
    profiler.start()

    engine = OnnxRuntimeEngine(exported_models_320["int8_u8s8_tier_s"], num_threads=2, use_io_binding=True)
    engine.load()
    profiler.mark_model_loaded()

    synth = SyntheticVideoGenerator(total_frames=5, width=320, height=320, stride=1)
    for frame_idx, raw_frame in synth:
        with profiler.frame_scope(frame_idx) as scope:
            with scope.record("t_decode_ms"):
                _ = raw_frame.shape
            with scope.record("t_preprocess_ms"):
                tensor = preprocess_frame(raw_frame, target_shape=(320, 320))
            with scope.record("t_inference_ms"):
                raw_out = engine.forward(tensor)
            with scope.record("t_postprocess_ms"):
                _ = postprocess_yolo(raw_out)

    profiler.stop()
    metrics = profiler.compute_metrics()

    assert metrics["warmup_frames_processed"] == 2
    assert metrics["measured_frames_processed"] == 3
    assert metrics["stages"]["t_inference_ms"]["mean"] > 0.0
    assert metrics["effective_fps"] > 0.0
    assert metrics["memory_uss_mb"]["peak_inference_uss_mb"] > 0.0
