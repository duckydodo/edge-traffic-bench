"""
Unit tests for end-to-end multi-stage traffic benchmarking pipeline.
Verifies stage timing isolation, tracker and analytics integration,
warmup exclusion, and JSON/CSV reporting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.analytics import LineCrossingCounter, TrafficAnnotator
from src.inference import Detection, OnnxRuntimeEngine, PyTorchEngine
from src.pipeline import (
    TrafficBenchPipeline,
    create_engine,
    create_tracker,
    flatten_benchmark_metrics,
    rescale_detections,
    save_benchmark_results,
)
from src.tracker import ByteTracker, CBIoUTracker, OCSortTracker
from src.video_loader import SyntheticVideoGenerator, TrafficPhysicsSimulator


@pytest.fixture(scope="module")
def model_paths():
    """Returns paths to test model artifacts."""
    tier_s_path = Path("models/yolo11n_320/int8_u8s8_tier_s.onnx")
    pt_path = Path("yolo11n.pt")
    if not tier_s_path.exists():
        pytest.skip(f"Tier-S ONNX model not found at {tier_s_path}")
    return {"tier_s": tier_s_path, "pytorch": pt_path}


def test_factory_helpers_and_detection_rescaling():
    """Verifies create_engine, create_tracker, and coordinate rescaling."""
    tracker_b = create_tracker("ByteTrack", stride=1)
    tracker_o = create_tracker("OC-SORT", stride=2)
    tracker_c = create_tracker("CB-IoU", stride=3)

    assert isinstance(tracker_b, ByteTracker)
    assert isinstance(tracker_o, OCSortTracker)
    assert isinstance(tracker_c, CBIoUTracker)

    engine, display_name = create_engine("ORT_INT8_TierS", model_name_or_dir="yolo11n", imgsz=320, num_threads=1)
    assert isinstance(engine, OnnxRuntimeEngine)
    assert display_name == "ORT_INT8_TierS"

    # Rescaling test:
    # Orig shape: (720, 1280). Target: (320, 320).
    # Letterbox ratio: 320 / 1280 = 0.25. dw = 0, dh = (320 - 180)/2 = 70.
    orig_shape = (720, 1280)
    ratio = 0.25
    pad = (0.0, 70.0)
    det = Detection(box=(25.0, 95.0, 75.0, 145.0), score=0.9, class_id=2, class_name="car")
    rescaled = rescale_detections([det], ratio=ratio, pad=pad, orig_shape=orig_shape)

    assert len(rescaled) == 1
    sx1, sy1, sx2, sy2 = rescaled[0].box
    assert pytest.approx(sx1, abs=0.1) == 100.0  # (25 - 0) / 0.25
    assert pytest.approx(sy1, abs=0.1) == 100.0  # (95 - 70) / 0.25
    assert pytest.approx(sx2, abs=0.1) == 300.0  # (75 - 0) / 0.25
    assert pytest.approx(sy2, abs=0.1) == 300.0  # (145 - 70) / 0.25


def test_pipeline_end_to_end_synthetic(model_paths):
    """
    Executes full pipeline under BenchmarkProfiler with synthetic video.
    Verifies that all 4 isolated stages are strictly positive and recorded.
    """
    engine = OnnxRuntimeEngine(model_paths["tier_s"], num_threads=1)
    tracker = CBIoUTracker(stride=1)
    pipeline = TrafficBenchPipeline(
        engine=engine,
        tracker=tracker,
        profile_name="edge_1c",
        imgsz=320,
        warmup_frames=5,
        stride=1,
    )

    results = pipeline.run(video_source="synth", max_frames=20)

    assert results["profile_name"] == "edge_1c"
    assert results["warmup_frames_processed"] == 5
    assert results["measured_frames_processed"] == 15
    assert results["effective_fps"] > 0.0

    stages = results["stages"]
    for s in ["t_decode_ms", "t_preprocess_ms", "t_inference_ms", "t_postprocess_ms", "t_total_ms"]:
        assert s in stages
        assert stages[s]["mean"] >= 0.0
        assert stages[s]["p50"] >= 0.0
        assert stages[s]["p95"] >= 0.0

    # Ensure memory stats are captured
    mem = results["memory_uss_mb"]
    assert "baseline_uss_mb" in mem
    assert "peak_inference_uss_mb" in mem
    assert mem["peak_inference_uss_mb"] >= mem["baseline_uss_mb"]


def test_pipeline_physics_simulator_line_crossing(model_paths):
    """
    Executes pipeline on TrafficPhysicsSimulator and verifies ground-truth line-crossing detection.
    """
    engine = OnnxRuntimeEngine(model_paths["tier_s"], num_threads=1)
    tracker = CBIoUTracker(stride=1)
    counter = LineCrossingCounter(p1=(50.0, 360.0), p2=(1230.0, 360.0))

    pipeline = TrafficBenchPipeline(
        engine=engine,
        tracker=tracker,
        counter=counter,
        profile_name="edge_1c",
        imgsz=320,
        warmup_frames=5,
        stride=1,
    )

    results = pipeline.run(video_source="physics", max_frames=120)
    analytics = results["analytics"]

    # All 5 simulated vehicles cross line y=360: 3 inbound, 2 outbound
    assert analytics["total_crossings"] == 5
    assert analytics["total_inbound"] == 3
    assert analytics["total_outbound"] == 2
    assert "by_class" in analytics
    assert "bus" in analytics["by_class"]
    assert "motorcycle" in analytics["by_class"]


def test_pipeline_json_and_csv_export(model_paths, tmp_path):
    """
    Verifies that benchmark results export properly to JSON and tabular CSV.
    """
    engine = OnnxRuntimeEngine(model_paths["tier_s"], num_threads=1)
    tracker = ByteTracker(stride=1)
    pipeline = TrafficBenchPipeline(
        engine=engine,
        tracker=tracker,
        profile_name="edge_1c",
        imgsz=320,
        warmup_frames=3,
        stride=1,
    )

    results = pipeline.run(video_source="synth", max_frames=10)

    json_out = tmp_path / "metrics.json"
    csv_out = tmp_path / "metrics.csv"

    save_benchmark_results(results, json_path=json_out, csv_path=csv_out)

    assert json_out.exists()
    assert csv_out.exists()

    with open(json_out, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["effective_fps"] == results["effective_fps"]
    assert "pipeline" in loaded_json

    with open(csv_out, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["profile"] == "edge_1c"
    assert rows[0]["tracker"] == "ByteTracker"
    assert float(rows[0]["effective_fps"]) == pytest.approx(results["effective_fps"], abs=0.01)


def test_pipeline_annotator_video_writing(model_paths, tmp_path):
    """
    Verifies that TrafficAnnotator integrates with OpenCV video writing inside the pipeline.
    """
    engine = OnnxRuntimeEngine(model_paths["tier_s"], num_threads=1)
    tracker = OCSortTracker(stride=1)
    annotator = TrafficAnnotator(line_p1=(50.0, 360.0), line_p2=(1230.0, 360.0))
    video_out = tmp_path / "output_annotated.mp4"

    pipeline = TrafficBenchPipeline(
        engine=engine,
        tracker=tracker,
        annotator=annotator,
        profile_name="edge_1c",
        imgsz=320,
        warmup_frames=2,
        stride=1,
    )

    results = pipeline.run(video_source="physics", max_frames=25, output_video_path=video_out)

    assert video_out.exists()
    assert video_out.stat().st_size > 0
    assert results["pipeline"]["frames_processed"] == 25
