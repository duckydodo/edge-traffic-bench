"""
Unit tests for quantitative DetectionEvaluator and TrackingAndCountingEvaluator.
Verifies COCO-style scale-partitioned mAP (mAP_small, mAP_medium, mAP_large),
MOTA, IDF1, HOTA, vehicle-partitioned IDSW/FRAG, and Counting MAE/MAPE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.evaluator import (
    DetectionEvaluator,
    MultiScaleTrafficStream,
    TrackingAndCountingEvaluator,
    compute_ap_101_point,
    compute_box_iou,
    compute_matrix_iou,
)
from src.inference import OnnxRuntimeEngine
from src.tracker import ByteTracker, CBIoUTracker


def test_iou_primitives():
    """Verifies pairwise and matrix IoU calculations."""
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (5.0, 0.0, 15.0, 10.0)
    # Inter: 5x10 = 50. Area1 = 100, Area2 = 100. Union = 150. IoU = 50/150 = 1/3.
    assert pytest.approx(compute_box_iou(b1, b2), abs=1e-4) == 1.0 / 3.0

    boxes_a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    boxes_b = np.array([[5, 0, 15, 10], [0, 0, 10, 10]], dtype=np.float32)
    mat = compute_matrix_iou(boxes_a, boxes_b)
    assert mat.shape == (2, 2)
    assert pytest.approx(mat[0, 0], abs=1e-4) == 1.0 / 3.0
    assert pytest.approx(mat[0, 1], abs=1e-4) == 1.0
    assert mat[1, 0] == 0.0
    assert mat[1, 1] == 0.0


def test_detection_evaluator_ap_controlled_cases():
    """
    Verifies scale-partitioned mAP calculation under controlled synthetic targets:
      - Small: area < 1024 px^2
      - Medium: 1024 <= area < 9216 px^2
      - Large: area >= 9216 px^2
    """
    evaluator = DetectionEvaluator()

    # Frame 0: 1 small box (20x20=400), 1 medium box (50x50=2500), 1 large box (100x100=10000)
    gt_by_frame: Dict[int, List[Dict[str, Any]]] = {
        0: [
            {"box": (10.0, 10.0, 30.0, 30.0), "class_id": 2, "class_name": "car"},      # small
            {"box": (50.0, 50.0, 100.0, 100.0), "class_id": 2, "class_name": "car"},   # medium
            {"box": (150.0, 150.0, 250.0, 250.0), "class_id": 2, "class_name": "car"}, # large
        ]
    }

    # Case 1: Exact perfect predictions
    preds_perfect: Dict[int, List[Dict[str, Any]]] = {
        0: [
            {"box": (10.0, 10.0, 30.0, 30.0), "score": 0.95, "class_id": 2},
            {"box": (50.0, 50.0, 100.0, 100.0), "score": 0.92, "class_id": 2},
            {"box": (150.0, 150.0, 250.0, 250.0), "score": 0.89, "class_id": 2},
        ]
    }

    res_perfect = evaluator.evaluate_predictions(gt_by_frame, preds_perfect)
    assert res_perfect["mAP_50"] == 1.0
    assert res_perfect["mAP_50_95"] == 1.0
    assert res_perfect["mAP_small"] == 1.0
    assert res_perfect["mAP_medium"] == 1.0
    assert res_perfect["mAP_large"] == 1.0

    # Case 2: Missing small object prediction (small object collapses)
    preds_no_small: Dict[int, List[Dict[str, Any]]] = {
        0: [
            {"box": (50.0, 50.0, 100.0, 100.0), "score": 0.92, "class_id": 2},
            {"box": (150.0, 150.0, 250.0, 250.0), "score": 0.89, "class_id": 2},
        ]
    }
    res_no_small = evaluator.evaluate_predictions(gt_by_frame, preds_no_small)
    assert res_no_small["mAP_small"] == 0.0
    assert res_no_small["mAP_medium"] == 1.0
    assert res_no_small["mAP_large"] == 1.0


def test_quantization_fidelity_tier_s_vs_full():
    """
    Verifies that Tier-S INT8 achieves significantly higher Quantization Fidelity (q_mAP_small
    and q_mAP_50_95) than unmitigated Full INT8 on multi-scale synthetic streams.
    """
    model_dir = Path("models/yolo11n_320")
    tier_s_path = model_dir / "int8_u8s8_tier_s.onnx"
    full_path = model_dir / "int8_u8s8_full.onnx"
    fp32_path = model_dir / "fp32.onnx"

    if not tier_s_path.exists() or not full_path.exists() or not fp32_path.exists():
        pytest.skip("Exported ONNX models not found")

    det_eval = DetectionEvaluator()
    ref_engine = OnnxRuntimeEngine(fp32_path, num_threads=2)
    tier_s_engine = OnnxRuntimeEngine(tier_s_path, num_threads=2)
    full_engine = OnnxRuntimeEngine(full_path, num_threads=2)

    # Evaluate Tier-S fidelity
    res_tier_s = det_eval.evaluate_quantization_fidelity(
        eval_stream=None,
        test_engine=tier_s_engine,
        ref_engine=ref_engine,
        imgsz=320,
        num_frames=15,
    )

    # Evaluate Full INT8 fidelity
    res_full = det_eval.evaluate_quantization_fidelity(
        eval_stream=None,
        test_engine=full_engine,
        ref_engine=ref_engine,
        imgsz=320,
        num_frames=15,
    )

    # Tier-S must retain positive fidelity on small objects and beat Full INT8
    assert res_tier_s["q_mAP_50_95"] > 0.20, f"Tier-S q_mAP_50_95 was {res_tier_s['q_mAP_50_95']}"
    assert res_tier_s["q_mAP_small"] > 0.15, f"Tier-S q_mAP_small was {res_tier_s['q_mAP_small']}"
    assert res_tier_s["q_mAP_50_95"] > res_full["q_mAP_50_95"]


def test_tracking_evaluator_perfect_tracking():
    """Verifies that identical GT and predicted trajectories yield perfect 1.0 MOTA, IDF1, and HOTA."""
    evaluator = TrackingAndCountingEvaluator()

    gt_tracks: Dict[int, List[Dict[str, Any]]] = {}
    pred_tracks: Dict[int, List[Dict[str, Any]]] = {}

    for t in range(10):
        frame_items = [
            {"track_id": 1, "box": (100.0, 50.0 + t * 5.0, 130.0, 90.0 + t * 5.0), "class_name": "motorcycle"},
            {"track_id": 2, "box": (300.0, 20.0 + t * 8.0, 360.0, 100.0 + t * 8.0), "class_name": "car"},
        ]
        gt_tracks[t] = frame_items
        pred_tracks[t] = [dict(item) for item in frame_items]

    metrics = evaluator.evaluate_trajectories(gt_tracks, pred_tracks)

    assert metrics["mota"] == 1.0
    assert metrics["idf1"] == 1.0
    assert metrics["hota"] == 1.0
    assert metrics["idsw_total"] == 0
    assert metrics["frag_total"] == 0


def test_tracking_evaluator_partitioned_idsw_and_frag():
    """
    Verifies that ID switches and track fragmentations are partitioned
    into Two-Wheelers vs Large Vehicles.
    """
    evaluator = TrackingAndCountingEvaluator()

    gt_tracks: Dict[int, List[Dict[str, Any]]] = {}
    pred_tracks: Dict[int, List[Dict[str, Any]]] = {}

    for t in range(10):
        # GT Track 1: motorcycle (Two-Wheeler)
        # GT Track 2: bus (Large Vehicle)
        gt_tracks[t] = [
            {"track_id": 1, "box": (100.0, float(t * 10), 125.0, float(t * 10 + 40)), "class_name": "motorcycle"},
            {"track_id": 2, "box": (400.0, float(t * 5), 480.0, float(t * 5 + 120)), "class_name": "bus"},
        ]

        # Predictions:
        # Motorcycle: ID=101 for t < 5, switches to ID=102 for t >= 5 -> 1 IDSW on Two-Wheeler
        # Bus: Tracked with ID=201, but drops out at frames 4 and 5, reappears at 6 -> 1 FRAG on Large Vehicle
        preds = []
        if t < 5:
            preds.append({"track_id": 101, "box": (100.0, float(t * 10), 125.0, float(t * 10 + 40)), "class_name": "motorcycle"})
        else:
            preds.append({"track_id": 102, "box": (100.0, float(t * 10), 125.0, float(t * 10 + 40)), "class_name": "motorcycle"})

        if t not in (4, 5):
            preds.append({"track_id": 201, "box": (400.0, float(t * 5), 480.0, float(t * 5 + 120)), "class_name": "bus"})

        pred_tracks[t] = preds

    metrics = evaluator.evaluate_trajectories(gt_tracks, pred_tracks)

    assert metrics["idsw_two_wheelers"] == 1
    assert metrics["idsw_large_vehicles"] == 0
    assert metrics["idsw_total"] == 1
    assert metrics["frag_large_vehicles"] == 1
    assert metrics["frag_two_wheelers"] == 0
    assert metrics["frag_total"] == 1
    assert metrics["mota"] < 1.0


def test_counting_mae_and_mape():
    """Verifies counting MAE and MAPE calculation against ground truth."""
    evaluator = TrackingAndCountingEvaluator()

    gt_crossings = {
        "total_crossings": 5,
        "by_class": {
            "bus": {"total": 1},
            "motorcycle": {"total": 2},
            "car": {"total": 1},
            "truck": {"total": 1},
        },
    }

    # Predicted summary: missed 1 motorcycle (pred=1 instead of 2)
    pred_summary = {
        "total_crossings": 4,
        "by_class": {
            "bus": {"total": 1},
            "motorcycle": {"total": 1},
            "car": {"total": 1},
            "truck": {"total": 1},
        },
    }

    errs = evaluator.compute_counting_errors(pred_summary, gt_crossings)
    # Errors: bus=0, motorcycle=1, car=0, truck=0. Mean error = 1/4 = 0.25.
    # MAPE: bus=0%, motorcycle=1/2=50%, car=0%, truck=0%. Mean = 50%/4 = 12.5%.
    assert pytest.approx(errs["counting_mae"], abs=1e-3) == 0.25
    assert pytest.approx(errs["counting_mape_percent"], abs=1e-2) == 12.5
    assert errs["total_counting_error"] == 1


def test_multi_stride_tracking_superiority():
    """
    Verifies that multi-stride evaluation demonstrates:
      1. Effective FPS scales up with stride S.
      2. CB-IoU maintains higher MOTA and lower counting error than ByteTrack at stride S=3.
    """
    evaluator = TrackingAndCountingEvaluator()

    res_byte = evaluator.evaluate_multi_stride(lambda s: ByteTracker(stride=s), strides=[1, 3], total_frames=60)
    res_cbiou = evaluator.evaluate_multi_stride(lambda s: CBIoUTracker(stride=s), strides=[1, 3], total_frames=60)

    # 1. Effective FPS at S=3 is significantly higher than S=1
    assert res_byte[1]["effective_fps"] > res_byte[0]["effective_fps"] * 1.5

    # 2. At S=3, CB-IoU has higher MOTA and fewer/equal ID switches than ByteTrack
    assert res_cbiou[1]["mota"] > res_byte[1]["mota"]
    assert res_cbiou[1]["idsw_two_wheelers"] <= res_byte[1]["idsw_two_wheelers"]
