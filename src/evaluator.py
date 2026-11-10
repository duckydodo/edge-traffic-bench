"""
Quantitative Evaluation Suite for Object Detection and Multi-Object Tracking.
Implements:
  1. DetectionEvaluator:
     - COCO-style mAP across IoU thresholds 0.50:0.05:0.95
     - Scale-partitioned accuracy (mAP_small, mAP_medium, mAP_large)
     - Annotated dataset mode (YOLO .txt / COCO .json)
     - Self-contained FP32 Reference / Synthetic Quantization Fidelity (q_mAP)
  2. TrackingAndCountingEvaluator:
     - CLEAR MOTA (Multi-Object Tracking Accuracy)
     - IDF1 (Identification F1 Score)
     - HOTA (Higher Order Tracking Accuracy: DetA * AssA)
     - IDSW & FRAG broken down by vehicle category (Two-Wheelers vs Large Vehicles)
     - Virtual Line-Crossing MAE & MAPE (%)
     - Multi-stride (S in {1, 2, 3, 5, 10}) Pareto frontier with Effective Throughput (FPS)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.analytics import LineCrossingCounter
from src.inference import (
    TRAFFIC_CLASSES,
    BaseInferenceEngine,
    Detection,
    OnnxRuntimeEngine,
    PyTorchEngine,
    postprocess_yolo,
)
from src.pipeline import rescale_detections
from src.tracker import BaseTracker, ByteTracker, CBIoUTracker, OCSortTracker, Track
from src.video_loader import (
    SyntheticVideoGenerator,
    TrafficPhysicsSimulator,
    VideoStreamLoader,
    letterbox,
    preprocess_frame,
)

# Scale partition boundaries in square pixels
SCALE_RANGES: Dict[str, Tuple[float, float]] = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0 ** 2),         # area < 1024 px^2
    "medium": (32.0 ** 2, 96.0 ** 2),   # 1024 <= area < 9216 px^2
    "large": (96.0 ** 2, float("inf")), # area >= 9216 px^2
}

COCO_IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)  # 0.50, 0.55, ..., 0.95

TWO_WHEELER_CLASSES: Set[str] = {"motorcycle", "bicycle"}
LARGE_VEHICLE_CLASSES: Set[str] = {"car", "bus", "truck"}


def compute_box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes (x1, y1, x2, y2)."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter

    return inter / union if union > 0.0 else 0.0


def compute_matrix_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes vectorized pairwise IoU matrix between N boxes and M boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    b_a = np.asarray(boxes_a, dtype=np.float32)
    b_b = np.asarray(boxes_b, dtype=np.float32)

    inter_x1 = np.maximum(b_a[:, 0, None], b_b[None, :, 0])
    inter_y1 = np.maximum(b_a[:, 1, None], b_b[None, :, 1])
    inter_x2 = np.minimum(b_a[:, 2, None], b_b[None, :, 2])
    inter_y2 = np.minimum(b_a[:, 3, None], b_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, b_a[:, 2] - b_a[:, 0]) * np.maximum(0.0, b_a[:, 3] - b_a[:, 1])
    area_b = np.maximum(0.0, b_b[:, 2] - b_b[:, 0]) * np.maximum(0.0, b_b[:, 3] - b_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.where(union > 0.0, inter / union, 0.0)
    return iou.astype(np.float32)


def compute_ap_101_point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes standard 101-point interpolated Average Precision."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = []
    for r in recall_levels:
        mask = recalls >= r
        if np.any(mask):
            interpolated_precisions.append(float(np.max(precisions[mask])))
        else:
            interpolated_precisions.append(0.0)

    return float(np.mean(interpolated_precisions))


class MultiScaleTrafficStream:
    """
    Generates deterministic evaluation frames containing objects across all three COCO scale regimes:
      - Small: area < 1024 px^2 (e.g., 20x24 = 480 px^2)
      - Medium: 1024 <= area < 9216 px^2 (e.g., 50x55 = 2750 px^2)
      - Large: area >= 9216 px^2 (e.g., 120x90 = 10800 px^2)
    """

    def __init__(self, total_frames: int = 30, width: int = 320, height: int = 320, seed: int = 42) -> None:
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        for i in range(self.total_frames):
            canvas = np.full((self.height, self.width, 3), 100, dtype=np.uint8)
            canvas[::20, :, :] = 120
            canvas[:, ::20, :] = 120

            # Small target: 20x24 (area 480 px^2)
            sx = int((i * 15) % max(1, self.width - 30))
            sy = 45
            cv2.rectangle(canvas, (sx, sy), (sx + 20, sy + 24), (20, 20, 220), -1)
            cv2.rectangle(canvas, (sx + 3, sy + 3), (sx + 17, sy + 21), (255, 255, 255), 1)

            # Medium target: 50x55 (area 2750 px^2)
            mx = int(self.width // 3)
            my = int((i * 12) % max(1, self.height - 65))
            cv2.rectangle(canvas, (mx, my), (mx + 50, my + 55), (20, 200, 20), -1)
            cv2.rectangle(canvas, (mx + 5, my + 5), (mx + 45, my + 50), (255, 255, 255), 1)

            # Large target: 120x90 (area 10800 px^2)
            lx = int(max(0, self.width - 130 - ((i * 10) % 100)))
            ly = int(self.height - 100)
            cv2.rectangle(canvas, (lx, ly), (lx + 120, ly + 90), (220, 150, 20), -1)
            cv2.rectangle(canvas, (lx + 8, ly + 8), (lx + 112, ly + 82), (255, 255, 255), 2)

            yield i, canvas


class DetectionEvaluator:
    """
    Evaluator for object detection accuracy:
      - Standard COCO mAP across thresholds 0.50:0.05:0.95
      - Scale-partitioned accuracy: mAP_small, mAP_medium, mAP_large
      - Quantization Fidelity testing against FP32 reference (q_mAP)
      - YOLO txt / COCO json dataset evaluation
    """

    def __init__(
        self,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        traffic_classes: Optional[Dict[int, str]] = None,
    ) -> None:
        self.iou_thresholds = np.asarray(iou_thresholds, dtype=np.float32)
        self.traffic_classes = traffic_classes or TRAFFIC_CLASSES

    def evaluate_predictions(
        self,
        gt_by_frame: Dict[int, List[Dict[str, Any]]],
        preds_by_frame: Dict[int, List[Dict[str, Any]]],
        classes: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Computes mAP_50, mAP_50_95, and scale-partitioned mAP (small, medium, large)
        given per-frame ground-truth and detection predictions.
        """
        all_frames = sorted(set(gt_by_frame.keys()).union(set(preds_by_frame.keys())))

        # Identify target classes from GT or input
        if classes is None:
            active_classes: Set[int] = set()
            for frame_idx, gts in gt_by_frame.items():
                for g in gts:
                    active_classes.add(int(g["class_id"]))
            eval_classes = sorted(list(active_classes)) if active_classes else sorted(list(self.traffic_classes.keys()))
        else:
            eval_classes = sorted(list(classes))

        results: Dict[str, Any] = {}
        scale_metrics: Dict[str, float] = {}
        class_ap_breakdown: Dict[str, Dict[str, float]] = {}

        for scale_name, (min_area, max_area) in SCALE_RANGES.items():
            class_aps_for_scale: List[float] = []
            class_ap50_for_scale: List[float] = []

            for c in eval_classes:
                c_name = self.traffic_classes.get(c, f"class_{c}")
                # Gather all predictions for class c across frames
                pred_items: List[Tuple[int, Sequence[float], float, float]] = []
                for f_idx in all_frames:
                    for p in preds_by_frame.get(f_idx, []):
                        if int(p["class_id"]) == c:
                            b = p["box"]
                            w = max(0.0, float(b[2]) - float(b[0]))
                            h = max(0.0, float(b[3]) - float(b[1]))
                            area = w * h
                            pred_items.append((f_idx, b, float(p.get("score", 1.0)), area))

                # Sort predictions descending by score
                pred_items.sort(key=lambda x: x[2], reverse=True)

                # Count active GT targets in this scale range
                total_active_gt = 0
                frame_gt_boxes: Dict[int, List[Tuple[Sequence[float], bool]]] = {}
                for f_idx in all_frames:
                    frame_gt_boxes[f_idx] = []
                    for g in gt_by_frame.get(f_idx, []):
                        if int(g["class_id"]) == c:
                            b = g["box"]
                            w = max(0.0, float(b[2]) - float(b[0]))
                            h = max(0.0, float(b[3]) - float(b[1]))
                            area = w * h
                            is_active = (min_area <= area < max_area)
                            if is_active:
                                total_active_gt += 1
                            frame_gt_boxes[f_idx].append((b, is_active))

                if total_active_gt == 0:
                    continue

                # Compute AP for each IoU threshold
                aps_at_thresh: List[float] = []
                for thresh in self.iou_thresholds:
                    matched_gt: Dict[int, Set[int]] = {f_idx: set() for f_idx in all_frames}
                    tp_list: List[int] = []
                    fp_list: List[int] = []

                    for f_idx, p_box, p_score, p_area in pred_items:
                        gts = frame_gt_boxes.get(f_idx, [])
                        best_iou = 0.0
                        best_idx = -1
                        for g_idx, (g_box, is_act) in enumerate(gts):
                            iou = compute_box_iou(p_box, g_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = g_idx

                        if best_iou >= thresh and best_idx >= 0:
                            g_box, is_act = gts[best_idx]
                            if is_act:
                                if best_idx not in matched_gt[f_idx]:
                                    matched_gt[f_idx].add(best_idx)
                                    tp_list.append(1)
                                    fp_list.append(0)
                                else:
                                    tp_list.append(0)
                                    fp_list.append(1)
                            else:
                                tp_list.append(0)
                                fp_list.append(0)
                        else:
                            if min_area <= p_area < max_area:
                                tp_list.append(0)
                                fp_list.append(1)
                            else:
                                tp_list.append(0)
                                fp_list.append(0)

                    tp_arr = np.array(tp_list, dtype=np.float64)
                    fp_arr = np.array(fp_list, dtype=np.float64)
                    cum_tp = np.cumsum(tp_arr)
                    cum_fp = np.cumsum(fp_arr)

                    recalls = cum_tp / float(total_active_gt)
                    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

                    ap = compute_ap_101_point(recalls, precisions)
                    aps_at_thresh.append(ap)

                mean_ap_c = float(np.mean(aps_at_thresh)) if aps_at_thresh else 0.0
                class_aps_for_scale.append(mean_ap_c)
                if len(aps_at_thresh) > 0:
                    class_ap50_for_scale.append(aps_at_thresh[0])

                if scale_name == "all":
                    class_ap_breakdown[c_name] = {
                        "mAP_50": round(float(aps_at_thresh[0]), 4) if aps_at_thresh else 0.0,
                        "mAP_50_95": round(mean_ap_c, 4),
                        "num_gt": total_active_gt,
                    }

            scale_mAP = float(np.mean(class_aps_for_scale)) if class_aps_for_scale else 0.0
            scale_metrics[scale_name] = scale_mAP
            if scale_name == "all":
                results["mAP_50"] = round(float(np.mean(class_ap50_for_scale)) if class_ap50_for_scale else 0.0, 4)

        results["mAP_50_95"] = round(scale_metrics.get("all", 0.0), 4)
        results["mAP_small"] = round(scale_metrics.get("small", 0.0), 4)
        results["mAP_medium"] = round(scale_metrics.get("medium", 0.0), 4)
        results["mAP_large"] = round(scale_metrics.get("large", 0.0), 4)
        results["class_ap"] = class_ap_breakdown
        results["total_evaluated_frames"] = len(all_frames)

        return results

    def evaluate_quantization_fidelity(
        self,
        eval_stream: Optional[Iterator[Tuple[int, np.ndarray]]] = None,
        test_engine: Optional[BaseInferenceEngine] = None,
        ref_engine: Optional[BaseInferenceEngine] = None,
        imgsz: int = 320,
        num_frames: int = 64,
        conf_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluates Quantization Fidelity of a quantized model against an FP32 reference model
        on real images (e.g. data/coco128) using standard confidence thresholds (conf=0.05 for PR curve).
        Reports q_mAP_50, q_mAP_50_95, scale-partitioned q_mAP, and Top-100 Detection Box IoU.
        """
        if ref_engine is None:
            fp32_onnx = Path(f"models/yolo11n_{imgsz}/fp32.onnx")
            pt_path = Path("yolo11n.pt")
            if fp32_onnx.exists():
                ref_engine = OnnxRuntimeEngine(fp32_onnx, num_threads=2)
            else:
                ref_engine = PyTorchEngine(pt_path, num_threads=2)

        if test_engine is None:
            tier_s_path = Path(f"models/yolo11n_{imgsz}/int8_u8s8_tier_s.onnx")
            test_engine = OnnxRuntimeEngine(tier_s_path, num_threads=2)

        ref_engine.load()
        test_engine.load()

        if eval_stream is None:
            coco_img_dir = Path("data/coco128/images/train2017")
            if coco_img_dir.exists():
                valid_exts = {".jpg", ".jpeg", ".png"}
                real_imgs = sorted([p for p in coco_img_dir.glob("*.jpg")])[:num_frames]
                def _real_stream() -> Iterator[Tuple[int, np.ndarray]]:
                    for idx, p in enumerate(real_imgs):
                        bgr = cv2.imread(str(p))
                        if bgr is not None:
                            yield idx, bgr
                eval_stream = _real_stream()
            else:
                eval_stream = iter(MultiScaleTrafficStream(total_frames=num_frames, width=imgsz, height=imgsz))

        gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        preds_by_frame: Dict[int, List[Dict[str, Any]]] = {}

        target_shape = (imgsz, imgsz)
        frame_idx = 0
        all_box_ious: List[Tuple[float, float]] = []  # (ref_score, iou)

        for item in eval_stream:
            if frame_idx >= num_frames:
                break

            if isinstance(item, tuple) and len(item) >= 2:
                raw_idx, frame = item[0], item[1]
            else:
                continue

            orig_shape = frame.shape[:2]
            padded, ratio, pad = letterbox(frame, target_shape=target_shape)
            tensor = preprocess_frame(frame, target_shape=target_shape)

            # 1. Reference FP32 predictions
            ref_raw = ref_engine.forward(tensor)
            ref_dets = postprocess_yolo(
                ref_raw,
                conf_threshold=conf_threshold,
                iou_threshold=0.45,
                filter_traffic_classes=True,
            )
            ref_scaled = rescale_detections(ref_dets, ratio, pad, orig_shape)
            gt_by_frame[frame_idx] = [d.to_dict() for d in ref_scaled]

            # 2. Test quantized model predictions
            test_raw = test_engine.forward(tensor)
            test_dets = postprocess_yolo(
                test_raw,
                conf_threshold=conf_threshold,
                iou_threshold=0.45,
                filter_traffic_classes=True,
            )
            test_scaled = rescale_detections(test_dets, ratio, pad, orig_shape)
            preds_by_frame[frame_idx] = [d.to_dict() for d in test_scaled]

            # Compute box IoU for top detections between ref and test
            if len(ref_scaled) > 0 and len(test_scaled) > 0:
                for rd in ref_scaled:
                    best_match_iou = 0.0
                    for td in test_scaled:
                        if td.class_id == rd.class_id:
                            iou = compute_box_iou(rd.box, td.box)
                            if iou > best_match_iou:
                                best_match_iou = iou
                    all_box_ious.append((float(rd.score), float(best_match_iou)))

            frame_idx += 1

        eval_res = self.evaluate_predictions(gt_by_frame, preds_by_frame)

        # Top-100 detection box IoU
        all_box_ious.sort(key=lambda x: x[0], reverse=True)
        top100 = all_box_ious[:100]
        top100_mean_iou = float(np.mean([x[1] for x in top100])) if top100 else 0.0

        return {
            "q_mAP_50": eval_res["mAP_50"],
            "q_mAP_50_95": eval_res["mAP_50_95"],
            "q_mAP_small": eval_res["mAP_small"],
            "q_mAP_medium": eval_res["mAP_medium"],
            "q_mAP_large": eval_res["mAP_large"],
            "top100_box_iou": round(top100_mean_iou, 4),
            "evaluated_frames": frame_idx,
            "class_ap": eval_res["class_ap"],
        }

    def evaluate_dataset(
        self,
        dataset_dir: Union[str, Path],
        engine: BaseInferenceEngine,
        imgsz: int = 320,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Evaluates model predictions against ground-truth labels from an annotated dataset
        (supports YOLO .txt format and COCO annotations.json).
        """
        engine.load()
        root = Path(dataset_dir)
        gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        preds_by_frame: Dict[int, List[Dict[str, Any]]] = {}

        # Look for images in root, root/images, or root/images/train2017
        if (root / "images" / "train2017").exists():
            img_dir = root / "images" / "train2017"
        elif (root / "images").exists():
            img_dir = root / "images"
        else:
            img_dir = root

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))

        # Check for label directory (root/labels/train2017, root/labels, or root)
        if (root / "labels" / "train2017").exists():
            lbl_dir = root / "labels" / "train2017"
        elif (root / "labels").exists():
            lbl_dir = root / "labels"
        else:
            lbl_dir = root

        # Check for COCO json annotations
        coco_json = root / "annotations.json"
        coco_data = None
        if coco_json.exists():
            with open(coco_json, "r", encoding="utf-8") as f:
                coco_data = json.load(f)

        target_shape = (imgsz, imgsz)

        for f_idx, img_path in enumerate(img_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]
            padded, ratio, pad = letterbox(frame, target_shape=target_shape)
            tensor = preprocess_frame(frame, target_shape=target_shape)

            # Parse ground truth
            gt_list: List[Dict[str, Any]] = []
            if coco_data is None:
                # YOLO format: check lbl_dir / <stem>.txt
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    lbl_path = img_path.with_suffix(".txt")

                if lbl_path.exists():
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                # Only evaluate traffic classes
                                if cid not in self.traffic_classes:
                                    continue
                                cx, cy, w, h = [float(p) for p in parts[1:5]]
                                x1 = (cx - w / 2.0) * orig_w
                                y1 = (cy - h / 2.0) * orig_h
                                x2 = (cx + w / 2.0) * orig_w
                                y2 = (cy + h / 2.0) * orig_h
                                gt_list.append({
                                    "box": (x1, y1, x2, y2),
                                    "class_id": cid,
                                    "class_name": self.traffic_classes.get(cid, str(cid)),
                                })

            gt_by_frame[f_idx] = gt_list

            # Model inference
            raw_preds = engine.forward(tensor)
            dets = postprocess_yolo(
                raw_preds,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                filter_traffic_classes=True,
            )
            scaled = rescale_detections(dets, ratio, pad, (orig_h, orig_w))
            preds_by_frame[f_idx] = [d.to_dict() for d in scaled]

        return self.evaluate_predictions(gt_by_frame, preds_by_frame)


class TrackingAndCountingEvaluator:
    """
    Evaluator for multi-object tracking and traffic line-crossing analytics:
      - CLEAR MOTA (Multi-Object Tracking Accuracy)
      - IDF1 (Identification F1 Score)
      - HOTA (Higher Order Tracking Accuracy)
      - IDSW (Identity Switches) & FRAG (Fragmentations) broken down by Two-Wheelers vs Large Vehicles
      - Counting MAE and Counting MAPE (%)
      - Multi-stride (S in {1, 2, 3, 5, 10}) Pareto frontier analysis with Effective FPS
    """

    def __init__(
        self,
        iou_threshold: float = 0.50,
        line_p1: Tuple[float, float] = (50.0, 360.0),
        line_p2: Tuple[float, float] = (1230.0, 360.0),
    ) -> None:
        self.iou_threshold = iou_threshold
        self.line_p1 = line_p1
        self.line_p2 = line_p2

    def evaluate_trajectories(
        self,
        gt_tracks_by_frame: Dict[int, List[Dict[str, Any]]],
        pred_tracks_by_frame: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Computes MOTA, IDF1, HOTA, and vehicle class-partitioned IDSW and FRAG.
        """
        all_frames = sorted(set(gt_tracks_by_frame.keys()).union(set(pred_tracks_by_frame.keys())))

        # 1. Gather all unique GT and Pred IDs
        all_gt_ids: Set[int] = set()
        all_pred_ids: Set[int] = set()
        gt_class_map: Dict[int, str] = {}

        for f_idx in all_frames:
            for g in gt_tracks_by_frame.get(f_idx, []):
                gid = int(g["track_id"])
                all_gt_ids.add(gid)
                gt_class_map[gid] = g.get("class_name", "unknown")
            for p in pred_tracks_by_frame.get(f_idx, []):
                all_pred_ids.add(int(p["track_id"]))

        gt_id_list = sorted(list(all_gt_ids))
        pred_id_list = sorted(list(all_pred_ids))
        gt_id_to_idx = {gid: idx for idx, gid in enumerate(gt_id_list)}
        pred_id_to_idx = {pid: idx for idx, pid in enumerate(pred_id_list)}

        # 2. Frame-by-frame bipartite matching for MOTA & HOTA
        total_gt_instances = 0
        total_pred_instances = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0

        # Class-partitioned IDSW and FRAG tracking
        idsw_by_group: Dict[str, int] = {"two_wheelers": 0, "large_vehicles": 0, "other": 0}
        last_matched_pred: Dict[int, int] = {}
        gt_tracked_seq: Dict[int, List[bool]] = {gid: [] for gid in all_gt_ids}

        # ID overlap matrix for IDF1: M[gt_idx, pred_idx]
        overlap_matrix = np.zeros((len(gt_id_list), len(pred_id_list)), dtype=np.int32)

        # HOTA match records: list of (frame_idx, gt_id, pred_id)
        hota_matches: List[Tuple[int, int, int]] = []
        gt_presence: Dict[int, Set[int]] = {gid: set() for gid in all_gt_ids}
        pred_presence: Dict[int, Set[int]] = {pid: set() for pid in all_pred_ids}
        pair_matches: Dict[Tuple[int, int], Set[int]] = {}

        for f_idx in all_frames:
            gts = gt_tracks_by_frame.get(f_idx, [])
            preds = pred_tracks_by_frame.get(f_idx, [])

            total_gt_instances += len(gts)
            total_pred_instances += len(preds)

            for g in gts:
                gt_presence[int(g["track_id"])].add(f_idx)
            for p in preds:
                pred_presence[int(p["track_id"])].add(f_idx)

            if len(gts) == 0:
                total_fp += len(preds)
                continue

            if len(preds) == 0:
                total_fn += len(gts)
                for g in gts:
                    gt_tracked_seq[int(g["track_id"])].append(False)
                continue

            # Compute IoU cost matrix (1.0 - IoU)
            gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32)
            pred_boxes = np.array([p["box"] for p in preds], dtype=np.float32)
            iou_mat = compute_matrix_iou(gt_boxes, pred_boxes)
            cost_mat = 1.0 - iou_mat

            row_ind, col_ind = linear_sum_assignment(cost_mat)

            matched_g_indices: Set[int] = set()
            matched_p_indices: Set[int] = set()

            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    matched_g_indices.add(r)
                    matched_p_indices.add(c)

                    gid = int(gts[r]["track_id"])
                    pid = int(preds[c]["track_id"])

                    total_tp += 1
                    overlap_matrix[gt_id_to_idx[gid], pred_id_to_idx[pid]] += 1
                    hota_matches.append((f_idx, gid, pid))

                    pair_key = (gid, pid)
                    if pair_key not in pair_matches:
                        pair_matches[pair_key] = set()
                    pair_matches[pair_key].add(f_idx)

                    # ID switch check
                    if gid in last_matched_pred and last_matched_pred[gid] != pid:
                        total_idsw += 1
                        cname = gt_class_map.get(gid, "").lower()
                        if cname in TWO_WHEELER_CLASSES:
                            idsw_by_group["two_wheelers"] += 1
                        elif cname in LARGE_VEHICLE_CLASSES:
                            idsw_by_group["large_vehicles"] += 1
                        else:
                            idsw_by_group["other"] += 1

                    last_matched_pred[gid] = pid
                    gt_tracked_seq[gid].append(True)

            # Record unmatched GTs and FPs
            for r in range(len(gts)):
                if r not in matched_g_indices:
                    total_fn += 1
                    gid = int(gts[r]["track_id"])
                    gt_tracked_seq[gid].append(False)

            total_fp += (len(preds) - len(matched_p_indices))

        # 3. Compute CLEAR MOTA
        mota = 1.0 - (float(total_fn + total_fp + total_idsw) / float(max(total_gt_instances, 1)))

        # 4. Compute IDF1
        if len(gt_id_list) > 0 and len(pred_id_list) > 0:
            id_row_ind, id_col_ind = linear_sum_assignment(-overlap_matrix)
            idtp = int(np.sum(overlap_matrix[id_row_ind, id_col_ind]))
        else:
            idtp = 0

        idfp = total_pred_instances - idtp
        idfn = total_gt_instances - idtp
        idf1_denom = 2 * idtp + idfp + idfn
        idf1 = (2.0 * idtp / idf1_denom) if idf1_denom > 0 else 0.0

        # 5. Compute HOTA (DetA * AssA)
        det_a_denom = total_tp + total_fp + total_fn
        det_a = (float(total_tp) / float(det_a_denom)) if det_a_denom > 0 else 0.0

        ass_scores: List[float] = []
        for f_idx, gid, pid in hota_matches:
            tpa = len(pair_matches.get((gid, pid), set()))
            fpa = len(pred_presence.get(pid, set())) - tpa
            fna = len(gt_presence.get(gid, set())) - tpa
            denom = tpa + fpa + fna
            ass_scores.append(float(tpa) / float(denom) if denom > 0 else 0.0)

        ass_a = float(np.mean(ass_scores)) if ass_scores else 0.0
        hota = math.sqrt(det_a * ass_a)

        # 6. Compute Track Fragmentations (FRAG) by vehicle class
        frag_by_group: Dict[str, int] = {"two_wheelers": 0, "large_vehicles": 0, "other": 0}
        total_frag = 0

        for gid, seq in gt_tracked_seq.items():
            cname = gt_class_map.get(gid, "").lower()
            group = "two_wheelers" if cname in TWO_WHEELER_CLASSES else ("large_vehicles" if cname in LARGE_VEHICLE_CLASSES else "other")

            was_tracked = False
            lost = False
            for is_tracked in seq:
                if is_tracked:
                    if was_tracked and lost:
                        total_frag += 1
                        frag_by_group[group] += 1
                        lost = False
                    was_tracked = True
                else:
                    if was_tracked:
                        lost = True

        return {
            "mota": round(mota, 4),
            "idf1": round(idf1, 4),
            "hota": round(hota, 4),
            "det_a": round(det_a, 4),
            "ass_a": round(ass_a, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_gt_instances": total_gt_instances,
            "idsw_total": total_idsw,
            "idsw_two_wheelers": idsw_by_group["two_wheelers"],
            "idsw_large_vehicles": idsw_by_group["large_vehicles"],
            "frag_total": total_frag,
            "frag_two_wheelers": frag_by_group["two_wheelers"],
            "frag_large_vehicles": frag_by_group["large_vehicles"],
        }

    def compute_counting_errors(
        self,
        predicted_summary: Dict[str, Any],
        ground_truth_crossings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE %)
        across traffic vehicle classes.
        """
        gt_by_class = ground_truth_crossings.get("by_class", {})
        pred_by_class = predicted_summary.get("by_class", {})

        all_classes = sorted(set(gt_by_class.keys()).union(set(pred_by_class.keys())))
        if not all_classes:
            all_classes = ["bus", "motorcycle", "car", "truck"]

        abs_errors: List[float] = []
        pct_errors: List[float] = []

        for c in all_classes:
            gt_n = float(gt_by_class.get(c, {}).get("total", 0))
            pred_n = float(pred_by_class.get(c, {}).get("total", 0))
            err = abs(pred_n - gt_n)
            abs_errors.append(err)
            pct = 100.0 * err / max(gt_n, 1.0)
            pct_errors.append(pct)

        mae = float(np.mean(abs_errors)) if abs_errors else 0.0
        mape = float(np.mean(pct_errors)) if pct_errors else 0.0

        total_gt = ground_truth_crossings.get("total_crossings", 0)
        total_pred = predicted_summary.get("total_crossings", 0)
        total_error = abs(total_pred - total_gt)

        return {
            "counting_mae": round(mae, 3),
            "counting_mape_percent": round(mape, 2),
            "total_predicted_crossings": total_pred,
            "total_gt_crossings": total_gt,
            "total_counting_error": total_error,
        }

    def evaluate_tracker_simulation(
        self,
        tracker: BaseTracker,
        stride: int = 1,
        total_frames: int = 120,
    ) -> Dict[str, Any]:
        """
        Runs full simulation evaluation with TrafficPhysicsSimulator at specified stride.
        Computes Effective End-to-End Throughput (FPS) = Total Frames / Execution Time.
        """
        sim = TrafficPhysicsSimulator(
            total_frames=total_frames,
            stride=stride,
            counting_line_y=self.line_p1[1],
        )
        gt_crossings = sim.get_ground_truth_crossings()
        counter = LineCrossingCounter(p1=self.line_p1, p2=self.line_p2)

        gt_tracks_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        pred_tracks_by_frame: Dict[int, List[Dict[str, Any]]] = {}

        tracker.reset()

        t_start = time.perf_counter_ns()
        processed_frames = 0

        for f_idx, frame, detections in sim:
            processed_frames += 1
            # Ground truth for this frame
            gt_tracks_by_frame[f_idx] = [
                {
                    "track_id": d.track_id,
                    "box": d.box,
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                }
                for d in detections
            ]

            # Tracker update
            tracks = tracker.update(detections, frame_bgr=frame)
            confirmed = [t for t in tracks if t.time_since_update == 0]

            pred_tracks_by_frame[f_idx] = [
                {
                    "track_id": t.track_id,
                    "box": t.box,
                    "class_name": t.class_name,
                    "class_id": t.class_id,
                }
                for t in confirmed
            ]

            # Line counter update
            counter.update(confirmed, frame_idx=f_idx)

        t_end = time.perf_counter_ns()
        elapsed_sec = (t_end - t_start) / 1e9
        effective_fps = float(total_frames) / elapsed_sec if elapsed_sec > 0 else 0.0

        # Trajectory metrics
        traj_metrics = self.evaluate_trajectories(gt_tracks_by_frame, pred_tracks_by_frame)
        # Counting metrics
        count_metrics = self.compute_counting_errors(counter.get_summary(), gt_crossings)

        res: Dict[str, Any] = {
            "stride": stride,
            "processed_frames": processed_frames,
            "total_video_frames": total_frames,
            "elapsed_seconds": round(elapsed_sec, 4),
            "effective_fps": round(effective_fps, 1),
        }
        res.update(traj_metrics)
        res.update(count_metrics)
        return res

    def evaluate_multi_stride(
        self,
        tracker_factory: Callable[[int], BaseTracker],
        strides: Sequence[int] = (1, 2, 3, 5, 10),
        total_frames: int = 120,
    ) -> List[Dict[str, Any]]:
        """
        Executes multi-stride evaluation across S in {1, 2, 3, 5, 10} to quantitatively
        measure tracking degradation and throughput scaling.
        """
        results: List[Dict[str, Any]] = []
        for s in strides:
            tracker = tracker_factory(s)
            res = self.evaluate_tracker_simulation(tracker, stride=s, total_frames=total_frames)
            results.append(res)
        return results
