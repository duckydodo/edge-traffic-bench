"""
Unified inference runtime engines and vectorized post-processing.
Supports PyTorch, ONNX Runtime (FP32, QDQ INT8, Tier-S), and OpenVINO IR
strictly conforming to hardware thread constraints.
Includes support for both One-to-Many (YOLO11) and End-to-End NMS-Free (YOLO26) formats.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import openvino as ov
import onnxruntime as ort
import torch

TRAFFIC_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "box": [round(float(c), 2) for c in self.box],
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }
        if self.track_id is not None:
            d["track_id"] = int(self.track_id)
        return d


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Vectorized class-aware Non-Maximum Suppression (NMS).
    Offsets box coordinates by class_id * max_coordinate to prevent inter-class suppression.
    """
    if len(boxes) == 0:
        return []

    # Large offset to perform class-independent NMS in a single vectorized pass
    max_coord = 10000.0
    offset = class_ids.astype(boxes.dtype) * max_coord
    b = boxes + offset[:, None]

    x1 = b[:, 0]
    y1 = b[:, 1]
    x2 = b[:, 2]
    y2 = b[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess_yolo(
    raw_predictions: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    filter_traffic_classes: bool = True,
) -> List[Detection]:
    """
    Vectorized NumPy post-processing for YOLO outputs.
    Automatically detects and handles:
      1. End-to-End One-to-One NMS-free format: (1, 300, 6) or (N, 6)
         columns: [x1, y1, x2, y2, score, class_id] -> Bypasses NMS entirely.
      2. Standard One-to-Many format: (1, 84, N) or (N, 84)
         columns: [cx, cy, w, h] + 80 class scores -> Vectorized conversion + NMS.
    """
    preds = np.asarray(raw_predictions)
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]  # (84, N) or (N, 84) or (N, 6)

    # 1. Detection of End-to-End One-to-One NMS-free format (YOLO26 / end2end)
    if preds.ndim == 2 and preds.shape[1] == 6:
        boxes_xyxy = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(int)

        mask = scores >= conf_threshold
        if filter_traffic_classes:
            traffic_mask = np.isin(class_ids, list(TRAFFIC_CLASSES.keys()))
            mask = mask & traffic_mask

        if not np.any(mask):
            return []

        filtered_boxes = boxes_xyxy[mask]
        filtered_scores = scores[mask]
        filtered_classes = class_ids[mask]

        detections: List[Detection] = []
        for i in range(len(filtered_scores)):
            cid = int(filtered_classes[i])
            cname = TRAFFIC_CLASSES.get(cid, f"class_{cid}")
            b = tuple(filtered_boxes[i].tolist())
            detections.append(
                Detection(
                    box=b,  # type: ignore
                    score=float(filtered_scores[i]),
                    class_id=cid,
                    class_name=cname,
                )
            )
        return detections

    # 2. Standard One-to-Many format (YOLO11)
    if preds.shape[0] == 84 or (preds.ndim == 2 and preds.shape[0] < preds.shape[1]):
        preds = preds.T  # (N, 84)

    if preds.shape[1] < 5:
        return []

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]  # (N, num_classes)

    class_ids = np.argmax(class_scores, axis=-1)
    scores = np.max(class_scores, axis=-1)

    mask = scores >= conf_threshold
    if filter_traffic_classes:
        traffic_mask = np.isin(class_ids, list(TRAFFIC_CLASSES.keys()))
        mask = mask & traffic_mask

    if not np.any(mask):
        return []

    filtered_boxes_xywh = boxes_xywh[mask]
    filtered_scores = scores[mask]
    filtered_classes = class_ids[mask]

    # Vectorized conversion: (cx, cy, w, h) -> (x1, y1, x2, y2)
    cx = filtered_boxes_xywh[:, 0]
    cy = filtered_boxes_xywh[:, 1]
    w = filtered_boxes_xywh[:, 2]
    h = filtered_boxes_xywh[:, 3]

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)

    # Class-aware NMS
    keep_indices = nms_numpy(
        boxes_xyxy,
        filtered_scores,
        filtered_classes,
        iou_threshold=iou_threshold,
    )

    detections = []
    for idx in keep_indices:
        cid = int(filtered_classes[idx])
        cname = TRAFFIC_CLASSES.get(cid, f"class_{cid}")
        b = tuple(boxes_xyxy[idx].tolist())
        detections.append(
            Detection(
                box=b,  # type: ignore
                score=float(filtered_scores[idx]),
                class_id=cid,
                class_name=cname,
            )
        )

    return detections


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two flattened output arrays."""
    flat_a = a.flatten().astype(np.float64)
    flat_b = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(flat_a, flat_b) / (norm_a * norm_b))


def compute_anchor_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> float:
    """
    Computes mean Intersection over Union (IoU) between two sets of bounding boxes.
    Boxes are expected to be (N, 4) in (cx, cy, w, h).
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return 0.0

    b1 = np.asarray(boxes1, dtype=np.float64)
    b2 = np.asarray(boxes2, dtype=np.float64)

    x1_1 = b1[:, 0] - b1[:, 2] / 2.0
    y1_1 = b1[:, 1] - b1[:, 3] / 2.0
    x2_1 = b1[:, 0] + b1[:, 2] / 2.0
    y2_1 = b1[:, 1] + b1[:, 3] / 2.0

    x1_2 = b2[:, 0] - b2[:, 2] / 2.0
    y1_2 = b2[:, 1] - b2[:, 3] / 2.0
    x2_2 = b2[:, 0] + b2[:, 2] / 2.0
    y2_2 = b2[:, 1] + b2[:, 3] / 2.0

    inter_x1 = np.maximum(x1_1, x1_2)
    inter_y1 = np.maximum(y1_1, y1_2)
    inter_x2 = np.minimum(x2_1, x2_2)
    inter_y2 = np.minimum(y2_1, y2_2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, x2_1 - x1_1) * np.maximum(0.0, y2_1 - y1_1)
    area2 = np.maximum(0.0, x2_2 - x1_2) * np.maximum(0.0, y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 1.0)
    return float(np.mean(iou))


class BaseInferenceEngine(abc.ABC):
    """Abstract base class for CPU inference runtime engines."""

    def __init__(self, model_path: Union[str, Path], num_threads: int = 1) -> None:
        self.model_path = Path(model_path)
        self.num_threads = num_threads
        self.is_loaded = False

    @abc.abstractmethod
    def load(self) -> None:
        """Loads and initializes the model runtime engine."""
        pass

    @abc.abstractmethod
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """Runs raw forward pass on float32 tensor of shape (1, 3, H, W)."""
        pass

    def predict(
        self,
        input_tensor: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """Runs forward pass and post-processes detections."""
        raw = self.forward(input_tensor)
        return postprocess_yolo(
            raw,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )


class PyTorchEngine(BaseInferenceEngine):
    """PyTorch CPU inference engine using TorchScript or Eager mode."""

    def __init__(self, model_path: Union[str, Path], num_threads: int = 1) -> None:
        super().__init__(model_path, num_threads)
        self.model: Optional[Any] = None

    def load(self) -> None:
        torch.set_num_threads(self.num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        from ultralytics import YOLO
        yolo_wrapper = YOLO(str(self.model_path))
        self.model = yolo_wrapper.model
        self.model.eval()
        self.is_loaded = True

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        if not self.is_loaded or self.model is None:
            self.load()

        with torch.no_grad():
            t_input = torch.from_numpy(input_tensor).float()
            preds = self.model(t_input)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            if hasattr(preds, "cpu"):
                preds = preds.cpu().numpy()
            return np.asarray(preds)


class OnnxRuntimeEngine(BaseInferenceEngine):
    """
    ONNX Runtime CPU inference engine with intra/inter thread tuning and
    pre-allocated IOBinding to minimize p99 tail-latency jitter.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        num_threads: int = 1,
        use_io_binding: bool = True,
    ) -> None:
        super().__init__(model_path, num_threads)
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "images"
        self.output_names: List[str] = []
        self.use_io_binding = use_io_binding
        self.io_binding: Optional[ort.IOBinding] = None

    def load(self) -> None:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        if self.use_io_binding:
            try:
                self.io_binding = self.session.io_binding()
            except Exception:
                self.io_binding = None

        self.is_loaded = True

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        if not self.is_loaded or self.session is None:
            self.load()

        tensor_f32 = np.ascontiguousarray(input_tensor, dtype=np.float32)

        if self.use_io_binding and self.io_binding is not None:
            self.io_binding.bind_cpu_input(self.input_name, tensor_f32)
            for out_name in self.output_names:
                self.io_binding.bind_output(out_name, "cpu")
            self.session.run_with_iobinding(self.io_binding)
            outputs = self.io_binding.copy_outputs_to_cpu()
            return np.asarray(outputs[0])

        outputs = self.session.run(None, {self.input_name: tensor_f32})
        return np.asarray(outputs[0])


class OpenVINOEngine(BaseInferenceEngine):
    """OpenVINO CPU inference engine with latency hint and stream caps."""

    def __init__(self, model_path: Union[str, Path], num_threads: int = 1) -> None:
        super().__init__(model_path, num_threads)
        self.compiled_model: Optional[Any] = None
        self.infer_request: Optional[Any] = None

    def load(self) -> None:
        core = ov.Core()
        xml_path = self.model_path if self.model_path.suffix == ".xml" else (self.model_path / "model.xml")
        if not xml_path.exists():
            raise FileNotFoundError(f"OpenVINO model file not found at: {xml_path}")

        ov_model = core.read_model(str(xml_path))
        config = {
            "INFERENCE_NUM_THREADS": str(self.num_threads),
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
        }
        self.compiled_model = core.compile_model(ov_model, "CPU", config=config)
        self.infer_request = self.compiled_model.create_infer_request()
        self.is_loaded = True

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        if not self.is_loaded or self.infer_request is None:
            self.load()

        tensor_f32 = np.ascontiguousarray(input_tensor, dtype=np.float32)
        results = self.infer_request.infer([tensor_f32])
        first_output = list(results.values())[0]
        return np.asarray(first_output)
