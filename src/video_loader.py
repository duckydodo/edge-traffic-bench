"""
Deterministic video and synthetic frame loader with configurable frame stride and letterbox preprocessing.
Includes TrafficPhysicsSimulator for multi-vehicle multi-rate tracker evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.inference import Detection


def letterbox(
    im: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Standard YOLO letterbox resizing with aspect ratio preservation and symmetric padding."""
    shape = im.shape[:2]
    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)

    r = min(target_shape[0] / shape[0], target_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = target_shape[1] - new_unpad[0], target_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (target_shape[1], target_shape[0])
        r = target_shape[1] / shape[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess_frame(
    frame: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    bgr_to_rgb: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Performs standard vision model preprocessing."""
    padded, _, _ = letterbox(frame, target_shape=target_shape)

    if bgr_to_rgb and padded.ndim == 3 and padded.shape[2] == 3:
        padded = padded[:, :, ::-1]

    chw = padded.transpose((2, 0, 1))
    contiguous = np.ascontiguousarray(chw)

    if normalize:
        tensor = contiguous.astype(np.float32) / 255.0
    else:
        tensor = contiguous

    return np.expand_dims(tensor, axis=0)


class SyntheticVideoGenerator:
    """Deterministic synthetic frame generator for reproducible benchmarks and unit tests."""

    VALID_STRIDES = {1, 2, 3, 5, 10}

    def __init__(
        self,
        total_frames: int = 200,
        width: int = 1920,
        height: int = 1080,
        stride: int = 1,
        seed: int = 42,
    ) -> None:
        if stride not in self.VALID_STRIDES:
            raise ValueError(f"Stride {stride} not supported. Must be one of {self.VALID_STRIDES}.")
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.stride = stride
        self.seed = seed

    def __len__(self) -> int:
        return (self.total_frames + self.stride - 1) // self.stride

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.height, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        base_canvas = np.stack([xx, yy, ((xx.astype(int) + yy.astype(int)) // 2).astype(np.uint8)], axis=-1)

        for raw_idx in range(self.total_frames):
            if raw_idx % self.stride != 0:
                continue

            frame = base_canvas.copy()
            for obj_id in range(5):
                box_w, box_h = 120, 80
                speed_x = (obj_id + 1) * 7
                speed_y = (obj_id + 1) * 3
                start_x = (obj_id * 350 + raw_idx * speed_x) % (self.width - box_w)
                start_y = (200 + obj_id * 150 + raw_idx * speed_y) % (self.height - box_h)
                color = (
                    int((obj_id * 60 + 50) % 256),
                    int((obj_id * 90 + 100) % 256),
                    int((obj_id * 130 + 150) % 256),
                )
                cv2.rectangle(frame, (start_x, start_y), (start_x + box_w, start_y + box_h), color, -1)

            yield raw_idx, frame


class TrafficPhysicsSimulator:
    """
    Deterministic multi-vehicle synthetic traffic physics simulator.
    Simulates:
      - Large slow buses/trucks on linear trajectories.
      - Small fast motorcycles executing lateral S-curve lane-weaving.
      - Brief occlusion behind the bus.
      - Known ground-truth line-crossing counts across a virtual counting line.
    """

    def __init__(
        self,
        total_frames: int = 120,
        width: int = 1280,
        height: int = 720,
        stride: int = 1,
        counting_line_y: float = 360.0,
    ) -> None:
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.stride = stride
        self.counting_line_y = counting_line_y

        # Define 5 simulated vehicles with exact physical properties
        self.vehicles = [
            {
                "id": 1,
                "class_id": 5,
                "class_name": "bus",
                "w": 90,
                "h": 180,
                "color": (20, 120, 240),  # Orange/amber
                "start_y": -180.0,
                "vy": 6.0,  # Moves down -> inbound
                "x_fn": lambda t: 360.0,
                "direction": "inbound",
            },
            {
                "id": 2,
                "class_id": 3,
                "class_name": "motorcycle",
                "w": 28,
                "h": 52,
                "color": (0, 0, 240),  # Red
                "start_y": -60.0,
                "vy": 12.0,  # Fast moves down -> inbound
                # S-curve lateral weaving
                "x_fn": lambda t: 390.0 + 75.0 * np.sin(0.12 * t),
                "direction": "inbound",
                "occlusion_frames": (18, 22),  # Brief occlusion behind bus
            },
            {
                "id": 3,
                "class_id": 2,
                "class_name": "car",
                "w": 65,
                "h": 110,
                "color": (0, 220, 0),  # Green
                "start_y": 750.0,
                "vy": -8.0,  # Moves up -> outbound
                "x_fn": lambda t: 700.0,
                "direction": "outbound",
            },
            {
                "id": 4,
                "class_id": 3,
                "class_name": "motorcycle",
                "w": 28,
                "h": 52,
                "color": (0, 240, 240),  # Yellow
                "start_y": 740.0,
                "vy": -13.0,  # Fast moves up -> outbound
                # S-curve lateral weaving
                "x_fn": lambda t: 760.0 + 65.0 * np.cos(0.14 * t),
                "direction": "outbound",
            },
            {
                "id": 5,
                "class_id": 7,
                "class_name": "truck",
                "w": 85,
                "h": 160,
                "color": (130, 130, 130),  # Grey
                "start_y": -160.0,
                "vy": 5.0,  # Moves down -> inbound
                "x_fn": lambda t: 920.0,
                "direction": "inbound",
            },
        ]

        # Exact ground truth summary across total_frames
        self.ground_truth_crossings = {
            "total_inbound": 3,
            "total_outbound": 2,
            "total_crossings": 5,
            "by_class": {
                "bus": {"inbound": 1, "outbound": 0, "total": 1},
                "motorcycle": {"inbound": 1, "outbound": 1, "total": 2},
                "car": {"inbound": 0, "outbound": 1, "total": 1},
                "truck": {"inbound": 1, "outbound": 0, "total": 1},
            },
        }

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, List[Detection]]]:
        """
        Yields (frame_idx, frame_bgr, visible_detections) for each strided frame.
        """
        # Base road canvas (asphalt grey with lane markings)
        road_canvas = np.full((self.height, self.width, 3), 45, dtype=np.uint8)
        # Lane divider lines
        for lx in (300, 600, 850, 1100):
            for ly in range(0, self.height, 40):
                cv2.line(road_canvas, (lx, ly), (lx, ly + 20), (200, 200, 200), 2)

        for t in range(self.total_frames):
            if t % self.stride != 0:
                continue

            frame = road_canvas.copy()
            detections: List[Detection] = []

            for veh in self.vehicles:
                # Compute position at frame t
                cy = veh["start_y"] + veh["vy"] * t
                cx = float(veh["x_fn"](t))
                w, h = veh["w"], veh["h"]
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0

                # Check if vehicle is inside canvas
                if y2 < 0 or y1 > self.height or x2 < 0 or x1 > self.width:
                    continue

                # Check occlusion window
                is_occluded = False
                if "occlusion_frames" in veh:
                    f_start, f_end = veh["occlusion_frames"]
                    if f_start <= t <= f_end:
                        is_occluded = True

                # Draw on frame
                box_int = (int(x1), int(y1), int(x2), int(y2))
                cv2.rectangle(frame, (box_int[0], box_int[1]), (box_int[2], box_int[3]), veh["color"], -1)
                # Headlight / texture details
                cv2.rectangle(frame, (box_int[0] + 4, box_int[1] + 4), (box_int[2] - 4, box_int[3] - 4), (255, 255, 255), 1)

                # If not occluded, emit detection
                if not is_occluded:
                    detections.append(
                        Detection(
                            box=(float(x1), float(y1), float(x2), float(y2)),
                            score=0.92,
                            class_id=veh["class_id"],
                            class_name=veh["class_name"],
                            track_id=veh["id"],
                        )
                    )

            yield t, frame, detections

    def get_ground_truth_crossings(self) -> Dict[str, Any]:
        """Returns exact ground-truth directional and class-specific line crossings."""
        return {
            "total_crossings": 5,
            "total_inbound": 3,
            "total_outbound": 2,
            "by_class": {
                "bus": {"inbound": 1, "outbound": 0, "total": 1},
                "motorcycle": {"inbound": 1, "outbound": 1, "total": 2},
                "car": {"inbound": 0, "outbound": 1, "total": 1},
                "truck": {"inbound": 1, "outbound": 0, "total": 1},
            },
        }

    def get_ground_truth_tracks(self) -> Dict[int, List[Dict[str, Any]]]:
        """Returns ground-truth active tracks for each sampled frame index."""
        gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for t, _, dets in self:
            gt_by_frame[t] = [
                {
                    "track_id": d.track_id,
                    "box": d.box,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
                for d in dets
            ]
        return gt_by_frame


class VideoStreamLoader:
    """Video stream loader supporting real video files and synthetic fallback."""

    VALID_STRIDES = {1, 2, 3, 5, 10}

    def __init__(
        self,
        video_path: Optional[Union[str, Path]] = None,
        stride: int = 1,
        max_frames: Optional[int] = None,
        synthetic_width: int = 1920,
        synthetic_height: int = 1080,
    ) -> None:
        if stride not in self.VALID_STRIDES:
            raise ValueError(f"Stride {stride} not supported. Must be one of {self.VALID_STRIDES}.")
        self.video_path = Path(video_path) if video_path else None
        self.stride = stride
        self.max_frames = max_frames
        self.synthetic_width = synthetic_width
        self.synthetic_height = synthetic_height

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        if self.video_path is not None and self.video_path.exists():
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {self.video_path}")
            
            raw_idx = 0
            yielded_count = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if raw_idx % self.stride == 0:
                        yield raw_idx, frame
                        yielded_count += 1
                        if self.max_frames and yielded_count >= self.max_frames:
                            break
                    raw_idx += 1
            finally:
                cap.release()
        else:
            total_frames = self.max_frames * self.stride if self.max_frames else 200
            gen = SyntheticVideoGenerator(
                total_frames=total_frames,
                width=self.synthetic_width,
                height=self.synthetic_height,
                stride=self.stride,
            )
            yielded_count = 0
            for raw_idx, frame in gen:
                yield raw_idx, frame
                yielded_count += 1
                if self.max_frames and yielded_count >= self.max_frames:
                    break
