"""
Multi-Rate Multi-Object Trackers for Edge CPU execution.
Contains:
  1. ByteTracker: Two-stage low-confidence association with stride-aware Kalman Filter.
  2. OCSortTracker: Observation-Centric SORT with OCM, ORU, and OCR for non-linear motion.
  3. CBIoUTracker: Cascaded Buffered-IoU + Fast 24-bin HSV Histogram gate for S >= 3.
Zero external neural Re-ID dependencies; fast vectorized NumPy / SciPy implementation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.inference import Detection


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU matrix between boxes_a (N, 4) and boxes_b (M, 4).
    Boxes in format [x1, y1, x2, y2].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    x1_b, y1_b, x2_b, y2_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(x1_a[:, None], x1_b[None, :])
    inter_y1 = np.maximum(y1_a[:, None], y1_b[None, :])
    inter_x2 = np.minimum(x2_a[:, None], x2_b[None, :])
    inter_y2 = np.minimum(y2_a[:, None], y2_b[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, x2_a - x1_a) * np.maximum(0.0, y2_a - y1_a)
    area_b = np.maximum(0.0, x2_b - x1_b) * np.maximum(0.0, y2_b - y1_b)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def buffer_boxes(boxes: np.ndarray, buffer_scale: float) -> np.ndarray:
    """
    Expands bounding boxes by buffer_scale * [w, h] to enable association across
    large inter-frame displacements where standard IoU is 0.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = np.asarray(boxes, dtype=np.float32).copy()
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    b[:, 0] -= buffer_scale * w
    b[:, 1] -= buffer_scale * h
    b[:, 2] += buffer_scale * w
    b[:, 3] += buffer_scale * h
    return b


def extract_hsv_histogram_24bin(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts an ultra-fast (<0.1ms) 24-bin HSV color histogram (8H, 8S, 8V)
    with channel-wise normalization and Hue weighting for appearance verification.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
        return np.ones(24, dtype=np.float32) / np.sqrt(24.0)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # 8 bins for H [0, 180], 8 bins for S [0, 256], 8 bins for V [0, 256]
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    norm_h = np.linalg.norm(hist_h)
    norm_s = np.linalg.norm(hist_s)
    norm_v = np.linalg.norm(hist_v)
    if norm_h > 0:
        hist_h /= norm_h
    if norm_s > 0:
        hist_s /= norm_s
    if norm_v > 0:
        hist_v /= norm_v

    # Weight Hue by 2.0 to prioritize chromatic separation
    combined = np.concatenate([2.0 * hist_h, hist_s, hist_v]).astype(np.float32)
    norm_c = np.linalg.norm(combined)
    if norm_c > 0:
        combined /= norm_c
    else:
        combined = np.ones(24, dtype=np.float32) / np.sqrt(24.0)
    return combined


@dataclass
class Track:
    track_id: int
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    score: float
    state: str = "confirmed"  # "tentative", "confirmed", "lost"
    history: List[Tuple[float, float]] = field(default_factory=list)  # [(cx, cy), ...]
    time_since_update: int = 0
    hits: int = 1
    age: int = 1

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            (self.box[0] + self.box[2]) / 2.0,
            (self.box[1] + self.box[3]) / 2.0,
        )


class KalmanFilter2D:
    """
    Stride-aware 8-dimensional Kalman filter for 2D bounding boxes.
    State: [x, y, a, h, vx, vy, va, vh]^T
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = dt
        # State transition matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        # Measurement matrix H
        self.H = np.eye(4, 8, dtype=np.float32)

        # Standard deviations
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Measurement: [x, y, a, h]"""
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel]))).astype(np.float32)

        mean = np.dot(self.F, mean)
        covariance = np.linalg.multi_dot([self.F, covariance, self.F.T]) + motion_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean = np.dot(self.H, mean)
        projected_cov = np.linalg.multi_dot([self.H, covariance, self.H.T])

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        meas_cov = np.diag(np.square(std)).astype(np.float32)
        projected_cov += meas_cov

        # Kalman gain K = covariance * H^T * inv(projected_cov)
        try:
            kalman_gain = np.linalg.multi_dot([
                covariance,
                self.H.T,
                np.linalg.inv(projected_cov),
            ])
        except np.linalg.LinAlgError:
            kalman_gain = np.linalg.multi_dot([
                covariance,
                self.H.T,
                np.linalg.pinv(projected_cov),
            ])

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_cov = covariance - np.linalg.multi_dot([kalman_gain, projected_cov, kalman_gain.T])
        return new_mean, new_cov


def xyxy_to_xyah(box: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Converts [x1, y1, x2, y2] to [x_center, y_center, aspect_ratio, height]."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    x = float(x1 + w / 2.0)
    y = float(y1 + h / 2.0)
    a = float(w / h)
    return np.array([x, y, a, h], dtype=np.float32)


def xyah_to_xyxy(state: Union[Sequence[float], np.ndarray]) -> Tuple[float, float, float, float]:
    """Converts [x_center, y_center, aspect_ratio, height] to [x1, y1, x2, y2]."""
    x, y, a, h = state[0], state[1], state[2], state[3]
    w = max(1.0, a * h)
    h = max(1.0, h)
    return (float(x - w / 2.0), float(y - h / 2.0), float(x + w / 2.0), float(y + h / 2.0))


class BaseTracker(abc.ABC):
    """Abstract base class for multi-object trackers."""

    def __init__(self, stride: int = 1, max_age: int = 30, min_hits: int = 3) -> None:
        self.stride = stride
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
        self._next_id = 1

    @abc.abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Processes current frame detections and returns active confirmed tracks."""
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets tracker state."""
        pass


class ByteTracker(BaseTracker):
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
    Implements two-stage association with high-confidence (0.5) and low-confidence (0.1) detections.
    """

    def __init__(
        self,
        stride: int = 1,
        max_age: int = 30,
        min_hits: int = 3,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.7,
    ) -> None:
        super().__init__(stride=stride, max_age=max_age, min_hits=min_hits)
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.kf = KalmanFilter2D(dt=float(stride))

        # Active internal tracks: dict[track_id, track_state]
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_count = 0
        self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[Track]:
        self.frame_count += 1

        # 1. Kalman Predict for existing tracks
        for tid, t in list(self.tracks.items()):
            mean, cov = self.kf.predict(t["mean"], t["cov"])
            t["mean"], t["cov"] = mean, cov
            t["time_since_update"] += 1
            t["age"] += 1
            t["box"] = xyah_to_xyxy(mean)

        # 2. Partition detections into high and low confidence
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []
        for d in detections:
            if d.score >= self.high_thresh:
                high_dets.append(d)
            elif d.score >= self.low_thresh:
                low_dets.append(d)

        # 3. Stage 1 Association: Confirmed/Active tracks with high_dets
        active_track_ids = [
            tid for tid, t in self.tracks.items()
            if t["time_since_update"] <= self.max_age
        ]
        track_boxes = np.array([self.tracks[tid]["box"] for tid in active_track_ids], dtype=np.float32)
        high_boxes = np.array([d.box for d in high_dets], dtype=np.float32)

        matched_tracks_1: Set[int] = set()
        matched_high_dets: Set[int] = set()

        if len(track_boxes) > 0 and len(high_boxes) > 0:
            iou_mat = box_iou_matrix(track_boxes, high_boxes)
            cost_mat = 1.0 - iou_mat
            row_inds, col_inds = linear_sum_assignment(cost_mat)

            for r, c in zip(row_inds, col_inds):
                if cost_mat[r, c] < self.match_thresh:
                    tid = active_track_ids[r]
                    matched_tracks_1.add(tid)
                    matched_high_dets.add(c)
                    det = high_dets[c]
                    # Update Kalman filter
                    meas = xyxy_to_xyah(det.box)
                    mean, cov = self.kf.update(self.tracks[tid]["mean"], self.tracks[tid]["cov"], meas)
                    self.tracks[tid]["mean"] = mean
                    self.tracks[tid]["cov"] = cov
                    self.tracks[tid]["box"] = det.box
                    self.tracks[tid]["score"] = det.score
                    self.tracks[tid]["time_since_update"] = 0
                    self.tracks[tid]["hits"] += 1
                    cx, cy = (det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0
                    self.tracks[tid]["history"].append((cx, cy))
                    if len(self.tracks[tid]["history"]) > 30:
                        self.tracks[tid]["history"].pop(0)

        # 4. Stage 2 Association: Remaining unmatched active tracks with low_dets
        unmatched_track_ids = [tid for tid in active_track_ids if tid not in matched_tracks_1]
        unmatched_track_boxes = np.array([self.tracks[tid]["box"] for tid in unmatched_track_ids], dtype=np.float32)
        low_boxes = np.array([d.box for d in low_dets], dtype=np.float32)

        if len(unmatched_track_boxes) > 0 and len(low_boxes) > 0:
            iou_mat_2 = box_iou_matrix(unmatched_track_boxes, low_boxes)
            cost_mat_2 = 1.0 - iou_mat_2
            row_inds, col_inds = linear_sum_assignment(cost_mat_2)

            for r, c in zip(row_inds, col_inds):
                if cost_mat_2[r, c] < 0.6:  # Strict match threshold for low conf
                    tid = unmatched_track_ids[r]
                    det = low_dets[c]
                    meas = xyxy_to_xyah(det.box)
                    mean, cov = self.kf.update(self.tracks[tid]["mean"], self.tracks[tid]["cov"], meas)
                    self.tracks[tid]["mean"] = mean
                    self.tracks[tid]["cov"] = cov
                    self.tracks[tid]["box"] = det.box
                    self.tracks[tid]["score"] = det.score
                    self.tracks[tid]["time_since_update"] = 0
                    self.tracks[tid]["hits"] += 1
                    cx, cy = (det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0
                    self.tracks[tid]["history"].append((cx, cy))
                    if len(self.tracks[tid]["history"]) > 30:
                        self.tracks[tid]["history"].pop(0)

        # 5. Initialize new tracks for unmatched high_dets
        for c, d in enumerate(high_dets):
            if c not in matched_high_dets:
                meas = xyxy_to_xyah(d.box)
                mean, cov = self.kf.initiate(meas)
                tid = self._next_id
                self._next_id += 1
                cx, cy = (d.box[0] + d.box[2]) / 2.0, (d.box[1] + d.box[3]) / 2.0
                self.tracks[tid] = {
                    "mean": mean,
                    "cov": cov,
                    "box": d.box,
                    "score": d.score,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "hits": 1,
                    "age": 1,
                    "time_since_update": 0,
                    "history": [(cx, cy)],
                }

        # 6. Cull dead tracks and return confirmed tracks
        output_tracks: List[Track] = []
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            if t["time_since_update"] > self.max_age:
                del self.tracks[tid]
                continue
            if t["hits"] >= self.min_hits or self.frame_count <= self.min_hits:
                state = "confirmed" if t["time_since_update"] == 0 else "lost"
                output_tracks.append(
                    Track(
                        track_id=tid,
                        box=t["box"],
                        class_id=t["class_id"],
                        class_name=t["class_name"],
                        score=t["score"],
                        state=state,
                        history=list(t["history"]),
                        time_since_update=t["time_since_update"],
                        hits=t["hits"],
                        age=t["age"],
                    )
                )

        return output_tracks


class OCSortTracker(BaseTracker):
    """
    Observation-Centric SORT (OC-SORT).
    Equipped with:
      1. OCM (Observation-Centric Momentum): Directional velocity consistency cost.
      2. ORU (Observation-Centric Re-Update): Virtual interpolated re-update upon re-association.
      3. OCR (Observation-Centric Recovery): Fallback matching against last raw observation.
    """

    def __init__(
        self,
        stride: int = 1,
        max_age: int = 30,
        min_hits: int = 3,
        iou_thresh: float = 0.3,
        lambda_ocm: float = 0.25,
        delta_t_ocm: int = 3,
    ) -> None:
        super().__init__(stride=stride, max_age=max_age, min_hits=min_hits)
        self.iou_thresh = iou_thresh
        self.lambda_ocm = lambda_ocm
        self.delta_t_ocm = delta_t_ocm
        self.kf = KalmanFilter2D(dt=float(stride))
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_count = 0
        self._next_id = 1

    def _compute_ocm_cost(
        self,
        track_obs_history: List[Tuple[int, Tuple[float, float]]],
        det_boxes: np.ndarray,
    ) -> np.ndarray:
        """Computes directional velocity consistency cost Cv in [0, 1]."""
        num_dets = len(det_boxes)
        if len(track_obs_history) < 2 or num_dets == 0:
            return np.zeros(num_dets, dtype=np.float32)

        # Calculate historical velocity vector
        last_f, last_p = track_obs_history[-1]
        idx = max(0, len(track_obs_history) - 1 - self.delta_t_ocm)
        prev_f, prev_p = track_obs_history[idx]

        df = max(1, last_f - prev_f)
        vx = (last_p[0] - prev_p[0]) / df
        vy = (last_p[1] - prev_p[1]) / df
        track_angle = np.arctan2(vy, vx)

        det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
        det_dx = det_cx - last_p[0]
        det_dy = det_cy - last_p[1]
        det_angles = np.arctan2(det_dy, det_dx)

        angle_diff = np.abs(track_angle - det_angles)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        cv_cost = (1.0 - np.cos(angle_diff)) / 2.0
        return cv_cost.astype(np.float32)

    def update(
        self,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[Track]:
        self.frame_count += 1

        # 1. Kalman Predict
        for tid, t in list(self.tracks.items()):
            mean, cov = self.kf.predict(t["mean"], t["cov"])
            t["mean"], t["cov"] = mean, cov
            t["time_since_update"] += 1
            t["age"] += 1
            t["box"] = xyah_to_xyxy(mean)

        active_track_ids = [
            tid for tid, t in self.tracks.items()
            if t["time_since_update"] <= self.max_age
        ]
        det_boxes = np.array([d.box for d in detections], dtype=np.float32)

        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()

        # 2. Stage 1 Association with OCM (IoU + Observation Momentum)
        if len(active_track_ids) > 0 and len(det_boxes) > 0:
            track_boxes = np.array([self.tracks[tid]["box"] for tid in active_track_ids], dtype=np.float32)
            iou_mat = box_iou_matrix(track_boxes, det_boxes)
            cost_mat = 1.0 - iou_mat

            for r, tid in enumerate(active_track_ids):
                cv_cost = self._compute_ocm_cost(self.tracks[tid]["observations"], det_boxes)
                cost_mat[r] += self.lambda_ocm * cv_cost

            row_inds, col_inds = linear_sum_assignment(cost_mat)
            for r, c in zip(row_inds, col_inds):
                if iou_mat[r, c] >= self.iou_thresh:
                    tid = active_track_ids[r]
                    matched_tracks.add(tid)
                    matched_dets.add(c)
                    self._apply_match(tid, detections[c])

        # 3. Stage 2: OCR (Observation-Centric Recovery) with Centroid-Distance-Assisted DIoU
        unmatched_track_ids = [tid for tid in active_track_ids if tid not in matched_tracks]
        unmatched_det_indices = [c for c in range(len(detections)) if c not in matched_dets]

        if len(unmatched_track_ids) > 0 and len(unmatched_det_indices) > 0:
            last_raw_boxes = np.array([
                self.tracks[tid]["last_raw_box"] for tid in unmatched_track_ids
            ], dtype=np.float32)
            rem_det_boxes = det_boxes[unmatched_det_indices]

            ocr_iou_mat = box_iou_matrix(last_raw_boxes, rem_det_boxes)

            # Center distance squared
            last_cx = (last_raw_boxes[:, 0] + last_raw_boxes[:, 2]) / 2.0
            last_cy = (last_raw_boxes[:, 1] + last_raw_boxes[:, 3]) / 2.0
            det_cx = (rem_det_boxes[:, 0] + rem_det_boxes[:, 2]) / 2.0
            det_cy = (rem_det_boxes[:, 1] + rem_det_boxes[:, 3]) / 2.0
            dist_sq = (last_cx[:, None] - det_cx[None, :]) ** 2 + (last_cy[:, None] - det_cy[None, :]) ** 2

            # Smallest enclosing box diagonal squared
            enc_x1 = np.minimum(last_raw_boxes[:, 0][:, None], rem_det_boxes[:, 0][None, :])
            enc_y1 = np.minimum(last_raw_boxes[:, 1][:, None], rem_det_boxes[:, 1][None, :])
            enc_x2 = np.maximum(last_raw_boxes[:, 2][:, None], rem_det_boxes[:, 2][None, :])
            enc_y2 = np.maximum(last_raw_boxes[:, 3][:, None], rem_det_boxes[:, 3][None, :])
            diag_sq = np.maximum(1.0, (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2)

            diou_mat = ocr_iou_mat - (dist_sq / diag_sq)
            ocr_cost_mat = 1.0 - diou_mat
            row_inds, col_inds = linear_sum_assignment(ocr_cost_mat)

            for r, c in zip(row_inds, col_inds):
                # Match if DIoU is above -0.2 (allowing moderate stride center shifts) OR raw IoU >= 0.15
                if diou_mat[r, c] >= -0.2 or ocr_iou_mat[r, c] >= 0.15:
                    tid = unmatched_track_ids[r]
                    det_idx = unmatched_det_indices[c]
                    matched_tracks.add(tid)
                    matched_dets.add(det_idx)
                    self._apply_match(tid, detections[det_idx])

        # 4. Initialize new tracks for unmatched detections
        for c, d in enumerate(detections):
            if c not in matched_dets and d.score >= 0.3:
                meas = xyxy_to_xyah(d.box)
                mean, cov = self.kf.initiate(meas)
                tid = self._next_id
                self._next_id += 1
                cx, cy = (d.box[0] + d.box[2]) / 2.0, (d.box[1] + d.box[3]) / 2.0
                self.tracks[tid] = {
                    "mean": mean,
                    "cov": cov,
                    "box": d.box,
                    "last_raw_box": d.box,
                    "score": d.score,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "hits": 1,
                    "age": 1,
                    "time_since_update": 0,
                    "last_updated_frame": self.frame_count,
                    "observations": [(self.frame_count, (cx, cy))],
                    "history": [(cx, cy)],
                }

        # 5. Cull dead tracks and return confirmed output
        output_tracks: List[Track] = []
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            if t["time_since_update"] > self.max_age:
                del self.tracks[tid]
                continue
            if t["hits"] >= self.min_hits or self.frame_count <= self.min_hits:
                state = "confirmed" if t["time_since_update"] == 0 else "lost"
                output_tracks.append(
                    Track(
                        track_id=tid,
                        box=t["box"],
                        class_id=t["class_id"],
                        class_name=t["class_name"],
                        score=t["score"],
                        state=state,
                        history=list(t["history"]),
                        time_since_update=t["time_since_update"],
                        hits=t["hits"],
                        age=t["age"],
                    )
                )

        return output_tracks

    def _apply_match(self, tid: int, det: Detection) -> None:
        """Applies detection update with ORU (Observation-Centric Re-Update)."""
        t = self.tracks[tid]
        gap = t["time_since_update"]

        # ORU: If track was lost for multiple frames, perform virtual re-update
        if gap >= 2 and len(t["observations"]) > 0:
            last_frame, (last_cx, last_cy) = t["observations"][-1]
            last_box = t["last_raw_box"]
            curr_box = det.box

            # Linearly interpolate virtual observations across the skipped frames
            for step in range(1, gap):
                alpha = step / float(gap)
                v_box = (
                    last_box[0] + alpha * (curr_box[0] - last_box[0]),
                    last_box[1] + alpha * (curr_box[1] - last_box[1]),
                    last_box[2] + alpha * (curr_box[2] - last_box[2]),
                    last_box[3] + alpha * (curr_box[3] - last_box[3]),
                )
                v_meas = xyxy_to_xyah(v_box)
                mean, cov = self.kf.predict(t["mean"], t["cov"])
                t["mean"], t["cov"] = self.kf.update(mean, cov, v_meas)

        # Standard measurement update
        meas = xyxy_to_xyah(det.box)
        mean, cov = self.kf.update(t["mean"], t["cov"], meas)
        t["mean"] = mean
        t["cov"] = cov
        t["box"] = det.box
        t["last_raw_box"] = det.box
        t["score"] = det.score
        t["time_since_update"] = 0
        t["hits"] += 1
        t["last_updated_frame"] = self.frame_count

        cx, cy = (det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0
        t["observations"].append((self.frame_count, (cx, cy)))
        if len(t["observations"]) > 20:
            t["observations"].pop(0)

        t["history"].append((cx, cy))
        if len(t["history"]) > 30:
            t["history"].pop(0)


class CBIoUTracker(BaseTracker):
    """
    Cascaded Buffered-IoU Tracker with Fast 24-bin HSV Histogram Appearance Gate.
    Specialized for Stride S >= 3 where fast weaving two-wheelers have displacement > L_box.
    """

    def __init__(
        self,
        stride: int = 1,
        max_age: int = 30,
        min_hits: int = 3,
        b1: float = 0.3,
        b2: float = 0.5,
        hsv_weight: float = 0.25,
        hsv_reject_thresh: float = 0.75,
    ) -> None:
        super().__init__(stride=stride, max_age=max_age, min_hits=min_hits)
        # Dynamic stride-aware buffer scaling for large inter-frame displacement
        self.b1 = b1 * (1.0 + 0.35 * (stride - 1))
        self.b2 = b2 * (1.0 + 0.50 * (stride - 1))
        self.hsv_weight = hsv_weight
        self.hsv_reject_thresh = hsv_reject_thresh
        self.kf = KalmanFilter2D(dt=float(stride))
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_count = 0
        self._next_id = 1

    def _extract_crop_hist(
        self,
        box: Tuple[float, float, float, float],
        frame_bgr: Optional[np.ndarray],
    ) -> np.ndarray:
        if frame_bgr is None:
            return np.ones(24, dtype=np.float32) / np.sqrt(24.0)
        h_img, w_img = frame_bgr.shape[:2]
        x1 = max(0, min(w_img - 1, int(box[0])))
        y1 = max(0, min(h_img - 1, int(box[1])))
        x2 = max(x1 + 1, min(w_img, int(box[2])))
        y2 = max(y1 + 1, min(h_img, int(box[3])))
        crop = frame_bgr[y1:y2, x1:x2]
        return extract_hsv_histogram_24bin(crop)

    def update(
        self,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[Track]:
        self.frame_count += 1

        # 1. Kalman Predict
        for tid, t in list(self.tracks.items()):
            mean, cov = self.kf.predict(t["mean"], t["cov"])
            t["mean"], t["cov"] = mean, cov
            t["time_since_update"] += 1
            t["age"] += 1
            t["box"] = xyah_to_xyxy(mean)

        active_track_ids = [
            tid for tid, t in self.tracks.items()
            if t["time_since_update"] <= self.max_age
        ]
        det_boxes = np.array([d.box for d in detections], dtype=np.float32)

        # Extract HSV histograms for all detections
        det_hists = [
            self._extract_crop_hist(d.box, frame_bgr)
            for d in detections
        ]

        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()

        # 2. Pass 1 Association: Cascaded Buffer Scale b1 (0.3)
        if len(active_track_ids) > 0 and len(det_boxes) > 0:
            track_boxes = np.array([self.tracks[tid]["box"] for tid in active_track_ids], dtype=np.float32)
            # Expand both track boxes and det boxes by b1
            buf_track_boxes = buffer_boxes(track_boxes, self.b1)
            buf_det_boxes = buffer_boxes(det_boxes, self.b1)

            b_iou_mat = box_iou_matrix(buf_track_boxes, buf_det_boxes)
            cost_mat = 1.0 - b_iou_mat

            # Add HSV appearance distance if frame available
            for r, tid in enumerate(active_track_ids):
                track_hist = self.tracks[tid]["hsv_hist"]
                for c in range(len(det_boxes)):
                    cos_dist = float(1.0 - np.dot(track_hist, det_hists[c]))
                    if cos_dist > self.hsv_reject_thresh:
                        cost_mat[r, c] += 10.0  # Reject match
                    else:
                        cost_mat[r, c] += self.hsv_weight * cos_dist

            row_inds, col_inds = linear_sum_assignment(cost_mat)
            for r, c in zip(row_inds, col_inds):
                if cost_mat[r, c] < 0.7:  # Match valid under b1
                    tid = active_track_ids[r]
                    matched_tracks.add(tid)
                    matched_dets.add(c)
                    self._apply_match(tid, detections[c], det_hists[c])

        # 3. Pass 2 Association: Expanded Buffer Scale b2 (0.5) for high-stride recovery
        unmatched_track_ids = [tid for tid in active_track_ids if tid not in matched_tracks]
        unmatched_det_indices = [c for c in range(len(detections)) if c not in matched_dets]

        if len(unmatched_track_ids) > 0 and len(unmatched_det_indices) > 0:
            track_boxes_2 = np.array([self.tracks[tid]["box"] for tid in unmatched_track_ids], dtype=np.float32)
            det_boxes_2 = det_boxes[unmatched_det_indices]

            # Expand by b2 (0.5)
            buf_track_boxes_2 = buffer_boxes(track_boxes_2, self.b2)
            buf_det_boxes_2 = buffer_boxes(det_boxes_2, self.b2)

            b_iou_mat_2 = box_iou_matrix(buf_track_boxes_2, buf_det_boxes_2)
            cost_mat_2 = 1.0 - b_iou_mat_2

            for r, tid in enumerate(unmatched_track_ids):
                track_hist = self.tracks[tid]["hsv_hist"]
                for c_idx, orig_c in enumerate(unmatched_det_indices):
                    cos_dist = float(1.0 - np.dot(track_hist, det_hists[orig_c]))
                    if cos_dist > self.hsv_reject_thresh:
                        cost_mat_2[r, c_idx] += 10.0
                    else:
                        cost_mat_2[r, c_idx] += self.hsv_weight * cos_dist

            row_inds, col_inds = linear_sum_assignment(cost_mat_2)
            for r, c in zip(row_inds, col_inds):
                if cost_mat_2[r, c] < 0.75:
                    tid = unmatched_track_ids[r]
                    det_idx = unmatched_det_indices[c]
                    matched_tracks.add(tid)
                    matched_dets.add(det_idx)
                    self._apply_match(tid, detections[det_idx], det_hists[det_idx])

        # 4. Initialize new tracks
        for c, d in enumerate(detections):
            if c not in matched_dets and d.score >= 0.3:
                meas = xyxy_to_xyah(d.box)
                mean, cov = self.kf.initiate(meas)
                tid = self._next_id
                self._next_id += 1
                cx, cy = (d.box[0] + d.box[2]) / 2.0, (d.box[1] + d.box[3]) / 2.0
                self.tracks[tid] = {
                    "mean": mean,
                    "cov": cov,
                    "box": d.box,
                    "score": d.score,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "hsv_hist": det_hists[c],
                    "hits": 1,
                    "age": 1,
                    "time_since_update": 0,
                    "history": [(cx, cy)],
                }

        # 5. Output confirmed tracks
        output_tracks: List[Track] = []
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            if t["time_since_update"] > self.max_age:
                del self.tracks[tid]
                continue
            if t["hits"] >= self.min_hits or self.frame_count <= self.min_hits:
                state = "confirmed" if t["time_since_update"] == 0 else "lost"
                output_tracks.append(
                    Track(
                        track_id=tid,
                        box=t["box"],
                        class_id=t["class_id"],
                        class_name=t["class_name"],
                        score=t["score"],
                        state=state,
                        history=list(t["history"]),
                        time_since_update=t["time_since_update"],
                        hits=t["hits"],
                        age=t["age"],
                    )
                )

        return output_tracks

    def _apply_match(self, tid: int, det: Detection, hist: np.ndarray) -> None:
        t = self.tracks[tid]
        meas = xyxy_to_xyah(det.box)
        mean, cov = self.kf.update(t["mean"], t["cov"], meas)
        t["mean"] = mean
        t["cov"] = cov
        t["box"] = det.box
        t["score"] = det.score
        t["time_since_update"] = 0
        t["hits"] += 1
        # Smooth appearance histogram: 0.9 old + 0.1 new
        t["hsv_hist"] = 0.9 * t["hsv_hist"] + 0.1 * hist
        norm = np.linalg.norm(t["hsv_hist"])
        if norm > 0:
            t["hsv_hist"] /= norm

        cx, cy = (det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0
        t["history"].append((cx, cy))
        if len(t["history"]) > 30:
            t["history"].pop(0)
