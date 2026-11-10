"""
Virtual line-crossing traffic analytics and telemetry annotation.
Implements:
  1. LineCrossingCounter: Algebraic signed cross-product counter preventing double-count jitter.
  2. TrafficAnnotator: High-performance OpenCV visualization with telemetry dashboard overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.tracker import Track

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 255, 0),          # Green
    "motorcycle": (255, 255, 0),  # Cyan
    "bus": (0, 165, 255),        # Orange
    "truck": (0, 0, 255),        # Red
    "bicycle": (255, 0, 255),    # Magenta
    "person": (255, 200, 0),     # Yellow-cyan
}


def signed_cross_product(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p: Tuple[float, float],
) -> float:
    """
    Computes algebraic signed cross product of vector (P2 - P1) with (P - P1).
    s(P) = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
    """
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """Checks whether line segment P1-P2 intersects segment Q1-Q2."""
    s1 = signed_cross_product(p1, p2, q1)
    s2 = signed_cross_product(p1, p2, q2)
    s3 = signed_cross_product(q1, q2, p1)
    s4 = signed_cross_product(q1, q2, p2)

    cond1 = (s1 <= 0 and s2 >= 0) or (s1 >= 0 and s2 <= 0)
    cond2 = (s3 <= 0 and s4 >= 0) or (s3 >= 0 and s4 <= 0)
    return cond1 and cond2


@dataclass
class CrossingEvent:
    track_id: int
    class_id: int
    class_name: str
    direction: str  # "inbound" or "outbound"
    point: Tuple[float, float]
    frame_idx: int


class LineCrossingCounter:
    """
    Virtual line-crossing counter tracking directional (inbound/outbound) and
    class-specific traffic volume across a configurable line segment P1-P2.
    Locks crossed track IDs into S_crossed to eliminate double-counting from jitter.
    """

    def __init__(
        self,
        p1: Tuple[float, float] = (100.0, 540.0),
        p2: Tuple[float, float] = (1820.0, 540.0),
    ) -> None:
        self.p1 = p1
        self.p2 = p2
        self.crossed_ids: Set[int] = set()
        self.last_centroids: Dict[int, Tuple[float, float]] = {}

        self.inbound_count = 0
        self.outbound_count = 0
        self.counts_by_class: Dict[str, Dict[str, int]] = {
            "car": {"inbound": 0, "outbound": 0},
            "motorcycle": {"inbound": 0, "outbound": 0},
            "bus": {"inbound": 0, "outbound": 0},
            "truck": {"inbound": 0, "outbound": 0},
            "bicycle": {"inbound": 0, "outbound": 0},
            "person": {"inbound": 0, "outbound": 0},
        }
        self.recent_events: List[CrossingEvent] = []
        self.flash_frames_remaining = 0

    def reset(self) -> None:
        self.crossed_ids.clear()
        self.last_centroids.clear()
        self.inbound_count = 0
        self.outbound_count = 0
        for c in self.counts_by_class.values():
            c["inbound"] = 0
            c["outbound"] = 0
        self.recent_events.clear()
        self.flash_frames_remaining = 0

    def update(self, tracks: List[Track], frame_idx: int = 0) -> List[CrossingEvent]:
        """
        Updates counter with current tracks.
        Returns list of new crossing events in this frame.
        """
        new_events: List[CrossingEvent] = []
        if self.flash_frames_remaining > 0:
            self.flash_frames_remaining -= 1

        for track in tracks:
            tid = track.track_id
            curr_c = track.centroid

            if tid in self.last_centroids:
                prev_c = self.last_centroids[tid]

                # If track hasn't crossed yet, test line segment crossing
                if tid not in self.crossed_ids:
                    s_prev = signed_cross_product(self.p1, self.p2, prev_c)
                    s_curr = signed_cross_product(self.p1, self.p2, curr_c)

                    # Sign change indicates crossing line plane (handles exact 0 boundary)
                    crossed_plane = (s_prev <= 0 < s_curr) or (s_prev < 0 <= s_curr) or \
                                    (s_prev >= 0 > s_curr) or (s_prev > 0 >= s_curr)

                    if crossed_plane:
                        if segments_intersect(self.p1, self.p2, prev_c, curr_c):
                            direction = "inbound" if s_curr > s_prev else "outbound"

                            self.crossed_ids.add(tid)
                            if direction == "inbound":
                                self.inbound_count += 1
                            else:
                                self.outbound_count += 1

                            cname = track.class_name
                            if cname not in self.counts_by_class:
                                self.counts_by_class[cname] = {"inbound": 0, "outbound": 0}
                            self.counts_by_class[cname][direction] += 1

                            event = CrossingEvent(
                                track_id=tid,
                                class_id=track.class_id,
                                class_name=cname,
                                direction=direction,
                                point=curr_c,
                                frame_idx=frame_idx,
                            )
                            new_events.append(event)
                            self.recent_events.append(event)
                            self.flash_frames_remaining = 8

            self.last_centroids[tid] = curr_c

        return new_events

    def get_summary(self) -> Dict[str, Any]:
        total_in = self.inbound_count
        total_out = self.outbound_count
        return {
            "total_inbound": total_in,
            "total_outbound": total_out,
            "total_crossings": total_in + total_out,
            "by_class": {
                c: {
                    "inbound": v["inbound"],
                    "outbound": v["outbound"],
                    "total": v["inbound"] + v["outbound"],
                }
                for c, v in self.counts_by_class.items()
                if (v["inbound"] + v["outbound"]) > 0
            },
        }


class TrafficAnnotator:
    """
    Renders bounding boxes, track labels, motion tails, virtual counting line,
    and a professional telemetry heads-up display overlay on video frames.
    """

    def __init__(
        self,
        line_p1: Tuple[float, float] = (100.0, 540.0),
        line_p2: Tuple[float, float] = (1820.0, 540.0),
    ) -> None:
        self.line_p1 = (int(line_p1[0]), int(line_p1[1]))
        self.line_p2 = (int(line_p2[0]), int(line_p2[1]))

    def annotate(
        self,
        frame_bgr: np.ndarray,
        tracks: List[Track],
        counter: LineCrossingCounter,
        engine_name: str = "ORT_INT8_TierS",
        stride: int = 1,
        fps: float = 30.0,
        latency_p95_ms: float = 18.0,
    ) -> np.ndarray:
        """Draws annotations and returns canvas copy."""
        canvas = frame_bgr.copy()

        # 1. Draw Virtual Counting Line (flashing yellow-cyan on event)
        line_color = (0, 255, 255) if counter.flash_frames_remaining > 0 else (255, 128, 0)
        thickness = 4 if counter.flash_frames_remaining > 0 else 2
        cv2.line(canvas, self.line_p1, self.line_p2, line_color, thickness)
        cv2.putText(
            canvas,
            "COUNTING LINE",
            (self.line_p1[0] + 10, self.line_p1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2,
        )

        # 2. Draw Tracks and Trajectory Tails
        for t in tracks:
            cname = t.class_name
            color = CLASS_COLORS.get(cname, (200, 200, 200))

            # Bounding box
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"ID:{t.track_id} {cname} {t.score:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(canvas, (x1, y1 - 20), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # Trajectory tail
            if len(t.history) >= 2:
                pts = np.array([[int(px), int(py)] for px, py in t.history], dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

        # 3. Telemetry HUD Overlay (Top-Left)
        hud_w, hud_h = 360, 160
        sub_img = canvas[10 : 10 + hud_h, 10 : 10 + hud_w]
        white_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        # Translucent dark background
        res = cv2.addWeighted(sub_img, 0.3, white_rect, 0.7, 1.0)
        canvas[10 : 10 + hud_h, 10 : 10 + hud_w] = res
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 80), 1)

        summary = counter.get_summary()
        telemetry_lines = [
            f"ENGINE:  {engine_name} (Stride S={stride})",
            f"FPS:     {fps:.1f}  |  p95: {latency_p95_ms:.1f}ms",
            f"INBOUND: {summary['total_inbound']}  |  OUTBOUND: {summary['total_outbound']}",
            f"TOTAL VEHICLES CROSSED: {summary['total_crossings']}",
        ]
        # Append class counts
        class_str = " | ".join([f"{k}:{v['total']}" for k, v in summary["by_class"].items() if v['total'] > 0])
        if class_str:
            telemetry_lines.append(f"CLASSES: {class_str}")

        y_offset = 32
        for line in telemetry_lines:
            cv2.putText(
                canvas,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
            )
            y_offset += 24

        return canvas
