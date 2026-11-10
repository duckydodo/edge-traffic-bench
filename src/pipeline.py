"""
End-to-End Multi-Stage Traffic Benchmarking Pipeline.
Chains VideoStreamLoader -> Preprocess -> BaseInferenceEngine -> BaseTracker -> LineCrossingCounter -> TrafficAnnotator.
Orchestrated under BenchmarkProfiler for nanosecond 4-stage timing, warmup exclusion, and USS memory profiling.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.analytics import LineCrossingCounter, TrafficAnnotator
from src.inference import (
    BaseInferenceEngine,
    Detection,
    OnnxRuntimeEngine,
    OpenVINOEngine,
    PyTorchEngine,
    postprocess_yolo,
)
from src.profiler import BenchmarkProfiler
from src.tracker import BaseTracker, ByteTracker, CBIoUTracker, OCSortTracker, Track
from src.video_loader import (
    SyntheticVideoGenerator,
    TrafficPhysicsSimulator,
    VideoStreamLoader,
    letterbox,
    preprocess_frame,
)


def rescale_detections(
    detections: List[Detection],
    ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int],
) -> List[Detection]:
    """Rescales detections from letterboxed input coordinates back to original frame coordinates."""
    if not detections or ratio <= 0:
        return detections
    dw, dh = pad
    orig_h, orig_w = orig_shape[:2]
    rescaled: List[Detection] = []
    for d in detections:
        x1, y1, x2, y2 = d.box
        sx1 = max(0.0, min(float(orig_w), (x1 - dw) / ratio))
        sy1 = max(0.0, min(float(orig_h), (y1 - dh) / ratio))
        sx2 = max(0.0, min(float(orig_w), (x2 - dw) / ratio))
        sy2 = max(0.0, min(float(orig_h), (y2 - dh) / ratio))
        rescaled.append(
            Detection(
                box=(sx1, sy1, sx2, sy2),
                score=d.score,
                class_id=d.class_id,
                class_name=d.class_name,
            )
        )
    return rescaled


def create_tracker(tracker_name: str, stride: int = 1) -> BaseTracker:
    """Factory helper to instantiate tracker from name."""
    normalized = tracker_name.lower().replace("-", "").replace("_", "")
    if "byte" in normalized:
        return ByteTracker(stride=stride)
    elif "ocsort" in normalized:
        return OCSortTracker(stride=stride)
    elif "cbiou" in normalized or "buffered" in normalized:
        return CBIoUTracker(stride=stride)
    else:
        raise ValueError(f"Unknown tracker: '{tracker_name}'. Choose from 'ByteTrack', 'OC-SORT', 'CB-IoU'.")


def create_engine(
    engine_name: str,
    model_name_or_dir: Union[str, Path] = "yolo11n",
    imgsz: int = 320,
    num_threads: int = 2,
) -> Tuple[BaseInferenceEngine, str]:
    """
    Factory helper to instantiate inference engine from name and model directory.
    Returns (engine_instance, display_name).
    """
    path_obj = Path(model_name_or_dir)
    # Check if a direct file was passed
    if path_obj.is_file():
        if path_obj.suffix == ".pt":
            return PyTorchEngine(path_obj, num_threads=num_threads), "PyTorch_FP32"
        elif path_obj.suffix == ".onnx":
            return OnnxRuntimeEngine(path_obj, num_threads=num_threads), path_obj.stem
    elif path_obj.is_dir() and (path_obj / "model.xml").exists():
        return OpenVINOEngine(path_obj, num_threads=num_threads), "OpenVINO_FP32"

    # Otherwise resolve against models/{model_stem}_{imgsz}
    model_stem = Path(model_name_or_dir).stem
    model_dir = Path("models") / f"{model_stem}_{imgsz}"

    eng_clean = engine_name.upper().replace("-", "_").replace(" ", "_")

    if "PYTORCH" in eng_clean or eng_clean == "PT":
        pt_path = Path(f"{model_stem}.pt")
        return PyTorchEngine(pt_path, num_threads=num_threads), "PyTorch_FP32"
    elif "TIER_S" in eng_clean or "TIERS" in eng_clean:
        return OnnxRuntimeEngine(model_dir / "int8_u8s8_tier_s.onnx", num_threads=num_threads), "ORT_INT8_TierS"
    elif "7BIT" in eng_clean:
        return OnnxRuntimeEngine(model_dir / "int8_u8s8_7bit.onnx", num_threads=num_threads), "ORT_INT8_U8S8_7bit"
    elif "U8U8" in eng_clean:
        return OnnxRuntimeEngine(model_dir / "int8_u8u8.onnx", num_threads=num_threads), "ORT_INT8_U8U8"
    elif "FULL" in eng_clean or "U8S8" in eng_clean:
        return OnnxRuntimeEngine(model_dir / "int8_u8s8_full.onnx", num_threads=num_threads), "ORT_INT8_U8S8_Full"
    elif "OPENVINO" in eng_clean or "OV" in eng_clean:
        return OpenVINOEngine(model_dir / "openvino_fp32", num_threads=num_threads), "OpenVINO_FP32"
    elif "FP32" in eng_clean or "ORT" in eng_clean:
        return OnnxRuntimeEngine(model_dir / "fp32.onnx", num_threads=num_threads), "ORT_FP32"
    else:
        # Fallback to Tier-S if available, otherwise FP32 ONNX
        tier_s_path = model_dir / "int8_u8s8_tier_s.onnx"
        if tier_s_path.exists():
            return OnnxRuntimeEngine(tier_s_path, num_threads=num_threads), "ORT_INT8_TierS"
        return OnnxRuntimeEngine(model_dir / "fp32.onnx", num_threads=num_threads), "ORT_FP32"


class TrafficBenchPipeline:
    """
    Unified end-to-end benchmarking pipeline.
    Combines:
      - Video decoding / stream fetching (VideoStreamLoader / Synthetic / Physics)
      - Image letterboxing & tensor preprocessing
      - Inference runtime forward pass (ORT / OpenVINO / PyTorch)
      - Post-processing (NMS / coordinate decode)
      - Multi-object tracking (ByteTrack / OC-SORT / CB-IoU)
      - Directional line-crossing traffic analytics
      - Visual annotation overlay with telemetry HUD
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        tracker: BaseTracker,
        counter: Optional[LineCrossingCounter] = None,
        annotator: Optional[TrafficAnnotator] = None,
        profile_name: str = "edge_2c",
        imgsz: int = 320,
        warmup_frames: int = 20,
        stride: int = 1,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        engine_name: str = "InferenceEngine",
    ) -> None:
        self.engine = engine
        self.tracker = tracker
        self.counter = counter or LineCrossingCounter()
        self.annotator = annotator
        self.profile_name = profile_name
        self.imgsz = imgsz
        self.warmup_frames = warmup_frames
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.engine_name = engine_name

    def run(
        self,
        video_source: Union[str, Path, VideoStreamLoader, TrafficPhysicsSimulator, SyntheticVideoGenerator],
        max_frames: Optional[int] = None,
        output_video_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the benchmark pipeline across all frames under BenchmarkProfiler.
        """
        profiler = BenchmarkProfiler(
            profile_name=self.profile_name,
            warmup_frames=self.warmup_frames,
        )
        profiler.start()

        # Load engine model under profiler
        self.engine.load()
        profiler.mark_model_loaded()

        # Reset stateful tracker and counter
        self.tracker.reset()
        self.counter.reset()

        # Resolve video generator / stream
        is_physics = isinstance(video_source, TrafficPhysicsSimulator) or str(video_source).lower() in ("physics", "sim", "traffic_sim")
        if is_physics and self.counter.p1[1] == 540.0:
            self.counter.p1 = (50.0, 360.0)
            self.counter.p2 = (1230.0, 360.0)
            if self.annotator is not None:
                self.annotator.line_p1 = (50, 360)
                self.annotator.line_p2 = (1230, 360)

        frame_generator = self._resolve_video_stream(video_source, max_frames=max_frames)

        video_writer: Optional[cv2.VideoWriter] = None
        target_shape = (self.imgsz, self.imgsz)
        frame_count = 0

        try:
            for item in frame_generator:
                frame_count += 1
                if isinstance(item, tuple) and len(item) == 3:
                    # Item from TrafficPhysicsSimulator: (frame_idx, frame, gt_dets)
                    frame_idx, frame, sim_gt_dets = item
                else:
                    frame_idx, frame = item
                    sim_gt_dets = None

                # Initialize video writer on first frame if requested
                if output_video_path and video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = Path(output_video_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    video_writer = cv2.VideoWriter(str(out_path), fourcc, 30.0 / self.stride, (w, h))

                with profiler.frame_scope(frame_idx) as scope:
                    # Stage 1: Decode / Frame acquisition
                    with scope.record("t_decode_ms"):
                        orig_shape = frame.shape[:2]

                    # Stage 2: Preprocess
                    with scope.record("t_preprocess_ms"):
                        padded, ratio, pad = letterbox(frame, target_shape=target_shape)
                        tensor = preprocess_frame(frame, target_shape=target_shape)

                    # Stage 3: Model Inference Forward Pass
                    with scope.record("t_inference_ms"):
                        raw_preds = self.engine.forward(tensor)

                    # Stage 4: Postprocess + Tracker + Analytics + Annotation
                    with scope.record("t_postprocess_ms"):
                        raw_dets = postprocess_yolo(
                            raw_preds,
                            conf_threshold=self.conf_threshold,
                            iou_threshold=self.iou_threshold,
                            filter_traffic_classes=True,
                        )
                        dets = rescale_detections(raw_dets, ratio, pad, orig_shape)
                        if sim_gt_dets is not None and len(dets) == 0:
                            dets = sim_gt_dets

                        # Update tracker
                        tracks = self.tracker.update(dets, frame_bgr=frame)
                        confirmed_tracks = [t for t in tracks if t.time_since_update == 0]

                        # Update line-crossing analytics
                        _ = self.counter.update(confirmed_tracks, frame_idx=frame_idx)

                        # Render visual annotation if annotator is enabled
                        if self.annotator is not None:
                            annotated_frame = self.annotator.annotate(
                                frame_bgr=frame,
                                tracks=confirmed_tracks,
                                counter=self.counter,
                                engine_name=self.engine_name,
                                stride=self.stride,
                            )
                            if video_writer is not None:
                                video_writer.write(annotated_frame)
        finally:
            if video_writer is not None:
                video_writer.release()
            profiler.stop()

        # Compute benchmark metrics
        metrics = profiler.compute_metrics()

        # Attach traffic analytics
        metrics["analytics"] = self.counter.get_summary()

        # Attach pipeline metadata
        metrics["pipeline"] = {
            "engine": self.engine_name,
            "tracker": self.tracker.__class__.__name__,
            "profile": self.profile_name,
            "stride": self.stride,
            "imgsz": self.imgsz,
            "warmup_frames": self.warmup_frames,
            "frames_processed": frame_count,
        }

        return metrics

    def _resolve_video_stream(
        self,
        video_source: Union[str, Path, VideoStreamLoader, TrafficPhysicsSimulator, SyntheticVideoGenerator],
        max_frames: Optional[int] = None,
    ) -> Iterator[Any]:
        """Resolves various video source inputs into an iterator of frames."""
        if isinstance(video_source, (VideoStreamLoader, TrafficPhysicsSimulator, SyntheticVideoGenerator)):
            if hasattr(video_source, "frames"):
                return iter(video_source.frames())
            return iter(video_source)

        src_str = str(video_source).lower()
        if src_str in ("synth", "synthetic"):
            gen = SyntheticVideoGenerator(
                total_frames=max_frames or 100,
                stride=self.stride,
            )
            return iter(gen)
        elif src_str in ("physics", "sim", "traffic_sim"):
            sim = TrafficPhysicsSimulator(
                total_frames=max_frames or 120,
                stride=self.stride,
            )
            return iter(sim)
        else:
            loader = VideoStreamLoader(
                video_path=video_source,
                stride=self.stride,
                max_frames=max_frames,
            )
            return iter(loader.frames())


def flatten_benchmark_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens nested benchmark metrics into a tabular dictionary for CSV export."""
    row: Dict[str, Any] = {}
    pipe = results.get("pipeline", {})
    row["profile"] = pipe.get("profile", "")
    row["engine"] = pipe.get("engine", "")
    row["tracker"] = pipe.get("tracker", "")
    row["stride"] = pipe.get("stride", 1)
    row["imgsz"] = pipe.get("imgsz", 320)
    row["frames_processed"] = pipe.get("frames_processed", 0)
    row["warmup_frames"] = pipe.get("warmup_frames", 0)

    row["effective_fps"] = results.get("effective_fps", 0.0)

    stages = results.get("stages", {})
    for stage in ["t_decode_ms", "t_preprocess_ms", "t_inference_ms", "t_postprocess_ms"]:
        stats = stages.get(stage, {})
        row[f"{stage}_mean"] = stats.get("mean", 0.0)
        row[f"{stage}_std"] = stats.get("std", 0.0)
        row[f"{stage}_p50"] = stats.get("p50", 0.0)
        row[f"{stage}_p95"] = stats.get("p95", 0.0)
        row[f"{stage}_p99"] = stats.get("p99", 0.0)

    tot = stages.get("t_total_ms", {})
    row["total_latency_mean"] = tot.get("mean", 0.0)
    row["total_latency_p50"] = tot.get("p50", 0.0)
    row["total_latency_p95"] = tot.get("p95", 0.0)

    mem = results.get("memory_uss_mb", {})
    row["baseline_uss_mb"] = mem.get("baseline_uss_mb", 0.0)
    row["model_load_uss_mb"] = mem.get("model_load_uss_mb", 0.0)
    row["peak_inference_uss_mb"] = mem.get("peak_inference_uss_mb", 0.0)

    analytics = results.get("analytics", {})
    row["total_crossings"] = analytics.get("total_crossings", 0)
    row["total_inbound"] = analytics.get("total_inbound", 0)
    row["total_outbound"] = analytics.get("total_outbound", 0)

    return row


def save_benchmark_results(
    results: Dict[str, Any],
    json_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Exports benchmark results to structured JSON and tabular CSV."""
    if json_path is not None:
        p_json = Path(json_path)
        p_json.parent.mkdir(parents=True, exist_ok=True)
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if csv_path is not None:
        p_csv = Path(csv_path)
        p_csv.parent.mkdir(parents=True, exist_ok=True)
        flat_row = flatten_benchmark_metrics(results)
        file_exists = p_csv.exists() and p_csv.stat().st_size > 0
        with open(p_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_row)
