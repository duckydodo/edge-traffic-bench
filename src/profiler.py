"""
Core benchmarking, stage timing, USS memory profiling, and CPU affinity binding harness.
"""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import psutil
import yaml

from src.hardware import (
    CPUHardwareInfo,
    configure_thread_environment,
    detect_hardware,
    set_cpu_affinity,
)

STAGES = ("t_decode_ms", "t_preprocess_ms", "t_inference_ms", "t_postprocess_ms")


@dataclass
class FrameMetrics:
    frame_idx: int
    is_warmup: bool
    t_decode_ms: float = 0.0
    t_preprocess_ms: float = 0.0
    t_inference_ms: float = 0.0
    t_postprocess_ms: float = 0.0

    @property
    def t_total_ms(self) -> float:
        return (
            self.t_decode_ms
            + self.t_preprocess_ms
            + self.t_inference_ms
            + self.t_postprocess_ms
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "is_warmup": self.is_warmup,
            "t_decode_ms": self.t_decode_ms,
            "t_preprocess_ms": self.t_preprocess_ms,
            "t_inference_ms": self.t_inference_ms,
            "t_postprocess_ms": self.t_postprocess_ms,
            "t_total_ms": self.t_total_ms,
        }


class MemorySampler:
    """
    Background daemon thread sampling process USS (Unique Set Size in MB) at 10Hz.
    psutil.Process().memory_full_info().uss measures memory unique to this process,
    excluding shared pages.
    """

    def __init__(self, interval_sec: float = 0.1, pid: Optional[int] = None) -> None:
        self.interval_sec = interval_sec
        self.process = psutil.Process(pid or os.getpid())
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.baseline_uss_mb: float = 0.0
        self.model_load_uss_mb: float = 0.0
        self.peak_inference_uss_mb: float = 0.0
        self._samples: List[float] = []

    def _sample_uss(self) -> float:
        try:
            uss_bytes = self.process.memory_full_info().uss
            return uss_bytes / (1024.0 * 1024.0)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            # Fallback to RSS if USS is inaccessible
            return self.process.memory_info().rss / (1024.0 * 1024.0)

    def get_current_uss_mb(self) -> float:
        return self._sample_uss()

    def record_baseline(self) -> float:
        current = self._sample_uss()
        with self._lock:
            self.baseline_uss_mb = current
            if current > self.peak_inference_uss_mb:
                self.peak_inference_uss_mb = current
        return current

    def record_model_load(self) -> float:
        current = self._sample_uss()
        with self._lock:
            self.model_load_uss_mb = current
            if current > self.peak_inference_uss_mb:
                self.peak_inference_uss_mb = current
        return current

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_uss()
            with self._lock:
                self._samples.append(sample)
                if sample > self.peak_inference_uss_mb:
                    self.peak_inference_uss_mb = sample
            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.record_baseline()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="MemorySamplerThread",
        )
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event.is_set() or self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def get_summary(self) -> Dict[str, float]:
        with self._lock:
            return {
                "baseline_uss_mb": round(self.baseline_uss_mb, 2),
                "model_load_uss_mb": round(self.model_load_uss_mb, 2),
                "peak_inference_uss_mb": round(self.peak_inference_uss_mb, 2),
                "samples_count": len(self._samples),
            }

    def __enter__(self) -> MemorySampler:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class FrameContext:
    """Tracks stage timing for an individual frame."""

    def __init__(self, frame_idx: int, is_warmup: bool) -> None:
        self.metrics = FrameMetrics(frame_idx=frame_idx, is_warmup=is_warmup)

    @contextmanager
    def record(self, stage_name: str) -> Iterator[None]:
        if stage_name not in STAGES:
            raise ValueError(f"Unknown stage '{stage_name}'. Valid stages: {STAGES}")
        t_start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            t_end_ns = time.perf_counter_ns()
            elapsed_ms = (t_end_ns - t_start_ns) / 1_000_000.0
            setattr(self.metrics, stage_name, getattr(self.metrics, stage_name) + elapsed_ms)


def compute_stage_statistics(values: Sequence[float]) -> Dict[str, float]:
    """Computes mean, std, p50, p95, and p99 for a sequence of latencies in ms."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


class BenchmarkProfiler:
    """
    Comprehensive profiler combining:
      1. Context-managed stage timer (t_decode_ms, t_preprocess_ms, t_inference_ms, t_postprocess_ms)
      2. Configurable warmup exclusion (default: 50 frames)
      3. Background USS memory sampler at 10Hz
      4. CPU core affinity binder and thread environment enforcement
    """

    def __init__(
        self,
        warmup_frames: int = 50,
        profile_name: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        cpu_affinity: Optional[List[int]] = None,
        num_threads: Optional[int] = None,
        memory_sample_interval_sec: float = 0.1,
    ) -> None:
        self.warmup_frames = warmup_frames
        self.profile_name = profile_name
        self.config_path = Path(config_path) if config_path else Path("configs/hardware_profiles.yaml")
        self.cpu_affinity = cpu_affinity
        self.num_threads = num_threads
        self.memory_sample_interval_sec = memory_sample_interval_sec

        self.memory_sampler = MemorySampler(interval_sec=memory_sample_interval_sec)
        self.hardware_info = detect_hardware()

        self.all_frames: List[FrameMetrics] = []
        self.warmup_records: List[FrameMetrics] = []
        self.measured_records: List[FrameMetrics] = []

        self._benchmark_start_ns: Optional[int] = None
        self._benchmark_end_ns: Optional[int] = None
        self._is_active = False
        self._applied_affinity: Optional[List[int]] = None

        self._load_and_apply_profile()

    def _load_and_apply_profile(self) -> None:
        """Loads hardware profile config and applies affinity and thread caps if requested."""
        if not self.profile_name and not self.cpu_affinity and not self.num_threads:
            return

        profile_cfg: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                root_cfg = yaml.safe_load(f) or {}
                profiles = root_cfg.get("profiles", {})
                if self.profile_name:
                    if self.profile_name not in profiles:
                        raise ValueError(
                            f"Profile '{self.profile_name}' not found in {self.config_path}. "
                            f"Available profiles: {list(profiles.keys())}"
                        )
                    profile_cfg = profiles[self.profile_name]

        # Determine affinity
        affinity_to_apply = self.cpu_affinity or profile_cfg.get("cpu_affinity")
        if affinity_to_apply:
            try:
                applied = set_cpu_affinity(affinity_to_apply)
                self._applied_affinity = sorted(list(applied))
            except Exception as e:
                # If running in restricted environment, log/store notice
                self._applied_affinity = affinity_to_apply

        # Determine thread settings
        threads_to_apply = self.num_threads or profile_cfg.get("num_threads")
        if threads_to_apply:
            opencv_threads = profile_cfg.get("opencv_threads", 1)
            env_vars = profile_cfg.get("env_vars", {})
            configure_thread_environment(
                num_threads=threads_to_apply,
                opencv_threads=opencv_threads,
                extra_env=env_vars,
            )

    def start(self) -> None:
        """Starts the benchmark harness and background memory sampler."""
        self._benchmark_start_ns = time.perf_counter_ns()
        self._is_active = True
        self.memory_sampler.start()

    def stop(self) -> None:
        """Stops the benchmark harness and background memory sampler."""
        self._benchmark_end_ns = time.perf_counter_ns()
        self._is_active = False
        self.memory_sampler.stop()

    def mark_model_loaded(self) -> float:
        """Marks the point where the model was loaded to capture model_load_uss_mb."""
        return self.memory_sampler.record_model_load()

    @contextmanager
    def frame_scope(self, frame_idx: int) -> Iterator[FrameContext]:
        """
        Context manager for an individual frame.
        Automatically classifies frame as warmup or measured.
        """
        is_warmup = frame_idx < self.warmup_frames
        ctx = FrameContext(frame_idx=frame_idx, is_warmup=is_warmup)
        try:
            yield ctx
        finally:
            self.record_frame(ctx.metrics)

    def record_frame(self, metrics: FrameMetrics) -> None:
        """Records a completed frame metric."""
        self.all_frames.append(metrics)
        if metrics.is_warmup:
            self.warmup_records.append(metrics)
        else:
            self.measured_records.append(metrics)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Aggregates metrics strictly excluding warmup frames:
        - Latency stats (mean, std, p50, p95, p99) for each stage and total per-frame time
        - Effective FPS
        - Memory profiling USS statistics
        """
        measured_count = len(self.measured_records)
        warmup_count = len(self.warmup_records)

        stage_metrics: Dict[str, Dict[str, float]] = {}
        stage_names = list(STAGES) + ["t_total_ms"]

        for stage in stage_names:
            if stage == "t_total_ms":
                vals = [m.t_total_ms for m in self.measured_records]
            else:
                vals = [getattr(m, stage) for m in self.measured_records]
            stage_metrics[stage] = compute_stage_statistics(vals)

        # Compute effective FPS:
        # Effective FPS computed over total inference pipeline sum or measured wall time
        if measured_count > 0:
            mean_total_ms = stage_metrics["t_total_ms"]["mean"]
            pipeline_fps = (1000.0 / mean_total_ms) if mean_total_ms > 0 else 0.0
            
            total_stage_time_sec = sum(m.t_total_ms for m in self.measured_records) / 1000.0
            effective_fps = (measured_count / total_stage_time_sec) if total_stage_time_sec > 0 else 0.0
        else:
            mean_total_ms = 0.0
            pipeline_fps = 0.0
            effective_fps = 0.0

        # Memory stats
        mem_summary = self.memory_sampler.get_summary()

        summary: Dict[str, Any] = {
            "profile_name": self.profile_name,
            "warmup_frames": self.warmup_frames,
            "warmup_frames_processed": warmup_count,
            "measured_frames_processed": measured_count,
            "effective_fps": round(effective_fps, 2),
            "pipeline_fps": round(pipeline_fps, 2),
            "stages": {
                stage: {k: round(v, 3) for k, v in stats.items()}
                for stage, stats in stage_metrics.items()
            },
            "memory_uss_mb": mem_summary,
            "hardware": self.hardware_info.to_dict(),
            "cpu_affinity": self._applied_affinity,
        }
        return summary

    def __enter__(self) -> BenchmarkProfiler:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
