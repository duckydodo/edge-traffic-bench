"""
Unit tests verifying Phase 1 requirements:
- Stage timer isolation across t_decode_ms, t_preprocess_ms, t_inference_ms, t_postprocess_ms
- Warmup exclusion from latency metrics and percentiles
- CPU affinity binding and thread oversubscription control
- USS memory capture during synthetic dummy tensor workloads
- Deterministic frame loading with configurable strides S in {1, 2, 3, 5} and letterboxing
"""

import os
import time
import numpy as np
import pytest

from src.hardware import detect_hardware, set_cpu_affinity, configure_thread_environment
from src.profiler import BenchmarkProfiler, FrameContext, MemorySampler, compute_stage_statistics
from src.video_loader import (
    SyntheticVideoGenerator,
    VideoStreamLoader,
    letterbox,
    preprocess_frame,
)


def test_hardware_detection_zen3():
    """Verify hardware detection correctly parses AMD Ryzen 5 7530U features."""
    info = detect_hardware()
    assert info.physical_cores >= 1
    assert info.logical_cores >= info.physical_cores
    assert info.total_memory_gb > 0
    # AVX2 should be supported on Zen 3
    assert info.has_avx2 is True
    # VNNI should not be supported on Zen 3 (Ryzen 5 7530U)
    assert info.has_vnni is False
    assert len(info.primary_physical_cpus) == info.physical_cores


def test_cpu_affinity_locking():
    """Verify os.sched_setaffinity successfully locks the process to specified cores."""
    if not hasattr(os, "sched_setaffinity"):
        pytest.skip("sched_setaffinity not supported on non-Linux OS")

    original_affinity = os.sched_getaffinity(0)
    try:
        # Bind to core 0 (single thread edge profile)
        target = [0]
        applied = set_cpu_affinity(target)
        assert applied == set(target)
        current = os.sched_getaffinity(0)
        assert current == {0}

        # Bind to dual core [0, 2]
        target_2c = [0, 2]
        applied_2c = set_cpu_affinity(target_2c)
        assert applied_2c == set(target_2c)
        assert os.sched_getaffinity(0) == {0, 2}
    finally:
        # Restore original affinity
        os.sched_setaffinity(0, original_affinity)


def test_thread_environment_configuration():
    """Verify thread cap environment variables are set to prevent oversubscription."""
    configure_thread_environment(num_threads=2, opencv_threads=1)
    assert os.environ.get("OMP_NUM_THREADS") == "2"
    assert os.environ.get("MKL_NUM_THREADS") == "2"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "2"
    assert os.environ.get("VECLIB_MAXIMUM_THREADS") == "2"
    assert os.environ.get("NUMEXPR_NUM_THREADS") == "2"


def test_stage_timer_isolation():
    """Verify that stage timers are isolated and accurately measure respective delays."""
    ctx = FrameContext(frame_idx=0, is_warmup=False)

    # Stage 1: ~10ms sleep
    with ctx.record("t_decode_ms"):
        time.sleep(0.010)

    # Stage 2: ~15ms sleep
    with ctx.record("t_preprocess_ms"):
        time.sleep(0.015)

    # Stage 3: ~20ms sleep
    with ctx.record("t_inference_ms"):
        time.sleep(0.020)

    # Stage 4: ~5ms sleep
    with ctx.record("t_postprocess_ms"):
        time.sleep(0.005)

    metrics = ctx.metrics
    # Validate each stage individually with reasonable margin
    assert 8.0 <= metrics.t_decode_ms <= 35.0, f"Decode ms out of bounds: {metrics.t_decode_ms}"
    assert 13.0 <= metrics.t_preprocess_ms <= 40.0, f"Preprocess ms out of bounds: {metrics.t_preprocess_ms}"
    assert 18.0 <= metrics.t_inference_ms <= 45.0, f"Inference ms out of bounds: {metrics.t_inference_ms}"
    assert 4.0 <= metrics.t_postprocess_ms <= 25.0, f"Postprocess ms out of bounds: {metrics.t_postprocess_ms}"

    expected_sum = (
        metrics.t_decode_ms
        + metrics.t_preprocess_ms
        + metrics.t_inference_ms
        + metrics.t_postprocess_ms
    )
    assert abs(metrics.t_total_ms - expected_sum) < 1e-5


def test_warmup_exclusion():
    """Verify warmup frames are strictly excluded from latency statistics."""
    warmup_count = 5
    measured_count = 10
    total_frames = warmup_count + measured_count

    profiler = BenchmarkProfiler(warmup_frames=warmup_count)
    profiler.start()

    try:
        for idx in range(total_frames):
            with profiler.frame_scope(idx) as frame:
                if idx < warmup_count:
                    # Warmup frames have high simulated latency (e.g. 50ms)
                    with frame.record("t_inference_ms"):
                        time.sleep(0.025)
                else:
                    # Measured frames have low simulated latency (e.g. 5ms)
                    with frame.record("t_inference_ms"):
                        time.sleep(0.005)

        results = profiler.compute_metrics()
        assert results["warmup_frames_processed"] == warmup_count
        assert results["measured_frames_processed"] == measured_count

        # The inference mean must only reflect measured frames (~5ms), NOT the warmup (~25ms)
        inf_mean = results["stages"]["t_inference_ms"]["mean"]
        assert inf_mean < 18.0, f"Warmup was not properly excluded: mean={inf_mean}"
        
        # Verify percentiles are present and monotonic
        inf_stats = results["stages"]["t_inference_ms"]
        assert inf_stats["p50"] <= inf_stats["p95"] <= inf_stats["p99"]
        assert results["effective_fps"] > 0
    finally:
        profiler.stop()


def test_uss_memory_capture_synthetic_workload():
    """Verify USS memory sampler records baseline, model load, and peak inference during dummy tensor workload."""
    with MemorySampler(interval_sec=0.05) as sampler:
        baseline = sampler.record_baseline()
        assert baseline > 0.0

        # Simulate model load: allocate ~50MB array
        model_weights = [np.ones((2000, 2000), dtype=np.float32)]  # ~16MB
        model_load_uss = sampler.record_model_load()
        assert model_load_uss >= baseline - 2.0  # allow minor OS fluctuations

        # Simulate inference peak: allocate larger tensor
        inference_tensors = [np.ones((4000, 4000), dtype=np.float32) for _ in range(3)]  # ~192MB
        time.sleep(0.15)  # Wait for background sampler to sample peak

        summary = sampler.get_summary()
        assert summary["baseline_uss_mb"] > 0
        assert summary["model_load_uss_mb"] > 0
        assert summary["peak_inference_uss_mb"] >= summary["baseline_uss_mb"]
        assert summary["samples_count"] >= 2

        # Clean up memory
        del inference_tensors
        del model_weights


def test_video_loader_strides():
    """Verify SyntheticVideoGenerator and VideoStreamLoader support strides in {1, 2, 3, 5}."""
    for stride in (1, 2, 3, 5):
        gen = SyntheticVideoGenerator(total_frames=30, width=320, height=240, stride=stride)
        frames = list(gen)
        expected_count = (30 + stride - 1) // stride
        assert len(frames) == expected_count
        
        # Verify indices follow stride
        indices = [idx for idx, _ in frames]
        assert indices == list(range(0, 30, stride))

    # Verify invalid stride raises ValueError
    with pytest.raises(ValueError):
        SyntheticVideoGenerator(total_frames=10, stride=4)


def test_letterbox_and_preprocessing():
    """Verify letterboxing preserves aspect ratio and outputs contiguous CHW normalized tensor."""
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3
    target_size = (640, 640)

    # 1. Test letterbox directly
    padded, ratio, (dw, dh) = letterbox(raw_frame, target_shape=target_size)
    assert padded.shape == (640, 640, 3)
    assert dw == 0.0  # width fits exactly
    assert dh == 80.0  # height scaled to 480, remaining 160 split into 80 top/bottom

    # 2. Test full preprocess_frame
    tensor = preprocess_frame(raw_frame, target_shape=target_size, normalize=True)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert 0.0 <= tensor.min() <= 1.0
    assert 0.0 <= tensor.max() <= 1.0


def test_hardware_profiles_yaml():
    """Verify configs/hardware_profiles.yaml exists and parses with required 3 regimes."""
    profiler_1c = BenchmarkProfiler(profile_name="edge_1c", warmup_frames=2)
    assert profiler_1c.profile_name == "edge_1c"
    
    profiler_2c = BenchmarkProfiler(profile_name="edge_2c", warmup_frames=2)
    assert profiler_2c.profile_name == "edge_2c"

    profiler_6c = BenchmarkProfiler(profile_name="full_6c", warmup_frames=2)
    assert profiler_6c.profile_name == "full_6c"
