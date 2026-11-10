"""
edge-traffic-bench: Core Benchmarking & Hardware Profiling Harness.
"""

from src.analytics import LineCrossingCounter, TrafficAnnotator
from src.benchmark_suite import (
    run_all_experiments,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
)
from src.evaluator import DetectionEvaluator, TrackingAndCountingEvaluator
from src.hardware import detect_hardware, print_hardware_summary
from src.inference import OnnxRuntimeEngine, OpenVINOEngine, PyTorchEngine
from src.pipeline import TrafficBenchPipeline, create_engine, create_tracker, save_benchmark_results
from src.plotter import generate_all_plots
from src.profiler import BenchmarkProfiler
from src.tracker import ByteTracker, CBIoUTracker, OCSortTracker

__version__ = "0.1.0"

__all__ = [
    "detect_hardware",
    "print_hardware_summary",
    "BenchmarkProfiler",
    "PyTorchEngine",
    "OnnxRuntimeEngine",
    "OpenVINOEngine",
    "ByteTracker",
    "OCSortTracker",
    "CBIoUTracker",
    "LineCrossingCounter",
    "TrafficAnnotator",
    "TrafficBenchPipeline",
    "DetectionEvaluator",
    "TrackingAndCountingEvaluator",
    "create_engine",
    "create_tracker",
    "save_benchmark_results",
    "run_all_experiments",
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
    "generate_all_plots",
]
