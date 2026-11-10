"""
CLI entry point for edge-traffic-bench.
Commands:
  hardware: Inspect hardware detection summary
  test-run: Quick verification benchmark
  export: Export model matrix (FP32, INT8 U8S8 full, INT8 U8S8 7-bit, INT8 U8U8, OpenVINO IR)
  benchmark-runtime: Benchmark runtimes and INT8 formats under hardware thread profiles
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.analytics import LineCrossingCounter, TrafficAnnotator
from src.benchmark_suite import run_all_experiments
from src.evaluator import DetectionEvaluator, TrackingAndCountingEvaluator
from src.exporter import export_model_matrix
from src.hardware import detect_hardware, print_hardware_summary
from src.plotter import generate_all_plots
from src.inference import (
    BaseInferenceEngine,
    OnnxRuntimeEngine,
    OpenVINOEngine,
    PyTorchEngine,
    compute_cosine_similarity,
)
from src.pipeline import (
    TrafficBenchPipeline,
    create_engine,
    create_tracker,
    save_benchmark_results,
)
from src.profiler import BenchmarkProfiler
from src.tracker import ByteTracker, CBIoUTracker, OCSortTracker
from src.video_loader import (
    TrafficPhysicsSimulator,
    VideoStreamLoader,
    preprocess_frame,
)


def run_benchmark_for_engine(
    engine: BaseInferenceEngine,
    engine_name: str,
    profile_name: str,
    imgsz: int = 320,
    warmup_frames: int = 20,
    max_frames: int = 60,
    stride: int = 1,
) -> dict:
    """Runs isolated 4-stage benchmark for a given inference engine."""
    profiler = BenchmarkProfiler(
        profile_name=profile_name,
        warmup_frames=warmup_frames,
    )
    profiler.start()

    # Load model under profiler
    engine.load()
    profiler.mark_model_loaded()

    loader = VideoStreamLoader(stride=stride, max_frames=max_frames)
    target_shape = (imgsz, imgsz)

    for frame_idx, raw_frame in loader.frames():
        with profiler.frame_scope(frame_idx) as scope:
            # Stage 1: decode
            with scope.record("t_decode_ms"):
                _ = raw_frame.shape

            # Stage 2: preprocess
            with scope.record("t_preprocess_ms"):
                tensor = preprocess_frame(raw_frame, target_shape=target_shape)

            # Stage 3: inference
            with scope.record("t_inference_ms"):
                raw_preds = engine.forward(tensor)

            # Stage 4: postprocess
            with scope.record("t_postprocess_ms"):
                _ = engine.predict(tensor)

    profiler.stop()
    metrics = profiler.compute_metrics()
    metrics["engine"] = engine_name
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="edge-traffic-bench",
        description="Edge Traffic Benchmark & Hardware Profiling CLI for Zen 3 AMD Ryzen",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: hardware
    hw_parser = subparsers.add_parser("hardware", help="Inspect and display hardware detection summary")
    hw_parser.add_argument("--json", action="store_true", help="Output summary in JSON format")

    # Command: test-run
    run_parser = subparsers.add_parser("test-run", help="Execute dummy synthetic benchmark run")
    run_parser.add_argument(
        "--profile",
        type=str,
        default="edge_1c",
        choices=["edge_1c", "edge_2c", "full_6c"],
        help="Hardware thread profile to apply",
    )
    run_parser.add_argument("--warmup", type=int, default=10, help="Number of warmup frames")
    run_parser.add_argument("--frames", type=int, default=30, help="Number of total frames")
    run_parser.add_argument("--stride", type=int, default=1, choices=[1, 2, 3, 5], help="Frame stride")

    # Command: export
    export_parser = subparsers.add_parser("export", help="Export model to 5-format Non-VNNI AVX2 matrix")
    export_parser.add_argument("--model", type=str, default="yolo11n", help="Model name or path (default: yolo11n)")
    export_parser.add_argument("--imgsz", type=int, default=320, help="Image size (default: 320)")
    export_parser.add_argument("--calib-frames", type=int, default=20, help="Calibration frames for PTQ")
    export_parser.add_argument("--output-dir", type=str, default="models", help="Output directory")

    # Command: benchmark-runtime
    bench_parser = subparsers.add_parser("benchmark-runtime", help="Benchmark runtimes across thread profiles")
    bench_parser.add_argument(
        "--profile",
        type=str,
        default="edge_2c",
        choices=["edge_1c", "edge_2c", "full_6c"],
        help="Hardware thread profile (default: edge_2c)",
    )
    bench_parser.add_argument("--model", type=str, default="yolo11n", help="Model name (default: yolo11n)")
    bench_parser.add_argument("--imgsz", type=int, default=320, help="Image size (default: 320)")
    bench_parser.add_argument("--warmup", type=int, default=15, help="Warmup frames (default: 15)")
    bench_parser.add_argument("--frames", type=int, default=45, help="Total frames (default: 45)")
    bench_parser.add_argument("--stride", type=int, default=1, choices=[1, 2, 3, 5], help="Frame stride")

    # Command: benchmark-tracker
    tracker_parser = subparsers.add_parser("benchmark-tracker", help="Benchmark trackers across multi-rate strides")
    tracker_parser.add_argument("--stride", type=int, default=1, choices=[1, 2, 3, 5, 10], help="Frame stride")
    tracker_parser.add_argument("--frames", type=int, default=120, help="Total simulation frames")

    # Command: run-pipeline
    pipe_parser = subparsers.add_parser("run-pipeline", help="Run unified end-to-end multi-stage traffic benchmark")
    pipe_parser.add_argument(
        "--profile",
        type=str,
        default="edge_2c",
        choices=["edge_1c", "edge_2c", "full_6c"],
        help="Hardware thread profile (default: edge_2c)",
    )
    pipe_parser.add_argument(
        "--engine",
        type=str,
        default="ORT_INT8_TierS",
        help="Inference engine (ORT_INT8_TierS, ORT_FP32, PyTorch_FP32, OpenVINO_FP32, or path to .onnx)",
    )
    pipe_parser.add_argument(
        "--tracker",
        type=str,
        default="CB-IoU",
        choices=["ByteTrack", "OC-SORT", "CB-IoU", "ByteTracker", "OCSortTracker", "CBIoUTracker"],
        help="Multi-object tracker (default: CB-IoU)",
    )
    pipe_parser.add_argument("--model", type=str, default="yolo11n", help="Model name or path (default: yolo11n)")
    pipe_parser.add_argument("--imgsz", type=int, default=320, help="Image size (default: 320)")
    pipe_parser.add_argument("--stride", type=int, default=1, choices=[1, 2, 3, 5, 10], help="Frame stride")
    pipe_parser.add_argument("--warmup", type=int, default=20, help="Warmup frames (default: 20)")
    pipe_parser.add_argument("--frames", type=int, default=60, help="Total frames (default: 60)")
    pipe_parser.add_argument("--video", type=str, default="synth", help="Video source (path, 'synth', or 'physics')")
    pipe_parser.add_argument("--json-out", type=str, default=None, help="Optional output JSON path")
    pipe_parser.add_argument("--csv-out", type=str, default=None, help="Optional output CSV path")
    pipe_parser.add_argument("--save-video", type=str, default=None, help="Optional annotated video path")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Quantitative evaluation of detection quantization & multi-stride tracking")
    eval_parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "detection", "tracking"],
        help="Evaluation mode: 'detection' (quantization fidelity), 'tracking' (multi-stride MOT), or 'all'",
    )
    eval_parser.add_argument("--model", type=str, default="yolo11n", help="Model name or path (default: yolo11n)")
    eval_parser.add_argument("--imgsz", type=int, default=320, help="Image size (default: 320)")
    eval_parser.add_argument("--dataset", type=str, default=None, help="Optional local annotated dataset path (YOLO txt or COCO json)")
    eval_parser.add_argument(
        "--tracker",
        type=str,
        default="all",
        choices=["all", "ByteTrack", "OC-SORT", "CB-IoU"],
        help="Tracker to evaluate (default: all)",
    )
    eval_parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10],
        help="Strides to evaluate (default: 1 2 3 5 10)",
    )
    eval_parser.add_argument("--frames", type=int, default=120, help="Simulation frames for tracking eval (default: 120)")
    eval_parser.add_argument("--json-out", type=str, default=None, help="Optional output JSON path")

    # Command: run-all-experiments
    suite_parser = subparsers.add_parser("run-all-experiments", help="Execute complete benchmark matrix (Exp 1, 2, 3) and generate publication plots")
    suite_parser.add_argument("--csv-dir", type=str, default="results/csv", help="Output directory for CSVs (default: results/csv)")
    suite_parser.add_argument("--fig-dir", type=str, default="results/figures", help="Output directory for figures (default: results/figures)")
    suite_parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")

    # Command: generate-plots
    plot_parser = subparsers.add_parser("generate-plots", help="Generate publication-grade figures from experiment CSVs")
    plot_parser.add_argument("--csv-dir", type=str, default="results/csv", help="Input directory of CSVs (default: results/csv)")
    plot_parser.add_argument("--fig-dir", type=str, default="results/figures", help="Output directory for figures (default: results/figures)")

    args = parser.parse_args()

    if args.command == "hardware" or args.command is None:
        hw_info = detect_hardware()
        if getattr(args, "json", False):
            print(json.dumps(hw_info.to_dict(), indent=2))
        else:
            print_hardware_summary(hw_info)
        return 0

    if args.command == "test-run":
        print(f"[*] Initializing benchmark harness with profile: '{args.profile}', warmup={args.warmup}, frames={args.frames}, stride={args.stride}")
        profiler = BenchmarkProfiler(
            profile_name=args.profile,
            warmup_frames=args.warmup,
        )
        profiler.start()
        profiler.mark_model_loaded()

        loader = VideoStreamLoader(stride=args.stride, max_frames=args.frames)
        for frame_idx, raw_frame in loader.frames():
            with profiler.frame_scope(frame_idx) as scope:
                with scope.record("t_decode_ms"):
                    _ = raw_frame.shape
                with scope.record("t_preprocess_ms"):
                    tensor = preprocess_frame(raw_frame)
                with scope.record("t_inference_ms"):
                    _ = tensor.sum()
                with scope.record("t_postprocess_ms"):
                    pass

        profiler.stop()
        metrics = profiler.compute_metrics()
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "export":
        model_name = args.model
        if not model_name.endswith(".pt") and not Path(model_name).exists():
            pt_path = f"{model_name}.pt"
        else:
            pt_path = model_name

        print(f"[*] Exporting quantization matrix for {pt_path} at resolution {args.imgsz}x{args.imgsz}...")
        artifacts = export_model_matrix(
            model_name_or_path=pt_path,
            imgsz=args.imgsz,
            base_output_dir=args.output_dir,
            num_calibration_frames=args.calib_frames,
        )
        print("[+] Export finished successfully. Generated artifacts:")
        for k, v in artifacts.items():
            print(f"  - {k}: {v}")
        return 0

    if args.command == "benchmark-runtime":
        model_name = args.model
        model_dir = Path("models") / f"{Path(model_name).stem}_{args.imgsz}"
        pt_path = Path(f"{Path(model_name).stem}.pt")

        if not model_dir.exists():
            print(f"[*] Model artifacts not found at {model_dir}. Exporting first...")
            export_model_matrix(
                model_name_or_path=pt_path if pt_path.exists() else "yolo11n.pt",
                imgsz=args.imgsz,
                num_calibration_frames=args.warmup,
            )

        # Thread count from profile name
        threads_map = {"edge_1c": 1, "edge_2c": 2, "full_6c": 6}
        num_threads = threads_map.get(args.profile, 2)

        engines_to_benchmark = [
            ("PyTorch_FP32", PyTorchEngine(pt_path if pt_path.exists() else "yolo11n.pt", num_threads=num_threads)),
            ("ORT_FP32", OnnxRuntimeEngine(model_dir / "fp32.onnx", num_threads=num_threads)),
            ("ORT_INT8_U8S8_Full", OnnxRuntimeEngine(model_dir / "int8_u8s8_full.onnx", num_threads=num_threads)),
            ("ORT_INT8_U8S8_7bit", OnnxRuntimeEngine(model_dir / "int8_u8s8_7bit.onnx", num_threads=num_threads)),
            ("ORT_INT8_U8U8", OnnxRuntimeEngine(model_dir / "int8_u8u8.onnx", num_threads=num_threads)),
            ("ORT_INT8_TierS", OnnxRuntimeEngine(model_dir / "int8_u8s8_tier_s.onnx", num_threads=num_threads)),
            ("OpenVINO_FP32", OpenVINOEngine(model_dir / "openvino_fp32", num_threads=num_threads)),
        ]

        print(f"\n=======================================================================================")
        print(f" BENCHMARK RUNTIME MATRIX: {model_name} ({args.imgsz}x{args.imgsz}) on Profile '{args.profile}'")
        print(f"=======================================================================================")
        print(f"{'Engine':<20} | {'FPS':>8} | {'Inf Mean':>10} | {'Inf p50':>10} | {'Inf p95':>10} | {'Peak USS':>10}")
        print("-" * 87)

        results = []
        for name, engine in engines_to_benchmark:
            try:
                res = run_benchmark_for_engine(
                    engine=engine,
                    engine_name=name,
                    profile_name=args.profile,
                    imgsz=args.imgsz,
                    warmup_frames=args.warmup,
                    max_frames=args.frames,
                    stride=args.stride,
                )
                inf_stats = res["stages"]["t_inference_ms"]
                fps = res["effective_fps"]
                peak_mem = res["memory_uss_mb"]["peak_inference_uss_mb"]
                print(
                    f"{name:<20} | {fps:>8.1f} | {inf_stats['mean']:>8.2f}ms | {inf_stats['p50']:>8.2f}ms | "
                    f"{inf_stats['p95']:>8.2f}ms | {peak_mem:>8.1f}MB"
                )
                results.append(res)
            except Exception as e:
                print(f"{name:<20} | FAILED: {e}")

        print("=" * 87 + "\n")
        return 0

    if args.command == "benchmark-tracker":
        sim = TrafficPhysicsSimulator(total_frames=args.frames, stride=args.stride)
        sim_data = list(sim)
        line = ((50.0, 360.0), (1230.0, 360.0))

        trackers = [
            ("ByteTrack", ByteTracker(stride=args.stride)),
            ("OC-SORT", OCSortTracker(stride=args.stride)),
            ("CB-IoU", CBIoUTracker(stride=args.stride)),
        ]

        print(f"\n=======================================================================================")
        print(f" TRACKER MULTI-RATE BENCHMARK (Stride S={args.stride}, {args.frames} frames)")
        print(f"=======================================================================================")
        print(f"{'Tracker':<15} | {'FPS':>8} | {'Latency':>10} | {'Crossings':>12} | {'Counting Error':>15} | {'IDs':>6}")
        print("-" * 87)

        for name, tracker in trackers:
            counter = LineCrossingCounter(p1=line[0], p2=line[1])
            unique_ids = set()

            t0 = time.perf_counter_ns()
            for f_idx, frame, dets in sim_data:
                tracks = tracker.update(dets, frame)
                for t in tracks:
                    if t.time_since_update == 0:
                        unique_ids.add(t.track_id)
                confirmed_tracks = [t for t in tracks if t.time_since_update == 0]
                counter.update(confirmed_tracks, frame_idx=f_idx)
            t1 = time.perf_counter_ns()

            elapsed_ms = (t1 - t0) / 1e6
            fps = len(sim_data) / ((t1 - t0) / 1e9) if (t1 - t0) > 0 else 0
            summary = counter.get_summary()
            crossings = summary["total_crossings"]
            error = abs(crossings - 5)
            avg_lat = elapsed_ms / len(sim_data) if len(sim_data) > 0 else 0

            print(
                f"{name:<15} | {fps:>8.1f} | {avg_lat:>8.3f}ms | {crossings:>10}/5 | "
                f"{error:>15} | {len(unique_ids):>6}"
            )

        print("=" * 87 + "\n")
        return 0

    if args.command == "run-pipeline":
        threads_map = {"edge_1c": 1, "edge_2c": 2, "full_6c": 6}
        num_threads = threads_map.get(args.profile, 2)

        model_stem = Path(args.model).stem
        model_dir = Path("models") / f"{model_stem}_{args.imgsz}"
        pt_path = Path(f"{model_stem}.pt")

        if not model_dir.exists() and "pytorch" not in args.engine.lower():
            print(f"[*] Export directory {model_dir} not found. Running auto-export for {pt_path}...")
            export_model_matrix(
                model_name_or_path=pt_path if pt_path.exists() else "yolo11n.pt",
                imgsz=args.imgsz,
                num_calibration_frames=args.warmup,
            )

        engine, display_name = create_engine(
            engine_name=args.engine,
            model_name_or_dir=args.model,
            imgsz=args.imgsz,
            num_threads=num_threads,
        )
        tracker = create_tracker(args.tracker, stride=args.stride)
        counter = LineCrossingCounter()
        annotator = TrafficAnnotator() if args.save_video else None

        pipeline = TrafficBenchPipeline(
            engine=engine,
            tracker=tracker,
            counter=counter,
            annotator=annotator,
            profile_name=args.profile,
            imgsz=args.imgsz,
            warmup_frames=args.warmup,
            stride=args.stride,
            engine_name=display_name,
        )

        print(f"\n=======================================================================================")
        print(f" RUNNING END-TO-END PIPELINE: {display_name} + {args.tracker} on Profile '{args.profile}'")
        print(f" Video: {args.video} | Res: {args.imgsz}x{args.imgsz} | Stride: {args.stride} | Frames: {args.frames}")
        print(f"=======================================================================================")

        results = pipeline.run(
            video_source=args.video,
            max_frames=args.frames,
            output_video_path=args.save_video,
        )

        print(f"\n--- Benchmark Results ---")
        print(f"Effective FPS: {results['effective_fps']:.2f}")
        total_lat = results["stages"].get("t_total_ms", {})
        print(
            f"Total Latency: Mean={total_lat.get('mean', 0.0):.2f}ms | "
            f"p50={total_lat.get('p50', 0.0):.2f}ms | "
            f"p95={total_lat.get('p95', 0.0):.2f}ms"
        )
        print(f"Stage Latencies (Mean):")
        for stage, stats in results["stages"].items():
            print(f"  - {stage:<18}: {stats['mean']:>6.2f}ms (p50: {stats['p50']:>6.2f}ms, p95: {stats['p95']:>6.2f}ms)")
        print(
            f"Memory (USS): Baseline={results['memory_uss_mb']['baseline_uss_mb']:.1f}MB | "
            f"Model Load={results['memory_uss_mb']['model_load_uss_mb']:.1f}MB | "
            f"Peak={results['memory_uss_mb']['peak_inference_uss_mb']:.1f}MB"
        )
        print(f"Traffic Analytics:")
        analytics = results["analytics"]
        print(
            f"  - Inbound: {analytics.get('total_inbound', 0)} | "
            f"Outbound: {analytics.get('total_outbound', 0)} | "
            f"Total: {analytics.get('total_crossings', 0)}"
        )
        print("=" * 87 + "\n")

        if args.json_out or args.csv_out:
            save_benchmark_results(results, json_path=args.json_out, csv_path=args.csv_out)
            if args.json_out:
                print(f"[+] Saved JSON metrics to {args.json_out}")
            if args.csv_out:
                print(f"[+] Saved CSV metrics to {args.csv_out}")

        return 0

    if args.command == "evaluate":
        all_eval_results: Dict[str, Any] = {}
        model_stem = Path(args.model).stem
        model_dir = Path("models") / f"{model_stem}_{args.imgsz}"
        pt_path = Path(f"{model_stem}.pt")

        if args.mode in ("all", "detection"):
            det_eval = DetectionEvaluator()
            print(f"\n=======================================================================================")
            print(f" STAGE 1: DETECTION QUANTIZATION & SCALE-PARTITIONED ACCURACY ({args.imgsz}x{args.imgsz})")
            print(f"=======================================================================================")

            if args.dataset:
                print(f"[*] Evaluating on dataset: {args.dataset}")
                tier_s_engine = OnnxRuntimeEngine(model_dir / "int8_u8s8_tier_s.onnx", num_threads=2)
                res = det_eval.evaluate_dataset(args.dataset, tier_s_engine, imgsz=args.imgsz)
                print(f"Dataset Results: mAP_50={res['mAP_50']:.4f} | mAP_50_95={res['mAP_50_95']:.4f} | mAP_small={res['mAP_small']:.4f}")
                all_eval_results["detection_dataset"] = res
            else:
                print(f"[*] Evaluating Quantization Fidelity (vs FP32 Reference) across Non-VNNI INT8 Matrix:")
                print(f"{'Format':<22} | {'q_mAP_50':>10} | {'q_mAP_50_95':>12} | {'q_mAP_small':>12} | {'q_mAP_med':>10} | {'q_mAP_large':>11}")
                print("-" * 87)

                models_to_test = [
                    ("ORT_INT8_TierS", model_dir / "int8_u8s8_tier_s.onnx"),
                    ("ORT_INT8_U8S8_7bit", model_dir / "int8_u8s8_7bit.onnx"),
                    ("ORT_INT8_U8U8", model_dir / "int8_u8u8.onnx"),
                    ("ORT_INT8_U8S8_Full", model_dir / "int8_u8s8_full.onnx"),
                ]

                ref_engine = OnnxRuntimeEngine(model_dir / "fp32.onnx", num_threads=2)
                fidelity_results = {}

                for fmt_name, path in models_to_test:
                    if not path.exists():
                        continue
                    test_eng = OnnxRuntimeEngine(path, num_threads=2)
                    q_res = det_eval.evaluate_quantization_fidelity(
                        eval_stream=None,
                        test_engine=test_eng,
                        ref_engine=ref_engine,
                        imgsz=args.imgsz,
                        num_frames=20,
                    )
                    fidelity_results[fmt_name] = q_res
                    print(
                        f"{fmt_name:<22} | {q_res['q_mAP_50']:>10.4f} | {q_res['q_mAP_50_95']:>12.4f} | "
                        f"{q_res['q_mAP_small']:>12.4f} | {q_res['q_mAP_medium']:>10.4f} | {q_res['q_mAP_large']:>11.4f}"
                    )

                print("=" * 87)
                all_eval_results["quantization_fidelity"] = fidelity_results

        if args.mode in ("all", "tracking"):
            print(f"\n=======================================================================================")
            print(f" STAGE 2: MULTI-STRIDE TRACKING & LINE-CROSSING ABLATION ({args.frames} frames)")
            print(f"=======================================================================================")

            mot_eval = TrackingAndCountingEvaluator()
            trackers_to_eval = []
            if args.tracker in ("all", "ByteTrack"):
                trackers_to_eval.append(("ByteTrack", lambda s: ByteTracker(stride=s)))
            if args.tracker in ("all", "OC-SORT"):
                trackers_to_eval.append(("OC-SORT", lambda s: OCSortTracker(stride=s)))
            if args.tracker in ("all", "CB-IoU"):
                trackers_to_eval.append(("CB-IoU", lambda s: CBIoUTracker(stride=s)))

            tracking_results = {}

            print(
                f"{'Tracker':<10} | {'S':>2} | {'Eff FPS':>8} | {'MOTA':>7} | {'IDF1':>7} | {'HOTA':>7} | "
                f"{'IDSW(2W/Lg)':>11} | {'FRAG(2W/Lg)':>11} | {'MAE':>5} | {'MAPE(%)':>7}"
            )
            print("-" * 92)

            for t_name, t_factory in trackers_to_eval:
                stride_res = mot_eval.evaluate_multi_stride(
                    tracker_factory=t_factory,
                    strides=args.strides,
                    total_frames=args.frames,
                )
                tracking_results[t_name] = stride_res
                for r in stride_res:
                    idsw_str = f"{r['idsw_two_wheelers']}/{r['idsw_large_vehicles']}"
                    frag_str = f"{r['frag_two_wheelers']}/{r['frag_large_vehicles']}"
                    print(
                        f"{t_name:<10} | {r['stride']:>2} | {r['effective_fps']:>8.1f} | {r['mota']:>7.3f} | "
                        f"{r['idf1']:>7.3f} | {r['hota']:>7.3f} | {idsw_str:>11} | {frag_str:>11} | "
                        f"{r['counting_mae']:>5.2f} | {r['counting_mape_percent']:>6.1f}%"
                    )

            print("=" * 92 + "\n")
            all_eval_results["tracking_multi_stride"] = tracking_results

        if args.json_out:
            out_p = Path(args.json_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(all_eval_results, f, indent=2)
            print(f"[+] Saved evaluation metrics to {args.json_out}\n")

        return 0

    if args.command == "run-all-experiments":
        t0 = time.time()
        print(f"\n=======================================================================================")
        print(f" LAUNCHING FULL REPRODUCIBLE EXPERIMENT SUITE")
        print(f" CSV Directory: {args.csv_dir} | Figures Directory: {args.fig_dir}")
        print(f"=======================================================================================\n")

        csv_files = run_all_experiments(output_dir=args.csv_dir)

        if not args.skip_plots:
            print(f"\n[*] Generating academic figures in {args.fig_dir}...")
            generate_all_plots(csv_dir=args.csv_dir, output_dir=args.fig_dir)

        elapsed = time.time() - t0
        print(f"\n[+] Complete experiment suite executed in {elapsed:.1f}s.")
        print(f"    - Exp 1: {csv_files['exp1']}")
        print(f"    - Exp 2: {csv_files['exp2']}")
        print(f"    - Exp 3: {csv_files['exp3']}")
        if not args.skip_plots:
            print(f"    - Figures: {args.fig_dir}/")
        print("=" * 87 + "\n")
        return 0

    if args.command == "generate-plots":
        print(f"\n[*] Generating academic figures from {args.csv_dir} to {args.fig_dir}...")
        generate_all_plots(csv_dir=args.csv_dir, output_dir=args.fig_dir)
        print(f"[+] All publication figures saved to {args.fig_dir}/")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
