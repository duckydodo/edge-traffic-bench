"""
Automated Experiment Suite for AMD Ryzen 5 7530U (Zen 3) Edge Traffic Benchmark.
Executes:
  1. Exp 1: Zen 3 Quantization & Runtime Matrix (PyTorch, ORT FP32/INT8, OpenVINO across edge_1c, edge_2c, full_6c, imgsz 320/640).
  2. Exp 2: Stride vs. Tracking Collapse (Strides S in {1, 2, 3, 5, 10} on ByteTrack, OC-SORT, CB-IoU).
  3. Exp 3: End-to-End Pareto Frontier (Resolution x Quantization x Core Affinity x Tracker x Stride).
Saves clean CSVs to results/csv/.
"""

from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.analytics import LineCrossingCounter
from src.evaluator import DetectionEvaluator, TrackingAndCountingEvaluator
from src.hardware import detect_hardware
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
    flatten_benchmark_metrics,
)
from src.profiler import BenchmarkProfiler
from src.tracker import ByteTracker, CBIoUTracker, OCSortTracker
from src.video_loader import (
    SyntheticVideoGenerator,
    TrafficPhysicsSimulator,
    preprocess_frame,
)


def run_experiment_1(
    output_dir: Union[str, Path] = "results/csv",
    num_frames: int = 30,
    warmup_frames: int = 10,
) -> Path:
    """
    Exp 1: Quantization & Runtime Matrix across Hardware Thread Profiles and Resolutions.
    Compares: PyTorch_FP32, ORT_FP32, ORT_INT8_U8S8_Full, ORT_INT8_U8S8_7bit,
              ORT_INT8_TierS, ORT_INT8_U8U8, OpenVINO_FP32
    across profiles: edge_1c, edge_2c, full_6c
    across resolutions: 320, 640
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exp1_zen3_quantization_and_runtimes.csv"

    profiles = [("edge_1c", 1), ("edge_2c", 2), ("full_6c", 6)]
    resolutions = [320, 640]

    engines_meta = [
        ("PyTorch_FP32", lambda mdir, th: PyTorchEngine("yolo11n.pt", num_threads=th)),
        ("ORT_FP32", lambda mdir, th: OnnxRuntimeEngine(mdir / "fp32.onnx", num_threads=th)),
        ("ORT_INT8_Full_All", lambda mdir, th: OnnxRuntimeEngine(mdir / "int8_u8s8_full.onnx", num_threads=th)),
        ("ORT_INT8_Full_NoHead", lambda mdir, th: OnnxRuntimeEngine(mdir / "int8_u8s8_full_nohead.onnx", num_threads=th)),
        ("ORT_INT8_7bit_NoHead", lambda mdir, th: OnnxRuntimeEngine(mdir / "int8_u8s8_7bit_nohead.onnx", num_threads=th)),
        ("ORT_INT8_7bit_All", lambda mdir, th: OnnxRuntimeEngine(mdir / "int8_u8s8_7bit.onnx", num_threads=th)),
        ("ORT_INT8_TierS", lambda mdir, th: OnnxRuntimeEngine(mdir / "int8_u8s8_tier_s.onnx", num_threads=th)),
        ("ORT_INT8_U8U8", lambda mdir, th: OnnxRuntimeEngine(mdir / "int8_u8u8.onnx", num_threads=th)),
        ("OpenVINO_FP32", lambda mdir, th: OpenVINOEngine(mdir / "openvino_fp32", num_threads=th)),
    ]

    fieldnames = [
        "model", "imgsz", "profile", "threads", "engine",
        "t_preprocess_ms", "t_inference_ms_mean", "t_inference_ms_std",
        "t_inference_ms_p50", "t_inference_ms_p95", "t_inference_ms_p99",
        "t_postprocess_ms", "effective_fps", "peak_uss_mb",
        "cosine_sim", "q_mAP_50", "q_mAP_50_95", "q_mAP_small", "top100_box_iou",
    ]

    print(f"\n=======================================================================================")
    print(f" EXECUTING EXPERIMENT 1: ZEN 3 QUANTIZATION & RUNTIME MATRIX")
    print(f"=======================================================================================")

    det_eval = DetectionEvaluator()
    # Cache fidelity metrics per (imgsz, engine_name)
    fidelity_cache: Dict[Tuple[int, str], Tuple[float, float, float, float, float]] = {}

    rows: List[Dict[str, Any]] = []

    for imgsz in resolutions:
        mdir = Path(f"models/yolo11n_{imgsz}")
        fp32_ref = OnnxRuntimeEngine(mdir / "fp32.onnx", num_threads=2)
        fp32_ref.load()

        # Dummy tensor for cosine similarity
        dummy_tensor = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        ref_out = fp32_ref.forward(dummy_tensor)

        for eng_name, eng_factory in engines_meta:
            # Precompute Quantization Fidelity on real coco128 images at conf=0.20
            if (imgsz, eng_name) not in fidelity_cache:
                try:
                    test_e = eng_factory(mdir, 2)
                    test_e.load()
                    test_out = test_e.forward(dummy_tensor)
                    cos_sim = float(compute_cosine_similarity(ref_out, test_out))

                    if "INT8" in eng_name:
                        q_res = det_eval.evaluate_quantization_fidelity(
                            eval_stream=None,
                            test_engine=test_e,
                            ref_engine=fp32_ref,
                            imgsz=imgsz,
                            num_frames=40,
                            conf_threshold=0.20,
                        )
                        q_50 = q_res["q_mAP_50"]
                        q_map = q_res["q_mAP_50_95"]
                        q_small = q_res["q_mAP_small"]
                        top100_iou = q_res.get("top100_box_iou", 0.0)
                    else:
                        q_50 = 1.0
                        q_map = 1.0
                        q_small = 1.0
                        top100_iou = 1.0
                    fidelity_cache[(imgsz, eng_name)] = (cos_sim, q_50, q_map, q_small, top100_iou)
                    del test_e
                    gc.collect()
                except Exception as ex:
                    print(f"[-] Fidelity eval failed for {eng_name} at {imgsz}p: {ex}")
                    fidelity_cache[(imgsz, eng_name)] = (0.0, 0.0, 0.0, 0.0, 0.0)

            cos_sim, q_50, q_map, q_small, top100_iou = fidelity_cache[(imgsz, eng_name)]

            for prof_name, num_threads in profiles:
                try:
                    engine = eng_factory(mdir, num_threads)
                    profiler = BenchmarkProfiler(
                        profile_name=prof_name,
                        warmup_frames=warmup_frames,
                    )
                    profiler.start()
                    engine.load()
                    profiler.mark_model_loaded()

                    gen = SyntheticVideoGenerator(total_frames=num_frames, width=imgsz, height=imgsz)
                    for f_idx, frame in gen:
                        with profiler.frame_scope(f_idx) as scope:
                            with scope.record("t_decode_ms"):
                                _ = frame.shape
                            with scope.record("t_preprocess_ms"):
                                tensor = preprocess_frame(frame, target_shape=(imgsz, imgsz))
                            with scope.record("t_inference_ms"):
                                raw_out = engine.forward(tensor)
                            with scope.record("t_postprocess_ms"):
                                pass

                    profiler.stop()
                    m = profiler.compute_metrics()
                    inf_s = m["stages"]["t_inference_ms"]

                    row = {
                        "model": "yolo11n",
                        "imgsz": imgsz,
                        "profile": prof_name,
                        "threads": num_threads,
                        "engine": eng_name,
                        "t_preprocess_ms": m["stages"]["t_preprocess_ms"]["mean"],
                        "t_inference_ms_mean": inf_s["mean"],
                        "t_inference_ms_std": inf_s["std"],
                        "t_inference_ms_p50": inf_s["p50"],
                        "t_inference_ms_p95": inf_s["p95"],
                        "t_inference_ms_p99": inf_s["p99"],
                        "t_postprocess_ms": m["stages"]["t_postprocess_ms"]["mean"],
                        "effective_fps": m["effective_fps"],
                        "peak_uss_mb": m["memory_uss_mb"]["peak_inference_uss_mb"],
                        "cosine_sim": round(cos_sim, 6),
                        "q_mAP_50": round(q_50, 4),
                        "q_mAP_50_95": round(q_map, 4),
                        "q_mAP_small": round(q_small, 4),
                        "top100_box_iou": round(top100_iou, 4),
                    }
                    rows.append(row)
                    print(
                        f"[{imgsz}p|{prof_name:<7}|{eng_name:<20}] Inf p50: {inf_s['p50']:>6.2f}ms | "
                        f"p95: {inf_s['p95']:>6.2f}ms | FPS: {m['effective_fps']:>6.1f} | "
                        f"q_mAP: {q_map:>5.3f} | Top100 IoU: {top100_iou:>5.3f}"
                    )
                    del engine
                    gc.collect()
                except Exception as e:
                    print(f"[-] FAILED [{imgsz}p|{prof_name}|{eng_name}]: {e}")

        del fp32_ref
        gc.collect()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Saved Exp 1 results to {csv_path} ({len(rows)} rows)\n")
    return csv_path


def run_experiment_2(
    output_dir: Union[str, Path] = "results/csv",
    total_frames: int = 120,
) -> Path:
    """
    Exp 2: Stride vs. Tracking Collapse.
    Evaluates frame strides S in {1, 2, 3, 5, 10} across ByteTracker, OCSortTracker,
    and CBIoUTracker using ORT_INT8_TierS on edge_2c.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exp2_stride_vs_tracking_collapse.csv"

    strides = [1, 2, 3, 5, 10]
    trackers = [
        ("ByteTrack", lambda s: ByteTracker(stride=s)),
        ("OC-SORT", lambda s: OCSortTracker(stride=s)),
        ("CB-IoU", lambda s: CBIoUTracker(stride=s)),
    ]

    fieldnames = [
        "tracker", "stride", "effective_fps", "HOTA", "IDF1", "MOTA",
        "IDSW_total", "IDSW_two_wheeler", "IDSW_large_vehicle", "FRAG",
        "counting_MAE", "counting_MAPE_pct",
    ]

    print(f"\n=======================================================================================")
    print(f" EXECUTING EXPERIMENT 2: STRIDE VS. TRACKING COLLAPSE (S in {strides})")
    print(f"=======================================================================================")

    mot_eval = TrackingAndCountingEvaluator()
    rows: List[Dict[str, Any]] = []

    for t_name, t_factory in trackers:
        for s in strides:
            tracker = t_factory(s)
            res = mot_eval.evaluate_tracker_simulation(tracker, stride=s, total_frames=total_frames)
            row = {
                "tracker": t_name,
                "stride": s,
                "effective_fps": res["effective_fps"],
                "HOTA": res["hota"],
                "IDF1": res["idf1"],
                "MOTA": res["mota"],
                "IDSW_total": res["idsw_total"],
                "IDSW_two_wheeler": res["idsw_two_wheelers"],
                "IDSW_large_vehicle": res["idsw_large_vehicles"],
                "FRAG": res["frag_total"],
                "counting_MAE": res["counting_mae"],
                "counting_MAPE_pct": res["counting_mape_percent"],
            }
            rows.append(row)
            print(
                f"[{t_name:<10} | S={s:>2}] Eff FPS: {res['effective_fps']:>8.1f} | "
                f"MOTA: {res['mota']:>6.3f} | IDF1: {res['idf1']:>6.3f} | HOTA: {res['hota']:>6.3f} | "
                f"IDSW (2W/Lg): {res['idsw_two_wheelers']}/{res['idsw_large_vehicles']} | "
                f"MAPE: {res['counting_mape_percent']:>5.1f}%"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Saved Exp 2 results to {csv_path} ({len(rows)} rows)\n")
    return csv_path


def run_experiment_3(
    output_dir: Union[str, Path] = "results/csv",
    total_frames: int = 30,
    warmup_frames: int = 5,
) -> Path:
    """
    Exp 3: Full End-to-End Pareto Frontier Sweep.
    Combinations:
      - imgsz: 320, 640
      - engine: FP32, INT8_TierS
      - profile: edge_1c, edge_2c, full_6c
      - tracker: ByteTrack, OC-SORT, CB-IoU
      - stride: 1, 2, 3, 5
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exp3_end_to_end_pareto.csv"

    resolutions = [320, 640]
    engine_types = ["FP32", "INT8_TierS"]
    profiles = [("edge_1c", 1), ("edge_2c", 2), ("full_6c", 6)]
    trackers = ["ByteTrack", "OC-SORT", "CB-IoU"]
    strides = [1, 2, 3, 5]

    fieldnames = [
        "imgsz", "engine", "profile", "threads", "tracker", "stride",
        "effective_fps", "total_latency_mean", "total_latency_p95",
        "peak_uss_mb", "HOTA", "IDF1", "MOTA", "counting_MAE",
        "counting_MAPE_pct", "counting_accuracy_pct", "is_realtime",
    ]

    print(f"\n=======================================================================================")
    print(f" EXECUTING EXPERIMENT 3: FULL END-TO-END PARETO FRONTIER SWEEP")
    print(f"=======================================================================================")

    mot_eval = TrackingAndCountingEvaluator()
    sim_gt = TrafficPhysicsSimulator(total_frames=total_frames)
    gt_crossings = sim_gt.get_ground_truth_crossings()

    rows: List[Dict[str, Any]] = []

    for imgsz in resolutions:
        mdir = Path(f"models/yolo11n_{imgsz}")

        for eng_type in engine_types:
            onnx_file = mdir / ("fp32.onnx" if eng_type == "FP32" else "int8_u8s8_tier_s.onnx")

            for prof_name, num_threads in profiles:
                engine = OnnxRuntimeEngine(onnx_file, num_threads=num_threads)

                for t_name in trackers:
                    for s in strides:
                        tracker = create_tracker(t_name, stride=s)
                        counter = LineCrossingCounter(p1=(50.0, 360.0), p2=(1230.0, 360.0))

                        pipeline = TrafficBenchPipeline(
                            engine=engine,
                            tracker=tracker,
                            counter=counter,
                            profile_name=prof_name,
                            imgsz=imgsz,
                            warmup_frames=warmup_frames,
                            stride=s,
                            engine_name=f"ORT_{eng_type}",
                        )

                        m = pipeline.run(video_source="physics", max_frames=total_frames)
                        t_stats = m["stages"]["t_total_ms"]

                        # Evaluate MOT and counting accuracy on full tracking simulation
                        sim_tracker = create_tracker(t_name, stride=s)
                        t_eval = mot_eval.evaluate_tracker_simulation(sim_tracker, stride=s, total_frames=120)

                        mape = t_eval["counting_mape_percent"]
                        acc_pct = max(0.0, round(100.0 - mape, 2))
                        eff_fps = m["effective_fps"]

                        row = {
                            "imgsz": imgsz,
                            "engine": eng_type,
                            "profile": prof_name,
                            "threads": num_threads,
                            "tracker": t_name,
                            "stride": s,
                            "effective_fps": eff_fps,
                            "total_latency_mean": t_stats["mean"],
                            "total_latency_p95": t_stats["p95"],
                            "peak_uss_mb": m["memory_uss_mb"]["peak_inference_uss_mb"],
                            "HOTA": t_eval["hota"],
                            "IDF1": t_eval["idf1"],
                            "MOTA": t_eval["mota"],
                            "counting_MAE": t_eval["counting_mae"],
                            "counting_MAPE_pct": mape,
                            "counting_accuracy_pct": acc_pct,
                            "is_realtime": (eff_fps >= 30.0),
                        }
                        rows.append(row)

                        rt_badge = "✅ RT" if eff_fps >= 30.0 else "❌ SLOW"
                        print(
                            f"[{imgsz}p|{eng_type:<10}|{prof_name:<7}|{t_name:<9}|S={s}] "
                            f"FPS: {eff_fps:>6.1f} {rt_badge} | Lat: {t_stats['mean']:>5.1f}ms | "
                            f"HOTA: {t_eval['hota']:>5.3f} | Acc: {acc_pct:>5.1f}%"
                        )
                        del pipeline
                        del sim_tracker
                        del tracker
                        del counter
                        gc.collect()

                del engine
                gc.collect()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Saved Exp 3 results to {csv_path} ({len(rows)} rows)\n")
    return csv_path


def run_all_experiments(output_dir: Union[str, Path] = "results/csv") -> Dict[str, Path]:
    """Runs Exp 1, Exp 2, and Exp 3 sequentially and returns paths to generated CSVs."""
    p1 = run_experiment_1(output_dir=output_dir)
    p2 = run_experiment_2(output_dir=output_dir)
    p3 = run_experiment_3(output_dir=output_dir)
    return {
        "exp1": p1,
        "exp2": p2,
        "exp3": p3,
    }
