"""
Publication-Grade Visualization Engine for edge-traffic-bench.
Generates:
  1. fig1_avx2_quantization_tradeoff.png: Latency vs. Quantization Fidelity on Zen 3.
  2. fig2_runtime_core_scaling.png: FPS, Tail Latency, and Memory across core regimes.
  3. fig3_stride_tracking_degradation.png: Multi-stride IDF1/HOTA & two-wheeler IDSW surge.
  4. fig4_pareto_frontier.png: Effective FPS vs. Counting Accuracy Pareto frontier.
  5. sample_annotated_frame.png: High-res annotated telemetry frame.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analytics import LineCrossingCounter, TrafficAnnotator
from src.tracker import ByteTracker, CBIoUTracker, OCSortTracker, Track
from src.video_loader import TrafficPhysicsSimulator


def set_academic_style() -> None:
    """Applies clean academic IEEE/ACM styling for matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })


def load_csv(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads CSV into list of typed dictionaries."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for k, v in r.items():
                try:
                    if "." in v:
                        parsed[k] = float(v)
                    elif v.lower() in ("true", "false"):
                        parsed[k] = (v.lower() == "true")
                    else:
                        parsed[k] = int(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def plot_fig1_quantization_tradeoff(
    exp1_csv: Union[str, Path],
    output_path: Union[str, Path] = "results/figures/fig1_avx2_quantization_tradeoff.png",
) -> None:
    """
    Fig 1: Latency vs. Quantization Fidelity across Zen 3 INT8 Matrix (320x320 on edge_2c).
    Highlights VPMADDUBSW arithmetic saturation and head node collapse vs. Tier-S preservation.
    """
    rows = load_csv(exp1_csv)
    # Filter 320x320 on edge_2c
    filtered = [
        r for r in rows
        if r.get("imgsz") == 320 and r.get("profile") == "edge_2c" and "ORT" in r.get("engine", "")
    ]

    engine_order = [
        ("ORT_FP32", "FP32 (Baseline)", "#1f77b4"),
        ("ORT_INT8_Full_All", "INT8 Full (All Nodes)", "#d62728"),
        ("ORT_INT8_Full_NoHead", "INT8 Full (Head Excluded)", "#e377c2"),
        ("ORT_INT8_7bit_NoHead", "INT8 7-bit (Head Excluded)", "#ff7f0e"),
        ("ORT_INT8_TierS", "INT8 Tier-S (Ours)", "#2ca02c"),
        ("ORT_INT8_U8U8", "INT8 U8U8 (Head Excluded)", "#8c564b"),
    ]

    labels = []
    p50_lat = []
    p95_lat = []
    q_map = []
    q_small = []
    colors = []

    for eng_key, label, color in engine_order:
        match = next((r for r in filtered if r.get("engine") == eng_key or (eng_key == "ORT_INT8_Full_All" and r.get("engine") == "ORT_INT8_U8S8_Full")), None)
        if match:
            labels.append(label)
            p50_lat.append(match.get("t_inference_ms_p50", 0.0))
            p95_lat.append(match.get("t_inference_ms_p95", 0.0))
            q_map.append(match.get("q_mAP_50_95", 0.0))
            q_small.append(match.get("q_mAP_small", 0.0))
            colors.append(color)

    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=300)

    x = np.arange(len(labels))
    width = 0.35

    # Panel A: Latency (p50 and p95 tail)
    b1 = ax1.bar(x - width/2, p50_lat, width, label="Inference p50", color=colors, alpha=0.85, edgecolor="black")
    b2 = ax1.bar(x + width/2, p95_lat, width, label="Inference p95 (Tail)", color=colors, alpha=0.45, hatch="//", edgecolor="black")
    max_y1 = max(p95_lat) if p95_lat else 30.0
    ax1.set_ylim(0.0, max_y1 * 1.25)
    for i in range(len(labels)):
        ax1.text(x[i] - width/2, p50_lat[i] + max_y1 * 0.015, f"{p50_lat[i]:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax1.text(x[i] + width/2, p95_lat[i] + max_y1 * 0.015, f"{p95_lat[i]:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_ylabel("Latency (ms) - Lower is Better")
    ax1.set_title("(a) Inference Latency (p50 & p95 Tail)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    # Panel B: Quantization Fidelity (mAP 50-95 and mAP Small)
    ax2.bar(x - width/2, q_map, width, label="Fidelity q-mAP 50:95", color="#1f77b4", alpha=0.85, edgecolor="black")
    ax2.bar(x + width/2, q_small, width, label="Small Object q-mAP", color="#2ca02c", alpha=0.85, hatch="..", edgecolor="black")
    ax2.set_ylim(0.0, 1.25)
    for i in range(len(labels)):
        ax2.text(x[i] - width/2, q_map[i] + 0.015, f"{q_map[i]:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax2.text(x[i] + width/2, q_small[i] + 0.015, f"{q_small[i]:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylabel("Fidelity vs. FP32 Reference - Higher is Better")
    ax2.set_title("(b) Quantization Fidelity & Small-Object Preservation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.axhline(1.0, color="gray", linestyle=":", label="FP32 Reference")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Generated Fig 1: {output_path}")


def plot_fig2_runtime_core_scaling(
    exp1_csv: Union[str, Path],
    output_path: Union[str, Path] = "results/figures/fig2_runtime_core_scaling.png",
) -> None:
    """
    Fig 2: Runtime & Core Scaling across edge_1c, edge_2c, full_6c for PyTorch, ORT, and OpenVINO.
    3-panel layout: Throughput (FPS), Tail Latency (p95 ms), and Peak Memory (USS MB).
    """
    rows = load_csv(exp1_csv)
    # Focus on 320x320 FP32 runtimes across the 3 core thread regimes
    targets = [
        ("PyTorch_FP32", "PyTorch (CPU)", "#1f77b4", "o"),
        ("ORT_FP32", "ONNX Runtime (MLAS)", "#2ca02c", "s"),
        ("OpenVINO_FP32", "OpenVINO (oneDNN)", "#ff7f0e", "^"),
    ]
    core_regimes = [("edge_1c", 1), ("edge_2c", 2), ("full_6c", 6)]

    set_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    ax_fps, ax_tail, ax_mem = axes

    x_labels = ["1 Core (edge_1c)", "2 Cores (edge_2c)", "6 Cores (full_6c)"]
    x_pos = np.arange(len(core_regimes))

    for eng_name, label, color, marker in targets:
        fps_vals = []
        tail_vals = []
        mem_vals = []
        for prof, _ in core_regimes:
            match = next((r for r in rows if r.get("imgsz") == 320 and r.get("profile") == prof and r.get("engine") == eng_name), None)
            if match:
                fps_vals.append(match.get("effective_fps", 0.0))
                tail_vals.append(match.get("t_inference_ms_p95", 0.0))
                mem_vals.append(match.get("peak_uss_mb", 0.0))
            else:
                fps_vals.append(0.0)
                tail_vals.append(0.0)
                mem_vals.append(0.0)

        ax_fps.plot(x_pos, fps_vals, marker=marker, color=color, linewidth=2.0, markersize=8, label=label)
        ax_tail.plot(x_pos, tail_vals, marker=marker, color=color, linewidth=2.0, markersize=8, label=label)
        ax_mem.plot(x_pos, mem_vals, marker=marker, color=color, linewidth=2.0, markersize=8, label=label)

    # Styling FPS
    ax_fps.set_title("(a) Inference Throughput")
    ax_fps.set_ylabel("Effective FPS - Higher is Better")
    ax_fps.set_xticks(x_pos)
    ax_fps.set_xticklabels(x_labels)
    ax_fps.grid(True)
    ax_fps.legend(loc="upper left")

    # Styling Tail Latency
    ax_tail.set_title("(b) Tail Latency (p95)")
    ax_tail.set_ylabel("Latency (ms) - Lower is Better")
    ax_tail.set_xticks(x_pos)
    ax_tail.set_xticklabels(x_labels)
    ax_tail.grid(True)
    ax_tail.legend(loc="upper right")

    # Styling Memory
    ax_mem.set_title("(c) Peak USS Memory Footprint")
    ax_mem.set_ylabel("Peak USS (MB) - Lower is Better")
    ax_mem.set_xticks(x_pos)
    ax_mem.set_xticklabels(x_labels)
    ax_mem.set_ylim(440, 620)
    ax_mem.grid(True)
    ax_mem.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Generated Fig 2: {output_path}")


def plot_fig3_stride_tracking_degradation(
    exp2_csv: Union[str, Path],
    output_path: Union[str, Path] = "results/figures/fig3_stride_tracking_degradation.png",
) -> None:
    """
    Fig 3: Tracking Degradation and Two-Wheeler Vulnerability over Frame Strides S in {1, 2, 3, 5, 10}.
    Panel (a): IDF1 & HOTA degradation across ByteTrack, OC-SORT, CB-IoU.
    Panel (b): Two-Wheeler vs Large-Vehicle IDSW surge and Counting MAPE (%).
    """
    rows = load_csv(exp2_csv)
    strides = [1, 2, 3, 5, 10]
    trackers = [
        ("ByteTrack", "#d62728", "o"),
        ("OC-SORT", "#1f77b4", "s"),
        ("CB-IoU", "#2ca02c", "^"),
    ]

    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    for t_name, color, marker in trackers:
        t_rows = [r for r in rows if r.get("tracker") == t_name]
        t_rows.sort(key=lambda x: x.get("stride", 0))

        s_vals = [r.get("stride") for r in t_rows]
        idf1_vals = [r.get("IDF1", 0.0) for r in t_rows]
        hota_vals = [r.get("HOTA", 0.0) for r in t_rows]
        mape_vals = [r.get("counting_MAPE_pct", 0.0) for r in t_rows]

        # Panel A: IDF1 (Solid) and HOTA (Dashed)
        ax1.plot(s_vals, idf1_vals, marker=marker, linestyle="-", color=color, linewidth=2.0, label=f"{t_name} (IDF1)")
        ax1.plot(s_vals, hota_vals, marker=marker, linestyle="--", color=color, linewidth=1.5, alpha=0.7, label=f"{t_name} (HOTA)")

        # Panel B: Counting MAPE
        ax2.plot(s_vals, mape_vals, marker=marker, linestyle="-", color=color, linewidth=2.0, label=f"{t_name} MAPE (%)")

    # Add IDSW breakdown for ByteTrack in Panel B (Two-Wheeler vs Large Vehicle)
    byte_rows = sorted([r for r in rows if r.get("tracker") == "ByteTrack"], key=lambda x: x.get("stride", 0))
    s_b = [r.get("stride") for r in byte_rows]
    idsw_2w = [r.get("IDSW_two_wheeler", 0) for r in byte_rows]
    idsw_lg = [r.get("IDSW_large_vehicle", 0) for r in byte_rows]

    ax2_twin = ax2.twinx()
    ax2_twin.plot(s_b, idsw_2w, color="purple", linestyle=":", marker="x", linewidth=2.0, label="ByteTrack Two-Wheeler IDSW")
    ax2_twin.plot(s_b, idsw_lg, color="brown", linestyle=":", marker="+", linewidth=1.5, label="ByteTrack Large-Vehicle IDSW")
    ax2_twin.set_ylabel("Identity Switches (Count)", color="purple")
    ax2_twin.tick_params(axis="y", labelcolor="purple")

    # Panel A styling
    ax1.set_title("(a) Trajectory Identity Preservation (IDF1 & HOTA)")
    ax1.set_xlabel("Frame Stride (S)")
    ax1.set_ylabel("Score [0.0 - 1.0] - Higher is Better")
    ax1.set_xticks(strides)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=8)

    # Panel B styling
    ax2.set_title("(b) Traffic Counting Error & IDSW Surge")
    ax2.set_xlabel("Frame Stride (S)")
    ax2.set_ylabel("Line-Crossing Counting MAPE (%) - Lower is Better")
    ax2.set_xticks(strides)
    ax2.set_ylim(0, 65)
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax2_twin.set_ylim(0, 7.5)
    ax2_twin.legend(loc="center right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Generated Fig 3: {output_path}")


def plot_fig4_pareto_frontier(
    exp3_csv: Union[str, Path],
    output_path: Union[str, Path] = "results/figures/fig4_pareto_frontier.png",
) -> None:
    """
    Fig 4: End-to-End Effective Throughput (FPS) vs Counting Accuracy (100 - MAPE %)
    Maps the Pareto frontier across combinations of Resolution, Quantization, Cores, Trackers, and Strides.
    """
    rows = load_csv(exp3_csv)

    set_academic_style()
    plt.figure(figsize=(10, 6.5), dpi=300)

    fps_list = []
    acc_list = []

    color_map = {
        (320, "INT8_TierS"): "#2ca02c",  # Green
        (320, "FP32"): "#1f77b4",        # Blue
        (640, "INT8_TierS"): "#ff7f0e",  # Orange
        (640, "FP32"): "#d62728",        # Red
    }

    marker_map = {
        "ByteTrack": "o",
        "OC-SORT": "s",
        "CB-IoU": "^",
    }

    # Group and scatter
    for (imgsz, eng), col in color_map.items():
        for trk, mkr in marker_map.items():
            subset = [
                r for r in rows
                if r.get("imgsz") == imgsz and r.get("engine") == eng and r.get("tracker") == trk
            ]
            if subset:
                xs = [r.get("effective_fps", 0.0) for r in subset]
                ys = [r.get("counting_accuracy_pct", 0.0) for r in subset]
                fps_list.extend(xs)
                acc_list.extend(ys)
                plt.scatter(
                    xs, ys,
                    color=col,
                    marker=mkr,
                    s=60,
                    alpha=0.75,
                    edgecolors="black",
                    linewidth=0.6,
                    label=f"{imgsz}p {eng} | {trk}" if (trk == "CB-IoU") else None,
                )

    assert len(fps_list) > 0, f"No valid data points found in {exp3_csv}!"
    print(f"[*] Plotting {len(fps_list)} configurations on Pareto frontier...")

    # Compute Pareto frontier points: max Accuracy for a given FPS
    pts = sorted(zip(fps_list, acc_list), key=lambda p: p[0])
    pareto_fps = []
    pareto_acc = []
    max_acc = -1.0

    # Backwards sweep to extract non-dominated points
    for x, y in reversed(pts):
        if y >= max_acc:
            pareto_fps.append(x)
            pareto_acc.append(y)
            max_acc = y

    pareto_fps.reverse()
    pareto_acc.reverse()

    # Draw Pareto boundary if non-dominated points exist
    if len(pareto_fps) > 0:
        plt.plot(pareto_fps, pareto_acc, color="#111111", linestyle="--", linewidth=2.0, label="Pareto-Optimal Frontier", zorder=5)
        plt.scatter(pareto_fps, pareto_acc, color="#ffd700", marker="*", s=160, edgecolors="black", linewidth=1.2, zorder=6, label="Non-Dominated Operating Points")

    # Real-Time threshold line (30 FPS)
    plt.axvline(30.0, color="#d62728", linestyle=":", linewidth=2.0, label="Real-Time Threshold (30 FPS)", zorder=4)

    plt.title("End-to-End Edge Traffic Analytics Pareto Frontier on AMD Ryzen 5 7530U")
    plt.xlabel("Effective End-to-End Throughput (FPS) - Higher is Better")
    plt.ylabel("Line-Crossing Counting Accuracy (100 - MAPE %) - Higher is Better")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0.0, 105.0)
    plt.xlim(0.0, max(fps_list) * 1.08 if fps_list else 100.0)
    plt.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Generated Fig 4: {output_path}")


def generate_sample_annotated_frame(
    output_path: Union[str, Path] = "results/sample_annotated_frame.png",
) -> None:
    """Generates an annotated telemetry HUD frame using TrafficAnnotator."""
    sim = TrafficPhysicsSimulator(total_frames=50, stride=1)
    line = ((50.0, 360.0), (1230.0, 360.0))
    counter = LineCrossingCounter(p1=line[0], p2=line[1])
    annotator = TrafficAnnotator(line_p1=line[0], line_p2=line[1])
    tracker = CBIoUTracker(stride=1)

    annotated = None
    for f_idx, frame, detections in sim:
        tracks = tracker.update(detections, frame_bgr=frame)
        confirmed = [t for t in tracks if t.time_since_update == 0]
        counter.update(confirmed, frame_idx=f_idx)
        if f_idx == 40:
            annotated = annotator.annotate(
                frame_bgr=frame,
                tracks=confirmed,
                counter=counter,
                engine_name="ORT_INT8_TierS",
                stride=1,
                fps=42.6,
                latency_p95_ms=23.5,
            )
            break

    if annotated is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), annotated)
        print(f"[+] Generated sample annotated frame: {output_path}")


def generate_all_plots(
    csv_dir: Union[str, Path] = "results/csv",
    output_dir: Union[str, Path] = "results/figures",
) -> None:
    """Generates Fig 1, Fig 2, Fig 3, Fig 4, and sample annotated frame."""
    c_dir = Path(csv_dir)
    o_dir = Path(output_dir)

    plot_fig1_quantization_tradeoff(c_dir / "exp1_zen3_quantization_and_runtimes.csv", o_dir / "fig1_avx2_quantization_tradeoff.png")
    plot_fig2_runtime_core_scaling(c_dir / "exp1_zen3_quantization_and_runtimes.csv", o_dir / "fig2_runtime_core_scaling.png")
    plot_fig3_stride_tracking_degradation(c_dir / "exp2_stride_vs_tracking_collapse.csv", o_dir / "fig3_stride_tracking_degradation.png")
    plot_fig4_pareto_frontier(c_dir / "exp3_end_to_end_pareto.csv", o_dir / "fig4_pareto_frontier.png")
    generate_sample_annotated_frame("results/sample_annotated_frame.png")
