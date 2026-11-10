# edge-traffic-bench

Rigorous research benchmarking harness evaluating lightweight object detection (YOLO11n/s, YOLO26n/s) and multi-object tracking across CPU runtimes (PyTorch, ONNX Runtime, OpenVINO, ExecuTorch) on an AMD Ryzen 5 7530U (Zen 3, 6C/12T, AVX2, no VNNI, 16GB RAM) running Linux.

---

## Key Architecture & Features

### 1. Thread Oversubscription Prevention
- Explicit control of OpenCV (`cv2.setNumThreads(1)`), NumPy/OpenBLAS/MKL (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`).
- Hardware thread regimes isolate CPU cores (`edge_1c`: 1 thread, `edge_2c`: 2 threads, `full_6c`: 6 threads).

### 2. Zen 3 Topology & Core Affinity Binding
- Parses physical vs logical core mapping via `/proc/cpuinfo`.
- Avoids SMT sibling collisions by mapping physical cores: `[0, 2, 4, 6, 8, 10]`.
- Locks process affinity via `os.sched_setaffinity`.

### 3. Granular Nanosecond 4-Stage Timing
- `t_decode_ms`: Video frame acquisition / decoding
- `t_preprocess_ms`: Letterbox resize, HWC to CHW transpose, contiguous array conversion, normalization
- `t_inference_ms`: Pure runtime forward pass
- `t_postprocess_ms`: Vectorized traffic class filtering + NMS + tracker update + virtual line crossing (+ telemetry HUD annotation)
- Uses `time.perf_counter_ns()`.

### 4. Tail Latency & Warmup Exclusion
- Configurable warmup frames (default: 50 frames) strictly excluded from all benchmark metrics.
- Computes `mean`, `std`, `p50`, `p95`, `p99` latencies and effective FPS.

### 5. Accurate Memory Profiling (USS)
- Uses `psutil.Process().memory_full_info().uss` (Unique Set Size in MB).
- Sampled by a background daemon thread at 10Hz.
- Records `baseline_uss_mb`, `model_load_uss_mb`, and `peak_inference_uss_mb`.

### 6. Non-VNNI AVX2 INT8 Quantization Matrix & Tier-S Spec
- Benchmarks arithmetic saturation (`VPMADDUBSW`) on Zen 3 vs 7-bit reduced range (`reduce_range=True`) and unsigned `U8U8` fallback.
- **Tier-S Zen 3 Spec (`int8_u8s8_tier_s.onnx`)**:
  - **Tail-60 Rule**: Excludes final ~60 decoupled head / DFL decode operator nodes from quantization to prevent coordinate drift.
  - **Percentile Calibration**: 99.999% symmetric activation calibration (`CalibPercentile=99.999`, `CalibTensorRangeSymmetric=True`).
  - Achieves **0.9974** Cosine Similarity and **0.9737** Box IoU agreement against FP32.
- 6 exported artifacts under `models/{model}_{imgsz}/` with total size < 30 MB (budget: 50 MB):
  - `fp32.onnx`
  - `int8_u8s8_full.onnx`
  - `int8_u8s8_7bit.onnx`
  - `int8_u8u8.onnx`
  - `int8_u8s8_tier_s.onnx`
  - `openvino_fp32/`

### 7. Multi-Rate Trackers (Throughput > 500 FPS)
- **ByteTracker**: Two-stage association matching high-confidence ($\tau_{\text{high}}=0.5$) and low-confidence ($\tau_{\text{low}}=0.1$) detections (~2,100 FPS).
- **OCSortTracker**: Observation-Centric SORT with Momentum (OCM), Re-Update (ORU), and Recovery (OCR) (~1,600 FPS).
- **CBIoUTracker**: Cascaded Buffered-IoU with dynamic stride scaling ($b_1 = 0.3 \cdot (1 + 0.35(S-1))$, $b_2 = 0.5 \cdot (1 + 0.50(S-1))$) fused with an ultra-fast (<0.1ms) 24-bin HSV color histogram appearance gate (~780 FPS). Maintains 0 counting error on weaving two-wheelers at $S \in \{3, 5\}$.

### 8. Virtual Line-Crossing Analytics & Telemetry HUD
- Algebraic signed cross-product segment intersection with directional resolution (`inbound` / `outbound`).
- Prevents double-counting via track ID locking in `crossed_ids`.
- `TrafficAnnotator` provides real-time OpenCV HUD with bounding boxes, motion tails, counting line flashing, and FPS/latency metrics.

### 9. Unified End-to-End Pipeline & Reporting
- `TrafficBenchPipeline`: Integrates Video $\to$ Preprocess $\to$ Inference $\to$ Tracker $\to$ Line Crossing $\to$ HUD Annotator under `BenchmarkProfiler`.
- Automated export to structured JSON and tabular CSV.

---

## Directory Structure
```
edge_traffic_bench/
├── configs/
│   └── hardware_profiles.yaml  # Thread profiles: edge_1c, edge_2c, full_6c
├── models/
│   └── yolo11n_320/            # Exported quantization matrix (27.6 MB)
├── src/
│   ├── __init__.py
│   ├── analytics.py            # LineCrossingCounter, TrafficAnnotator
│   ├── calibration.py          # TrafficCalibrationDataReader
│   ├── cli.py                  # CLI entry points
│   ├── exporter.py             # Non-VNNI AVX2 quantization matrix & Tail-60 rule
│   ├── hardware.py             # CPU topology detection, affinity & thread setup
│   ├── inference.py            # PyTorch, ONNX Runtime (IOBinding), OpenVINO engines
│   ├── pipeline.py             # TrafficBenchPipeline, JSON/CSV reporting
│   ├── profiler.py             # Nanosecond stage timer, USS sampler, BenchmarkProfiler
│   ├── tracker.py              # ByteTrack, OC-SORT, CB-IoU trackers
│   └── video_loader.py         # VideoStreamLoader, Synthetic & TrafficPhysics simulators
├── tests/
│   ├── __init__.py
│   ├── test_inference.py       # Quantization matrix & runtime tests
│   ├── test_pipeline.py        # End-to-end pipeline integration & export tests
│   ├── test_profiler.py        # Stage timing, affinity & USS memory tests
│   └── test_tracker.py         # Tracker throughput & multi-rate striding tests
├── pyproject.toml
└── requirements.txt
```

---

## CLI Usage

### 1. Inspect Hardware Detection Summary
```bash
edge-traffic-bench hardware
```

### 2. Export Model Quantization Matrix
```bash
edge-traffic-bench export --model yolo11n --imgsz 320 --calib-frames 20
```

### 3. Benchmark Inference Runtimes
```bash
edge-traffic-bench benchmark-runtime --profile edge_2c --model yolo11n --imgsz 320 --warmup 15 --frames 45
```

### 4. Benchmark Multi-Rate Trackers
```bash
edge-traffic-bench benchmark-tracker --stride 3 --frames 120
```

### 5. Run Full End-to-End Pipeline
```bash
# Synthetic stream on 2 threads with Tier-S INT8 and CB-IoU:
edge-traffic-bench run-pipeline --profile edge_2c --engine ORT_INT8_TierS --tracker CB-IoU --stride 1 --frames 60 --video synth --json-out results/metrics.json --csv-out results/metrics.csv

# Multi-vehicle physics simulation:
edge-traffic-bench run-pipeline --profile edge_2c --engine ORT_INT8_TierS --tracker CB-IoU --stride 1 --frames 120 --video physics
```

### 6. Quantitative Evaluation (Detection Quantization & Multi-Stride Tracking)
```bash
# Evaluate Quantization Fidelity (vs FP32 Reference) across Non-VNNI INT8 Matrix:
edge-traffic-bench evaluate --mode detection

# Evaluate Multi-Stride MOT (MOTA, IDF1, HOTA, IDSW, FRAG, MAE, MAPE, Eff FPS):
edge-traffic-bench evaluate --mode tracking --strides 1 2 3 5 10 --frames 120

# Run complete evaluation suite and save JSON report:
edge-traffic-bench evaluate --mode all --json-out results/eval_summary.json
```

---

## Running Test Suite
```bash
pytest -v tests/
```
All 32 unit tests execute and pass in ~18 seconds.
