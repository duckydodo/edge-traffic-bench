"""
Unit tests for Phase 3:
- Multi-Object Tracker Suite (ByteTracker, OCSortTracker, CBIoUTracker)
- High throughput verification (>500 FPS tracker-only stage on CPU)
- Exact ground-truth line-crossing counts at Stride S=1
- Superior ID-preservation and lower counting error of OCSortTracker and CBIoUTracker at Strides S=3 and S=5
- Fast 24-bin HSV appearance gate verification (<0.2ms and color distance)
- Virtual line-crossing analytics, signed cross-product, and telemetry HUD annotator
"""

import time
import cv2
import numpy as np
import pytest

from src.analytics import (
    CLASS_COLORS,
    LineCrossingCounter,
    TrafficAnnotator,
    segments_intersect,
    signed_cross_product,
)
from src.inference import Detection
from src.tracker import (
    ByteTracker,
    CBIoUTracker,
    OCSortTracker,
    box_iou_matrix,
    buffer_boxes,
    extract_hsv_histogram_24bin,
)
from src.video_loader import TrafficPhysicsSimulator


def test_trackers_execution_speed_above_500_fps():
    """Verify all 3 CPU trackers exceed 500 FPS in tracker-only mode."""
    sim = TrafficPhysicsSimulator(total_frames=120, stride=1)
    frames_data = list(sim)

    for TrackerCls in [ByteTracker, OCSortTracker, CBIoUTracker]:
        tracker = TrackerCls(stride=1)

        # Warmup
        for _, frame, dets in frames_data[:20]:
            tracker.update(dets, frame)
        tracker.reset()

        t0 = time.perf_counter_ns()
        iterations = 5
        for _ in range(iterations):
            for _, frame, dets in frames_data:
                tracker.update(dets, frame)
        t1 = time.perf_counter_ns()

        total_frames = iterations * len(frames_data)
        elapsed_sec = (t1 - t0) / 1e9
        fps = total_frames / elapsed_sec

        print(f"\n[Tracker Speed] {TrackerCls.__name__}: {fps:.1f} FPS")
        assert fps > 500.0, f"{TrackerCls.__name__} throughput {fps:.1f} FPS failed >500 FPS requirement"


def test_fast_hsv_histogram_gate():
    """Verify 24-bin HSV histogram executes in <0.2ms and discriminates vehicle colors."""
    crop_red = np.full((60, 40, 3), (0, 0, 240), dtype=np.uint8)     # BGR Red
    crop_yellow = np.full((60, 40, 3), (0, 240, 240), dtype=np.uint8)  # BGR Yellow
    crop_blue = np.full((60, 40, 3), (240, 0, 0), dtype=np.uint8)    # BGR Blue

    # 1. Benchmark runtime per crop
    t0 = time.perf_counter_ns()
    n_runs = 500
    for _ in range(n_runs):
        _ = extract_hsv_histogram_24bin(crop_red)
    t1 = time.perf_counter_ns()
    ms_per_crop = (t1 - t0) / (n_runs * 1e6)
    print(f"\n[HSV Gate Speed] Extraction latency: {ms_per_crop:.4f} ms/crop")
    assert ms_per_crop < 0.2, f"HSV histogram latency {ms_per_crop:.4f} ms exceeded 0.2ms limit"

    # 2. Check appearance distance separation
    hist_red = extract_hsv_histogram_24bin(crop_red)
    hist_yellow = extract_hsv_histogram_24bin(crop_yellow)
    hist_blue = extract_hsv_histogram_24bin(crop_blue)

    dist_diff = 1.0 - float(np.dot(hist_red, hist_yellow))
    dist_same = 1.0 - float(np.dot(hist_red, hist_red))
    dist_blue = 1.0 - float(np.dot(hist_red, hist_blue))

    assert dist_same < 1e-4, f"Identical crop distance should be ~0: {dist_same}"
    assert dist_diff > 0.4, f"Red vs Yellow distance too low: {dist_diff}"
    assert dist_blue > 0.5, f"Red vs Blue distance too low: {dist_blue}"


def test_exact_line_crossing_at_stride_1():
    """Verify all 3 trackers achieve exact ground-truth line-crossing counts at Stride S=1."""
    sim = TrafficPhysicsSimulator(total_frames=120, stride=1)
    line = ((50.0, 360.0), (1230.0, 360.0))
    expected = sim.ground_truth_crossings

    for TrackerCls in [ByteTracker, OCSortTracker, CBIoUTracker]:
        tracker = TrackerCls(stride=1)
        counter = LineCrossingCounter(p1=line[0], p2=line[1])

        for f_idx, frame, dets in sim:
            tracks = [t for t in tracker.update(dets, frame) if t.time_since_update == 0]
            counter.update(tracks, frame_idx=f_idx)

        summary = counter.get_summary()
        print(f"\n[Exact S=1 Counts] {TrackerCls.__name__}: {summary['total_crossings']} / {expected['total_crossings']}")
        assert summary["total_inbound"] == expected["total_inbound"]
        assert summary["total_outbound"] == expected["total_outbound"]
        assert summary["total_crossings"] == expected["total_crossings"]


def test_multirate_striding_superiority_s3_and_s5():
    """
    Evaluates tracking performance on fast weaving motorcycles under frame striding S in {3, 5}.
    Verifies that CBIoUTracker and OCSortTracker maintain superior ID preservation
    and lower counting error compared to ByteTracker.
    """
    for stride in (3, 5):
        sim = TrafficPhysicsSimulator(total_frames=120, stride=stride)
        line = ((50.0, 360.0), (1230.0, 360.0))

        # 1. Run ByteTrack
        bt = ByteTracker(stride=stride)
        bt_counter = LineCrossingCounter(p1=line[0], p2=line[1])
        for f_idx, frame, dets in sim:
            tracks = [t for t in bt.update(dets, frame) if t.time_since_update == 0]
            bt_counter.update(tracks, frame_idx=f_idx)
        bt_res = bt_counter.get_summary()
        bt_err = abs(bt_res["total_crossings"] - 5)

        # 2. Run OCSortTracker
        oc = OCSortTracker(stride=stride)
        oc_counter = LineCrossingCounter(p1=line[0], p2=line[1])
        for f_idx, frame, dets in sim:
            tracks = [t for t in oc.update(dets, frame) if t.time_since_update == 0]
            oc_counter.update(tracks, frame_idx=f_idx)
        oc_res = oc_counter.get_summary()
        oc_err = abs(oc_res["total_crossings"] - 5)

        # 3. Run CBIoUTracker
        cbiou = CBIoUTracker(stride=stride)
        cbiou_counter = LineCrossingCounter(p1=line[0], p2=line[1])
        for f_idx, frame, dets in sim:
            tracks = [t for t in cbiou.update(dets, frame) if t.time_since_update == 0]
            cbiou_counter.update(tracks, frame_idx=f_idx)
        cbiou_res = cbiou_counter.get_summary()
        cbiou_err = abs(cbiou_res["total_crossings"] - 5)

        print(f"\n[Multi-Rate Stride S={stride} Evaluation]")
        print(f"  ByteTrack: Total Crossings={bt_res['total_crossings']}/5, Counting Error={bt_err}")
        print(f"  OC-SORT:   Total Crossings={oc_res['total_crossings']}/5, Counting Error={oc_err}")
        print(f"  CB-IoU:    Total Crossings={cbiou_res['total_crossings']}/5, Counting Error={cbiou_err}")

        # CBIoUTracker maintains zero error via cascaded expansion; OC-SORT maintains equal or lower error than ByteTracker
        assert cbiou_err < bt_err, f"At S={stride}, CB-IoU error {cbiou_err} did not beat ByteTrack {bt_err}"
        assert oc_err <= bt_err, f"At S={stride}, OC-SORT error {oc_err} exceeded ByteTrack {bt_err}"
        assert cbiou_res["total_crossings"] == 5


def test_algebraic_line_crossing_and_annotator():
    """Verify algebraic signed cross-product calculation, crossing detection, and HUD annotation."""
    p1 = (100.0, 500.0)
    p2 = (1000.0, 500.0)

    # 1. Test signed cross product signs
    point_above = (500.0, 400.0)
    point_below = (500.0, 600.0)
    s_above = signed_cross_product(p1, p2, point_above)
    s_below = signed_cross_product(p1, p2, point_below)
    assert s_above < 0, "Point above line should have negative cross product"
    assert s_below > 0, "Point below line should have positive cross product"

    # 2. Test intersection logic
    q1 = (500.0, 400.0)
    q2 = (500.0, 600.0)
    assert segments_intersect(p1, p2, q1, q2) is True
    # Non-intersecting parallel line
    q3 = (500.0, 200.0)
    q4 = (500.0, 300.0)
    assert segments_intersect(p1, p2, q3, q4) is False

    # 3. Test TrafficAnnotator drawing
    annotator = TrafficAnnotator(line_p1=p1, line_p2=p2)
    counter = LineCrossingCounter(p1=p1, p2=p2)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    dummy_track = Detection(
        box=(300.0, 450.0, 380.0, 550.0),
        score=0.9,
        class_id=2,
        class_name="car",
    )
    from src.tracker import Track
    track_obj = Track(
        track_id=1,
        box=dummy_track.box,
        class_id=2,
        class_name="car",
        score=0.9,
        history=[(340.0, 480.0), (340.0, 520.0)],
    )

    annotated = annotator.annotate(
        frame_bgr=canvas,
        tracks=[track_obj],
        counter=counter,
        engine_name="ORT_INT8_TierS",
        stride=2,
        fps=45.2,
        latency_p95_ms=17.5,
    )
    assert annotated.shape == canvas.shape
    assert np.any(annotated > 0)
