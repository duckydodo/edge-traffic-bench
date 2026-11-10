"""
Model exporter creating the Non-VNNI AVX2 quantization matrix and OpenVINO IR.
Outputs 6 formats stored in models/{model_name}_{imgsz}/:
  1. fp32.onnx (Baseline FP32, opset 17, simplified)
  2. int8_u8s8_full.onnx (Static QDQ INT8, U8 activations, S8 weights, per-channel, reduce_range=False)
  3. int8_u8s8_7bit.onnx (Static QDQ INT8, U8 activations, S8 weights, per-channel, reduce_range=True)
  4. int8_u8u8.onnx (Static QDQ INT8, U8 activations, U8 weights, per-tensor, reduce_range=False)
  5. int8_u8s8_tier_s.onnx (Tier-S Zen 3: 7-bit + Tail-60 Head Node Exclusion + Percentile 99.999% symmetric calib)
  6. openvino_fp32/ (OpenVINO IR model.xml + model.bin converted from fp32.onnx)
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
import openvino as ov
import psutil
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

from src.calibration import TrafficCalibrationDataReader


def get_head_nodes_to_exclude(
    onnx_model_path: Union[str, Path],
    max_tail_nodes: int = 60,
) -> List[str]:
    """
    Inspects the ONNX graph and identifies the post-neck detection head / DFL /
    coordinate decode nodes (or topological tail nodes leading to the final output)
    to exclude from quantization, preventing coordinate drift and mAP collapse.
    
    Traces backwards from the graph output(s) to collect the ancestor subgraph
    (Softmax, integral projection Conv, Sigmoid, Sub, Add, Mul, Div, Concat)
    and supplements with topological tail nodes.
    """
    model = onnx.load(str(onnx_model_path))
    graph = model.graph

    producer_map: Dict[str, Any] = {}
    for node in graph.node:
        for out in node.output:
            producer_map[out] = node

    visited: List[str] = []
    visited_set = set()
    queue = [out.name for out in graph.output]

    # 1. Backwards BFS traversal from graph outputs
    while queue and len(visited) < max_tail_nodes:
        tensor_name = queue.pop(0)
        node = producer_map.get(tensor_name)
        if node and node.name and node.name not in visited_set:
            visited.append(node.name)
            visited_set.add(node.name)
            for inp in node.input:
                if inp in producer_map:
                    queue.append(inp)

    # 2. Complement with topological tail nodes if fewer than max_tail_nodes
    if len(visited) < max_tail_nodes:
        for node in reversed(graph.node):
            if node.name and node.name not in visited_set:
                visited.append(node.name)
                visited_set.add(node.name)
                if len(visited) >= max_tail_nodes:
                    break

    return visited[:max_tail_nodes]


def export_fp32_onnx(
    pt_model_path: Union[str, Path],
    output_onnx_path: Union[str, Path],
    imgsz: Union[int, Tuple[int, int]] = 640,
    opset: int = 17,
) -> Path:
    """Exports PyTorch model (.pt) to simplified FP32 ONNX using Ultralytics."""
    pt_path = Path(pt_model_path)
    output_path = Path(output_onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = imgsz if isinstance(imgsz, int) else imgsz[0]
    model = YOLO(str(pt_path))
    exported_file = model.export(
        format="onnx",
        imgsz=size,
        opset=opset,
        simplify=True,
    )
    exported_path = Path(exported_file)
    if exported_path.resolve() != output_path.resolve():
        shutil.move(str(exported_path), str(output_path))

    return output_path


def quantize_onnx_model(
    preprocessed_onnx: Union[str, Path],
    output_onnx_path: Union[str, Path],
    calibration_reader: CalibrationDataReader,
    activation_type: QuantType = QuantType.QUInt8,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = True,
    reduce_range: bool = False,
    nodes_to_exclude: Optional[List[str]] = None,
    op_types_to_quantize: Optional[List[str]] = None,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Executes memory-safe static QDQ quantization.
    Restricts op_types_to_quantize to ['Conv', 'MatMul'] to prevent RAM explosion.
    """
    output_path = Path(output_onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if op_types_to_quantize is None:
        op_types_to_quantize = ["Conv", "MatMul"]

    opts = dict(extra_options or {})

    calibration_reader.rewind()
    try:
        quantize_static(
            model_input=str(preprocessed_onnx),
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=op_types_to_quantize,
            per_channel=per_channel,
            reduce_range=reduce_range,
            activation_type=activation_type,
            weight_type=weight_type,
            nodes_to_exclude=nodes_to_exclude,
            calibrate_method=calibrate_method,
            extra_options=opts,
        )
    finally:
        gc.collect()

    return output_path


def export_openvino_ir(
    fp32_onnx_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Path:
    """Converts FP32 ONNX model to OpenVINO IR (model.xml + model.bin)."""
    onnx_path = Path(fp32_onnx_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = out_dir / "model.xml"
    ov.save_model(ov_model, str(xml_path))
    del ov_model
    gc.collect()
    return out_dir


def export_model_matrix(
    model_name_or_path: Union[str, Path] = "yolo11n.pt",
    imgsz: int = 320,
    base_output_dir: Union[str, Path] = "models",
    calibration_data_dir: Optional[Union[str, Path]] = None,
    num_calibration_frames: Optional[int] = None,
    include_tier_s: bool = True,
) -> Dict[str, Path]:
    """
    Exports the comprehensive AVX2 quantization and runtime matrix:
      1. fp32.onnx (Baseline FP32)
      2. int8_u8s8_full.onnx (Static QDQ, all nodes quantized, reduce_range=False)
      3. int8_u8s8_full_nohead.onnx (Static QDQ, Tail-60 excluded, reduce_range=False, MinMax)
      4. int8_u8s8_7bit_nohead.onnx (Static QDQ, Tail-60 excluded, reduce_range=True, MinMax)
      5. int8_u8s8_7bit.onnx (Static QDQ, all nodes quantized, reduce_range=True, MinMax)
      6. int8_u8u8.onnx (Static QDQ U8U8, Tail-60 excluded, reduce_range=False)
      7. int8_u8s8_tier_s.onnx (Tier-S Zen 3: Tail-60 excluded, 7-bit + 99.999% Percentile Symmetric)
      8. openvino_fp32/ (OpenVINO IR model.xml + model.bin)
    """
    if num_calibration_frames is None:
        num_calibration_frames = 16 if imgsz <= 320 else 8

    model_path = Path(model_name_or_path)
    model_stem = model_path.stem
    target_dir = Path(base_output_dir) / f"{model_stem}_{imgsz}"
    target_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {}

    def _log_mem(step_name: str) -> None:
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"    -> {step_name} completed [RSS: {rss:.1f} MB]")

    # 1. FP32 ONNX
    fp32_path = target_dir / "fp32.onnx"
    print(f"[*] Step 1/8: Exporting FP32 ONNX to {fp32_path}...")
    export_fp32_onnx(model_path, fp32_path, imgsz=imgsz, opset=17)
    artifacts["fp32"] = fp32_path
    _log_mem("FP32 ONNX")

    # Pre-process ONNX graph with shape inference before static quantization
    preprocessed_onnx = target_dir / "_temp_preprocessed.onnx"
    print(f"[*] Preprocessing ONNX shape inference for static quantization...")
    quant_pre_process(str(fp32_path), str(preprocessed_onnx))

    try:
        # Identify detection head tail nodes for exclusion
        head_nodes = get_head_nodes_to_exclude(preprocessed_onnx, max_tail_nodes=60)
        print(f"[*] Identified {len(head_nodes)} detection-head tail nodes for selective quantization exclusion.")

        # Prepare calibration reader (defaults to data/coco128/images/train2017)
        calib_reader = TrafficCalibrationDataReader(
            data_dir=calibration_data_dir,
            imgsz=imgsz,
            num_frames=num_calibration_frames,
        )
        print(f"[*] Calibration reader initialized with {num_calibration_frames} real image frames.")

        # 2. INT8 U8S8 Full (all nodes quantized, exposes head quantization + VPMADDUBSW saturation)
        u8s8_full_path = target_dir / "int8_u8s8_full.onnx"
        print(f"[*] Step 2/8: Quantizing Static QDQ INT8 U8S8 Full (all nodes) to {u8s8_full_path}...")
        quantize_onnx_model(
            preprocessed_onnx=preprocessed_onnx,
            output_onnx_path=u8s8_full_path,
            calibration_reader=calib_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
        )
        artifacts["int8_u8s8_full"] = u8s8_full_path
        _log_mem("INT8 U8S8 Full")

        # 3. INT8 U8S8 Full No-Head (isolates pure VPMADDUBSW saturation in backbone/neck without head noise)
        u8s8_full_nh_path = target_dir / "int8_u8s8_full_nohead.onnx"
        print(f"[*] Step 3/8: Quantizing Static QDQ INT8 U8S8 Full (Tail-60 excluded) to {u8s8_full_nh_path}...")
        quantize_onnx_model(
            preprocessed_onnx=preprocessed_onnx,
            output_onnx_path=u8s8_full_nh_path,
            calibration_reader=calib_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            nodes_to_exclude=head_nodes,
            calibrate_method=CalibrationMethod.MinMax,
        )
        artifacts["int8_u8s8_full_nohead"] = u8s8_full_nh_path
        _log_mem("INT8 U8S8 Full No-Head")

        # 4. INT8 U8S8 7-bit No-Head (isolates exact gain of 7-bit reduce_range on Zen 3 with head excluded)
        u8s8_7bit_nh_path = target_dir / "int8_u8s8_7bit_nohead.onnx"
        print(f"[*] Step 4/8: Quantizing Static QDQ INT8 U8S8 7-bit (Tail-60 excluded) to {u8s8_7bit_nh_path}...")
        quantize_onnx_model(
            preprocessed_onnx=preprocessed_onnx,
            output_onnx_path=u8s8_7bit_nh_path,
            calibration_reader=calib_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
            nodes_to_exclude=head_nodes,
            calibrate_method=CalibrationMethod.MinMax,
        )
        artifacts["int8_u8s8_7bit_nohead"] = u8s8_7bit_nh_path
        _log_mem("INT8 U8S8 7-bit No-Head")

        # 5. INT8 U8S8 7-bit (all nodes quantized with 7-bit weights)
        u8s8_7bit_path = target_dir / "int8_u8s8_7bit.onnx"
        print(f"[*] Step 5/8: Quantizing Static QDQ INT8 U8S8 7-bit (all nodes) to {u8s8_7bit_path}...")
        quantize_onnx_model(
            preprocessed_onnx=preprocessed_onnx,
            output_onnx_path=u8s8_7bit_path,
            calibration_reader=calib_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )
        artifacts["int8_u8s8_7bit"] = u8s8_7bit_path
        _log_mem("INT8 U8S8 7-bit")

        # 6. INT8 U8U8 Unsigned with Head Excluded
        u8u8_path = target_dir / "int8_u8u8.onnx"
        print(f"[*] Step 6/8: Quantizing Static QDQ INT8 U8U8 (Tail-60 excluded) to {u8u8_path}...")
        quantize_onnx_model(
            preprocessed_onnx=preprocessed_onnx,
            output_onnx_path=u8u8_path,
            calibration_reader=calib_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False,
            nodes_to_exclude=head_nodes,
        )
        artifacts["int8_u8u8"] = u8u8_path
        _log_mem("INT8 U8U8")

        # 7. INT8 U8S8 Tier-S (Zen 3 Optimized Spec: 7-bit + Tail-60 Excluded + Asymmetric Activation Calibration)
        if include_tier_s:
            tier_s_path = target_dir / "int8_u8s8_tier_s.onnx"
            print(f"[*] Step 7/8: Quantizing Tier-S Zen 3 INT8 (Tail-60 Excluded, Asymmetric) to {tier_s_path}...")
            quantize_onnx_model(
                preprocessed_onnx=preprocessed_onnx,
                output_onnx_path=tier_s_path,
                calibration_reader=calib_reader,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
                nodes_to_exclude=head_nodes,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={
                    "CalibTensorRangeSymmetric": False,
                },
            )
            artifacts["int8_u8s8_tier_s"] = tier_s_path
            _log_mem("INT8 Tier-S")
    finally:
        # Clean up intermediate preprocessed graph to conserve disk space
        if preprocessed_onnx.exists():
            preprocessed_onnx.unlink()
        gc.collect()

    # 8. OpenVINO FP32 IR
    openvino_dir = target_dir / "openvino_fp32"
    print(f"[*] Step 8/8: Converting FP32 ONNX to OpenVINO IR in {openvino_dir}...")
    export_openvino_ir(fp32_path, openvino_dir)
    artifacts["openvino_fp32"] = openvino_dir
    _log_mem("OpenVINO FP32 IR")

    print(f"[+] All {len(artifacts)} model artifacts exported to {target_dir}")
    return artifacts
