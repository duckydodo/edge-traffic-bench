"""
Hardware detection, topology inspection, and thread/affinity configuration for edge benchmarking.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psutil


@dataclass
class CPUHardwareInfo:
    cpu_model: str = "Unknown"
    vendor_id: str = "Unknown"
    architecture: str = "x86_64"
    physical_cores: int = 0
    logical_cores: int = 0
    has_avx2: bool = False
    has_vnni: bool = False
    total_memory_gb: float = 0.0
    flags: List[str] = field(default_factory=list)
    topology: Dict[int, List[int]] = field(default_factory=dict)
    primary_physical_cpus: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "vendor_id": self.vendor_id,
            "architecture": self.architecture,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "has_avx2": self.has_avx2,
            "has_vnni": self.has_vnni,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "primary_physical_cpus": self.primary_physical_cpus,
        }


def detect_hardware() -> CPUHardwareInfo:
    """
    Detects CPU architecture, physical vs logical cores, AVX2 and VNNI extensions,
    and cache/topology details from /proc/cpuinfo and system files.
    """
    info = CPUHardwareInfo()
    info.logical_cores = os.cpu_count() or 1
    info.physical_cores = psutil.cpu_count(logical=False) or (info.logical_cores // 2 or 1)
    
    # Memory
    mem = psutil.virtual_memory()
    info.total_memory_gb = mem.total / (1024 ** 3)
    
    cpuinfo_path = Path("/proc/cpuinfo")
    topology: Dict[int, List[int]] = {}

    if cpuinfo_path.exists():
        current_processor = None
        current_core_id = None
        
        with open(cpuinfo_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_processor is not None and current_core_id is not None:
                        topology.setdefault(current_core_id, []).append(current_processor)
                    current_processor = None
                    current_core_id = None
                    continue
                
                if ":" in line:
                    key, val = [part.strip() for part in line.split(":", 1)]
                    if key == "model name" and info.cpu_model == "Unknown":
                        info.cpu_model = val
                    elif key == "vendor_id" and info.vendor_id == "Unknown":
                        info.vendor_id = val
                    elif key == "processor":
                        try:
                            current_processor = int(val)
                        except ValueError:
                            pass
                    elif key == "core id":
                        try:
                            current_core_id = int(val)
                        except ValueError:
                            pass
                    elif key in ("flags", "Features") and not info.flags:
                        flags_list = val.split()
                        info.flags = flags_list
                        flags_lower = [fl.lower() for fl in flags_list]
                        info.has_avx2 = "avx2" in flags_lower
                        # VNNI can appear as avx_vnni, avx512_vnni, or avx512vnni
                        info.has_vnni = any(
                            vnni_flag in flags_lower
                            for vnni_flag in ("avx_vnni", "avx512_vnni", "avx512vnni", "vnni")
                        )

            if current_processor is not None and current_core_id is not None:
                topology.setdefault(current_core_id, []).append(current_processor)

    info.topology = topology
    # Select first logical processor of each physical core
    primary_cpus: List[int] = []
    for core_id in sorted(topology.keys()):
        processors = topology[core_id]
        if processors:
            primary_cpus.append(min(processors))
    
    if not primary_cpus:
        # Fallback if topology parsing didn't find pairs
        primary_cpus = list(range(info.physical_cores))
        
    info.primary_physical_cpus = primary_cpus
    return info


def print_hardware_summary(info: Optional[CPUHardwareInfo] = None) -> None:
    """Pretty prints the hardware summary."""
    if info is None:
        info = detect_hardware()
        
    separator = "=" * 65
    print(separator)
    print("  HARDWARE DETECTION & TOPOLOGY SUMMARY")
    print(separator)
    print(f"  CPU Model:        {info.cpu_model}")
    print(f"  Vendor:           {info.vendor_id}")
    print(f"  Architecture:     {info.architecture}")
    print(f"  Physical Cores:   {info.physical_cores}")
    print(f"  Logical Cores:    {info.logical_cores}")
    print(f"  Total RAM:        {info.total_memory_gb:.2f} GB")
    print(f"  AVX2 Supported:   {'YES' if info.has_avx2 else 'NO'}")
    print(f"  VNNI Supported:   {'YES' if info.has_vnni else 'NO (Zen 3 limitation)'}")
    print(f"  Primary Core Map: {info.primary_physical_cpus}")
    if info.topology:
        print(f"  Core Topology:    {info.topology}")
    print(separator)


def set_cpu_affinity(core_ids: List[int]) -> Set[int]:
    """
    Locks the current process to specific CPU core IDs using os.sched_setaffinity.
    Returns the new affinity set.
    """
    if not hasattr(os, "sched_setaffinity"):
        raise NotImplementedError("CPU affinity binding via sched_setaffinity is only supported on Linux.")
    
    affinity_set = set(core_ids)
    os.sched_setaffinity(0, affinity_set)
    return os.sched_getaffinity(0)


def configure_thread_environment(
    num_threads: int,
    opencv_threads: int = 1,
    extra_env: Optional[Dict[str, str]] = None
) -> None:
    """
    Sets thread control environment variables for OpenMP, MKL, OpenBLAS, etc.,
    and configures OpenCV thread pool to prevent thread oversubscription.
    """
    env_vars = {
        "OMP_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
    }
    if extra_env:
        env_vars.update(extra_env)

    for k, v in env_vars.items():
        os.environ[k] = v

    # Configure OpenCV thread pool if cv2 is imported or available
    if "cv2" in sys.modules:
        import cv2
        cv2.setNumThreads(opencv_threads)
    else:
        try:
            import cv2
            cv2.setNumThreads(opencv_threads)
        except ImportError:
            pass
