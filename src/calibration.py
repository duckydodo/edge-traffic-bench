"""
Calibration data reader for ONNX Runtime static post-training quantization (PTQ).
Feeds preprocessed (1, 3, H, W) float32 frames from real images or deterministic synthetic frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

from src.video_loader import SyntheticVideoGenerator, preprocess_frame


class TrafficCalibrationDataReader(CalibrationDataReader):
    """
    Supplies calibration data to ONNX Runtime static quantization.
    Can ingest frames from a directory of real images or generate deterministic
    synthetic traffic frames when no real image directory is specified.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        imgsz: Union[int, Tuple[int, int]] = 640,
        num_frames: int = 30,
        input_name: str = "images",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if isinstance(imgsz, int):
            self.target_shape = (imgsz, imgsz)
        else:
            self.target_shape = imgsz

        self.data_dir = Path(data_dir) if data_dir else None
        if self.data_dir is None:
            default_coco = Path("data/coco128/images/train2017")
            if default_coco.exists() and default_coco.is_dir():
                self.data_dir = default_coco

        self.num_frames = num_frames
        self.input_name = input_name
        self.seed = seed

        self._frame_paths: List[Path] = []
        if self.data_dir and self.data_dir.exists() and self.data_dir.is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self._frame_paths = sorted([
                p for p in self.data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in valid_exts
            ])[: self.num_frames]

        self._range_slice: Optional[Tuple[int, int]] = None
        self._iterator: Optional[Iterator[np.ndarray]] = None
        self.rewind()

    def _create_generator(self) -> Iterator[np.ndarray]:
        paths = self._frame_paths
        if self._range_slice is not None:
            start, end = self._range_slice
            paths = paths[start:end]

        if paths:
            for p in paths:
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    tensor = preprocess_frame(bgr, target_shape=self.target_shape, normalize=True)
                    yield tensor
        else:
            # Deterministic synthetic traffic frames
            synth = SyntheticVideoGenerator(
                total_frames=self.num_frames,
                width=max(self.target_shape[1], 640),
                height=max(self.target_shape[0], 480),
                stride=1,
                seed=self.seed,
            )
            for _, bgr in synth:
                tensor = preprocess_frame(bgr, target_shape=self.target_shape, normalize=True)
                yield tensor

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Returns the next calibration sample as {input_name: tensor} or None when exhausted."""
        if self._iterator is None:
            self.rewind()

        try:
            sample = next(self._iterator)
            return {self.input_name: sample}
        except StopIteration:
            return None

    def rewind(self) -> None:
        """Resets the data reader to the beginning."""
        self._iterator = self._create_generator()

    def set_range(self, start_index: int, end_index: int) -> None:
        """Sets active slice of dataset for strided calibration."""
        self._range_slice = (start_index, end_index)
        self.rewind()

    def __len__(self) -> int:
        return len(self._frame_paths) if self._frame_paths else self.num_frames
