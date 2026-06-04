"""Loading XCT image stacks from disk.

XCT acquisitions are stored as a sequence of per-slice grayscale TIFF images.
These helpers stack the slices into a single ``(H, W, D)`` volume (or the
``(D, H, W)`` order expected by GPU labeling backends).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_tiff_stack(
    directory: str | Path,
    pattern: str = "*.tif",
    crop: tuple[int, int, int, int] | None = None,
    dtype=np.uint8,
):
    """Load a directory of grayscale TIFF slices into a ``(H, W, D)`` volume.

    Parameters
    ----------
    directory:
        Folder containing the XCT slices.
    pattern:
        Glob pattern for the slices (sorted by trailing slice index).
    crop:
        Optional ``(row0, row1, col0, col1)`` region of interest applied to
        every slice.
    dtype:
        Output dtype (XCT is typically 8-bit or 16-bit grayscale).

    Returns
    -------
    np.ndarray
        Volume of shape ``(H, W, D)``.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern), key=_slice_index)
    if not files:
        raise FileNotFoundError(f"No slices matching {pattern!r} in {directory}")

    slices = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            raise IOError(f"Could not read {f}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if crop is not None:
            r0, r1, c0, c1 = crop
            gray = gray[r0:r1, c0:c1]
        slices.append(gray.astype(dtype))
    return np.stack(slices, axis=2)


def to_zyx(volume: np.ndarray) -> np.ndarray:
    """Reorder a ``(H, W, D)`` volume to ``(D, H, W)`` (slice-first).

    GPU labeling backends such as pyclesperanto expect the depth axis first.
    """
    return volume.transpose(2, 0, 1)


def _slice_index(path: Path) -> int:
    """Extract a trailing integer slice index from a filename for sorting."""
    stem = path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else 0
