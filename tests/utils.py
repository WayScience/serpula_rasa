"""
Image test fixtures.
"""

import pathlib

import numpy as np
import tifffile as tiff


def write_synth_tiff(path: pathlib.Path, shape=(64, 96), dtype=np.uint16) -> np.ndarray:
    """Create and save a simple ramp image; return the array used."""
    arr = (
        np.arange(shape[0] * shape[1], dtype=np.uint32) % np.iinfo(dtype).max
    ).astype(dtype)
    arr = arr.reshape(shape)
    tiff.imwrite(str(path), arr)
    return arr
