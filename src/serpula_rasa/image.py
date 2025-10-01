"""
Image utility functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing import Mapping, Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from serpula_rasa.meta import OME_ARROW_SCHEMA, OME_DTYPE_MAP, OME_ARROW_TAG_TYPE, OME_ARROW_TAG_VERSION

@dataclass
class ChannelMeta:
    """Minimal, practical channel metadata for profiling workflows."""
    id: str
    name: str
    emission_um: Optional[float] = None
    excitation_um: Optional[float] = None
    illumination: Optional[str] = None
    color_rgba: Optional[int] = None  # preferred display color (packed RGBA)

def _default_channels(C: int) -> List[ChannelMeta]:
    return [ChannelMeta(id=f"C{c}", name=f"Channel-{c}") for c in range(C)]

def _normalize_to_TCZYX(arr: np.ndarray) -> Tuple[int, int, int, int, int, np.ndarray]:
    """
    Accept common shapes and normalize to (T, C, Z, Y, X).
    Accepted: (Y,X), (C,Y,X), (Z,Y,X), (C,Z,Y,X), (T,C,Y,X), (T,C,Z,Y,X)
    Heuristics:
      • For 3D, small first dim (<=8) ⇒ channels; otherwise z-slices.
      • For 4D, if last two look spatial and the 2nd dim small (<=8) ⇒ (T,C,Y,X).
    """
    if arr.ndim == 2:  # (Y, X)
        Y, X = arr.shape
        return (1, 1, 1, Y, X, arr[None, None, None, :, :])

    if arr.ndim == 3:
        a, b, c = arr.shape
        if a <= 8:      # (C, Y, X)
            C, Y, X = a, b, c
            return (1, C, 1, Y, X, arr[None, :, None, :, :])
        else:           # (Z, Y, X)
            Z, Y, X = a, b, c
            return (1, 1, Z, Y, X, arr[None, None, :, :, :])

    if arr.ndim == 4:
        A, B, C_, D_ = arr.shape
        if C_ > 16 and D_ > 16 and B <= 8:  # (T, C, Y, X)
            T, C, Y, X = A, B, C_, D_
            return (T, C, 1, Y, X, arr[:, :, None, :, :])
        # (C, Z, Y, X)
        C, Z, Y, X = A, B, C_, D_
        return (1, C, Z, Y, X, arr[None, :, :, :, :])

    if arr.ndim == 5:  # (T, C, Z, Y, X)
        T, C, Z, Y, X = arr.shape
        return (T, C, Z, Y, X, arr)

    raise ValueError("Expected (Y,X),(C,Y,X),(Z,Y,X),(C,Z,Y,X),(T,C,Y,X),(T,C,Z,Y,X).")

def _coerce_pixel_dtype(arr5: np.ndarray) -> Tuple[np.ndarray, str]:
    """Keep numeric pixels in a widely-supported dtype; default to uint16."""
    dt = arr5.dtype
    if dt not in OME_DTYPE_MAP:
        if np.issubdtype(dt, np.floating):
            arr5 = arr5.astype(np.uint16, copy=False)
            dt = arr5.dtype
        else:
            arr5 = arr5.astype(np.uint16, copy=False)
            dt = arr5.dtype
    return arr5, OME_DTYPE_MAP[np.dtype(dt)]


def make_ome_arrow_row(
    *,
    image_id: str,
    name: str,
    pixels: np.ndarray,                 # XY or XYZ (+ optional C/T)
    channel_meta: Optional[Sequence[ChannelMeta]] = None,
    physical_size_xy_um: float = 0.108, # microscope pixel size (µm/px)
    physical_size_z_um: float = 1.0,    # z-step (µm)
    physical_unit: str = "µm",
    prefer_dimension_order_xyzct: bool = True,  # if Z==1 and False → "XYCT"
    acquisition_dt: Optional[datetime] = None,
) -> Mapping[str, object]:
    """
    Build ONE 'ome_arrow' struct (ONE image per row).
    - Pixels remain numeric (no PNG/JPEG), ready for analysis.
    - Planes are 2D slices per (t,c,z), flattened to Y*X for compact storage.
    """
    # Normalize to (T,C,Z,Y,X) and pick storage dtype
    T, C, Z, Y, X, arr5 = _normalize_to_TCZYX(pixels)
    arr5, pixel_type_str = _coerce_pixel_dtype(arr5)

    # Channels metadata (default label if none supplied)
    channels = list(channel_meta or _default_channels(C))
    if len(channels) != C:
        raise ValueError(f"channel_meta length {len(channels)} != C {C}")
    channels_field = [
        {
            "id": ch.id,
            "name": ch.name,
            "emission_um": ch.emission_um,
            "excitation_um": ch.excitation_um,
            "illumination": ch.illumination,
            "color_rgba": ch.color_rgba,
        }
        for ch in channels
    ]

    # Dimension order string (human hint)
    dimension_order = "XYZCT" if (Z > 1 or prefer_dimension_order_xyzct) else "XYCT"

    # Assemble per-plane numeric pixels (one entry per (t,c,z))
    planes: List[Mapping[str, object]] = []
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                plane = arr5[t, c, z]  # (Y, X)
                planes.append({
                    "z": z,
                    "t": t,
                    "c": c,
                    "pixels": plane.reshape(-1).astype(np.uint16).tolist(),
                })

    pixels_meta = {
        "dimension_order": dimension_order,
        "type": pixel_type_str,
        "size_x": int(X), "size_y": int(Y), "size_z": int(Z),
        "size_c": int(C), "size_t": int(T),
        "physical_size_x": float(physical_size_xy_um),
        "physical_size_y": float(physical_size_xy_um),
        "physical_size_z": float(physical_size_z_um),
        "physical_size_x_unit": physical_unit,
        "physical_size_y_unit": physical_unit,
        "physical_size_z_unit": physical_unit,
        "channels": channels_field,
    }

    return {
        "ome_arrow": {
            "type": OME_ARROW_TAG_TYPE,
            "version": OME_ARROW_TAG_VERSION,
            "id": image_id,
            "name": name,
            "acquisition_datetime": acquisition_dt or datetime.now(timezone.utc),
            "pixels_meta": pixels_meta,
            "planes": planes,
            "masks": None,
        }
    }


def validate_ome_arrow_table(table: pa.Table) -> None:
    """
    Validate the table matches OME_ARROW_SCHEMA and rows carry the correct tags.
    Raises ValueError on mismatch.
    """
    if table.schema.types != OME_ARROW_SCHEMA.types or table.schema.names != OME_ARROW_SCHEMA.names:
        raise ValueError("Schema mismatch: table does not match OME_ARROW_SCHEMA.")
    col = table["ome_arrow"]
    for i in range(len(col)):
        rec = col[i].as_py()
        if rec.get("type") != OME_ARROW_TAG_TYPE:
            raise ValueError(f"Row {i}: type tag != '{OME_ARROW_TAG_TYPE}'")
        if rec.get("version") != OME_ARROW_TAG_VERSION:
            raise ValueError(f"Row {i}: version tag != '{OME_ARROW_TAG_VERSION}'")


def write_ome_arrow_parquet(
    rows: Iterable[Mapping[str, object]],
    path: str,
    row_group_size: int = 1,
) -> None:
    """
    Write ome-arrow rows to Parquet with settings tuned for numeric arrays.
    """
    table = pa.Table.from_pylist(list(rows), schema=OME_ARROW_SCHEMA)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        data_page_version="2.0",
        row_group_size=row_group_size,  # often 1 row = 1 image; batch if desired
    )


def read_ome_arrow_first(path: str) -> Mapping[str, object]:
    """Quick peek: load the first ome-arrow record (no reshaping)."""
    tbl = pq.read_table(path, columns=["ome_arrow"])
    validate_ome_arrow_table(tbl)
    return tbl["ome_arrow"][0].as_py()



def reconstruct_tczyx_from_record(rec: Mapping[str, Any]) -> np.ndarray:
    """
    Reconstruct a 5D NumPy array (T, C, Z, Y, X) from an in-memory ome_arrow record.

    The `ome_arrow` struct stores image pixel data as a list of per-plane
    records (one per timepoint, channel, and z-slice), where each plane
    contains a flattened list of Y*X numeric values. This helper function
    reassembles those planes into a contiguous 5D NumPy array.

    Parameters
    ----------
    rec : Mapping[str, Any]
        A single `ome_arrow` record as returned by `.as_py()` from a PyArrow
        Table column (i.e., `tbl["ome_arrow"][i].as_py()`). The record must
        contain a valid `"pixels_meta"` dict and a list of `"planes"`.

    Returns
    -------
    np.ndarray
        A NumPy array with shape (T, C, Z, Y, X) and dtype ``np.uint16``.
        - T = number of timepoints
        - C = number of channels
        - Z = number of z-slices (1 for 2D images)
        - Y = image height (pixels)
        - X = image width  (pixels)

    Notes
    -----
    • For 2D single-channel images, the returned shape will be (1, 1, 1, Y, X).  
    • The dtype is fixed to ``np.uint16`` because pixel values are stored that way
      in ome-arrow by default. Adjust the coercion logic if you plan to allow
      multiple numeric dtypes.  
    • This function is useful outside of testing, e.g.:
      - Feeding the reconstructed array into scikit-image or napari
      - Exporting back to OME-TIFF or Zarr
      - Performing quantitative analysis with NumPy or PyTorch

    Examples
    --------
    >>> tbl = pq.read_table("example_2d_ome_arrow.parquet")
    >>> rec = tbl["ome_arrow"][0].as_py()
    >>> arr = reconstruct_tczyx_from_record(rec)
    >>> arr.shape
    (1, 1, 1, 64, 96)
    """
    pm = rec["pixels_meta"]
    T, C, Z = pm["size_t"], pm["size_c"], pm["size_z"]
    Y, X = pm["size_y"], pm["size_x"]

    out = np.empty((T, C, Z, Y, X), dtype=np.uint16)
    for pl in rec["planes"]:
        t, c, z = pl["t"], pl["c"], pl["z"]
        out[t, c, z] = np.asarray(pl["pixels"], dtype=np.uint16).reshape(Y, X)
    return out
