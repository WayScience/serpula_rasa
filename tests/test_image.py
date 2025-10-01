"""
Tests for ome-arrow work
"""

import numpy as np
import tifffile as tiff
import pyarrow.parquet as pq
import pytest

# import your ome-arrow helpers/globals
from serpula_rasa.image import (
    make_ome_arrow_row,
    write_ome_arrow_parquet,
    validate_ome_arrow_table,
    reconstruct_tczyx_from_record
)
from serpula_rasa.meta import OME_ARROW_TAG_TYPE, OME_ARROW_TAG_VERSION

@pytest.mark.parametrize("shape,dtype", [((64, 96), np.uint16), ((32, 48), np.uint8)])
def test_ome_arrow_roundtrip_from_2d_tiff(tmp_path, shape, dtype):
    """
    End-to-end:
      1) write synthetic 2D TIFF
      2) build ome_arrow row (XY only)
      3) write Parquet
      4) read + validate
      5) reconstruct array and compare with source
    """
    # 1) synthesize and save a 2D TIFF
    src = (np.arange(shape[0] * shape[1], dtype=np.uint32) % np.iinfo(dtype).max).astype(dtype)
    src = src.reshape(shape)  # (Y, X)
    tiff_path = tmp_path / "img.tif"
    tiff.imwrite(tiff_path, src)

    # 2) load TIFF and build a single ome_arrow row
    img = tiff.imread(tiff_path)  # (Y, X), preserves dtype
    assert img.shape == shape and img.dtype == dtype

    row = make_ome_arrow_row(
        image_id="img_0001",
        name="2D TIFF example",
        pixels=img,                       # accepts (Y, X)
        physical_size_xy_um=0.108,
        physical_size_z_um=1.0,
        physical_unit="µm",
        prefer_dimension_order_xyzct=False,  # use "XYCT" since Z==1
    )

    # 3) write the row to a Parquet file
    parquet_path = tmp_path / "example_2d_ome_arrow.parquet"
    write_ome_arrow_parquet([row], parquet_path, row_group_size=1)
    assert parquet_path.exists()

    # 4) read back and validate schema + tags
    tbl = pq.read_table(parquet_path)
    validate_ome_arrow_table(tbl)
    col = tbl["ome_arrow"]
    assert len(col) == 1

    rec = col[0].as_py()
    assert rec["type"] == OME_ARROW_TAG_TYPE
    assert rec["version"] == OME_ARROW_TAG_VERSION

    pm = rec["pixels_meta"]
    assert pm["size_z"] == 1
    assert pm["size_c"] == 1
    assert pm["size_t"] == 1
    assert pm["size_y"] == shape[0]
    assert pm["size_x"] == shape[1]
    # dimension order hint for Z==1
    assert pm["dimension_order"] in ("XYCT", "XYZCT")

    # 5) reconstruct and compare pixels
    arr = reconstruct_tczyx_from_record(rec)  # (T, C, Z, Y, X)
    assert arr.shape == (1, 1, 1, shape[0], shape[1])
    recon = arr[0, 0, 0]  # (Y, X)
    np.testing.assert_array_equal(recon.astype(dtype), src)

    # quick numeric sanity checks
    assert float(recon.mean()) == pytest.approx(float(src.mean()))
    assert float(recon.max()) == float(src.max())
