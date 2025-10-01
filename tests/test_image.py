"""
Tests for ome-arrow work
"""

import numpy as np
import tifffile as tiff
import pyarrow.parquet as pq
import pytest
import matplotlib

import tifffile as tiff

# import your ome-arrow helpers/globals
from serpula_rasa.image import (
    make_ome_arrow_row,
    write_ome_arrow_parquet,
    validate_ome_arrow_table,
    reconstruct_tczyx_from_record,
    ingest_ome_arrow_to_lance,
    show_images_from_lance,
)
from serpula_rasa.meta import OME_ARROW_TAG_TYPE, OME_ARROW_TAG_VERSION

matplotlib.use("Agg")


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

# test_ome_arrow_lance_ingest.py
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.utils import write_synth_tiff


def test_ingest_ome_arrow_to_lance_roundtrip(tmp_db: Path, tmp_path: Path):
    # 1) Create a couple of synthetic TIFFs
    img1 = write_synth_tiff(tmp_path / "a.tif", shape=(32, 48), dtype=np.uint16)
    img2 = write_synth_tiff(tmp_path / "b.tif", shape=(40, 40), dtype=np.uint8)

    # 2) Ingest into LanceDB as ome-arrow structs (one row per image)
    table = ingest_ome_arrow_to_lance(
        image_paths=[tmp_path / "a.tif", tmp_path / "b.tif"],
        db_path=tmp_db,
        table_name="ome_images",
        prefer_dimension_order_xyzct=False,   # 2D → "XYCT" hint
        batch_size=2,
    )

    # 3) Lance table exists and has 2 rows
    assert table.count_rows() == 2

    # 4) Pull back as Arrow, validate schema + type/version tags
    arr_tbl = table.to_arrow().select(["ome_arrow"])
    validate_ome_arrow_table(arr_tbl)

    # 5) Reconstruct arrays and compare with the source TIFFs (bit-exact)
    records = [arr_tbl["ome_arrow"][i].as_py() for i in range(len(arr_tbl))]
    # map by id (stem)
    rec_by_id = {rec["id"]: rec for rec in records}

    # a.tif
    rec_a = rec_by_id["a"]
    arr_a = reconstruct_tczyx_from_record(rec_a)  # (T,C,Z,Y,X)
    assert arr_a.shape == (1, 1, 1, 32, 48)
    np.testing.assert_array_equal(arr_a[0, 0, 0], img1)

    # b.tif
    rec_b = rec_by_id["b"]
    arr_b = reconstruct_tczyx_from_record(rec_b)
    assert arr_b.shape == (1, 1, 1, 40, 40)
    # stored as uint16; values equal after dtype cast
    np.testing.assert_array_equal(arr_b[0, 0, 0].astype(img2.dtype), img2)

    # 6) Type/version tags present
    for rec in records:
        assert rec["type"] == OME_ARROW_TAG_TYPE
        assert rec["version"] == OME_ARROW_TAG_VERSION

def test_show_images_from_lance_smoke(tmp_db: Path, tmp_path: Path):
    """
    Smoke-test the visualization helper. We don't assert on plots; we just
    ensure the function runs without raising (using Agg backend).
    """
    # Ingest one tiny image
    _ = write_synth_tiff(tmp_path / "c.tif", shape=(16, 16), dtype=np.uint16)
    table = ingest_ome_arrow_to_lance(
        image_paths=[tmp_path / "c.tif"],
        db_path=tmp_db,
        table_name="ome_images_show",
        prefer_dimension_order_xyzct=False,
        batch_size=1,
    )
    assert table.count_rows() == 1

    # Should not raise
    show_images_from_lance(
        db_path=tmp_db,
        table_name="ome_images_show",
        max_images=1,
        pick="first",
        cmap="gray",
    )
