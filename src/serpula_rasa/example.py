# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Example Pythonic image-based single-cell profiling
#
# The following is an example of performing image-based single-cell profiling using
# Python tools without leaving the kernel for GUI-based interaction.
#
# - load ExampleHuman images from the CellProfiler/examples repo
# - segment nuclei with Cellpose (v3 or v4)
# - extract features with cp_measure (classic CP-style measures)
# - write incrementally to LanceDB:
#     * images_raw: raw images as flattened lists (+ shape/dtype)
#     * masks: segmentation masks as flattened lists (+ lineage to image & algo)
#     * nuclei_features: per-object features (cp_measure)
#     * run_log: per-file processing status

# +
from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from pprint import pp
from shutil import copy2, copytree
from typing import Dict, List, Tuple

import cellpose
import imageio.v3 as iio
import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import models as cp_models
from serpula_rasa.image import ingest_ome_images_ome_arrow, show_images_from_lance

try:
    from cellpose import version as CP_VERSION
except Exception:
    CP_VERSION = "unknown"
from cp_measure.bulk import get_core_measurements

REPO_URL = "https://github.com/CellProfiler/examples.git"
SUBPATH = "ExampleHuman"
DEST_DIR = pathlib.Path("data/ExampleHuman")

IMAGES_DIR = Path("data/ExampleHuman/images")
LANCE_DIR = Path("./data/lance_db")

TABLE_FEATURES = "compartment_nuclei"
TABLE_LOG = "run_log"
TABLE_IMAGES = "images"
TABLE_MASKS = "images"

MODEL_TYPE = "nuclei"  # Cellpose pretrained nuclei model
GPU = True  # set True if you have CUDA ready
MIN_OBJECTS_TO_SAVE = 1
CHANNELS = (0, 0)  # single-channel nuclei, compatible with Cellpose v3/v4

# remove the lance database to refresh on each run
if pathlib.Path(LANCE_DIR).is_dir():
    shutil.rmtree(LANCE_DIR)

# +
# if we don't have the data already, create it
if not pathlib.Path(DEST_DIR).exists():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Clone shallowly for speed
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, tmp], check=True)

        src = Path(tmp) / SUBPATH
        if not src.exists():
            raise FileNotFoundError(f"Subpath {SUBPATH!r} not found in the repo.")

        # Copy the CONTENTS of ExampleHuman directly under data/
        for entry in src.iterdir():
            target = DEST_DIR / entry.name
            if entry.is_dir():
                copytree(entry, target, dirs_exist_ok=True)
            else:
                copy2(entry, target)

pp(list(pathlib.Path(DEST_DIR).rglob("*")))

# +
# create functions which can help us run the pipeline


def ndarray_to_list(arr: np.ndarray) -> List[float | int]:
    """
    Flatten to a Python list in C-order.
    Cast integer arrays to int (to keep sizes smaller) and float arrays to float.
    """
    if np.issubdtype(arr.dtype, np.integer):
        return arr.ravel(order="C").astype(np.int64, copy=False).tolist()
    else:
        return arr.ravel(order="C").astype(np.float32, copy=False).tolist()


def to_float01(img: np.ndarray) -> np.ndarray:
    """Scale image to float32 in [0,1] as cp_measure expects."""
    if img.dtype in (np.float32, np.float64):
        return np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)
    img = img.astype(np.float32, copy=False)
    if np.issubdtype(img.dtype, np.integer):
        maxv = float(np.iinfo(img.dtype).max)
    else:
        maxv = float(img.max() or 1.0)
    if maxv != 0:
        img /= maxv
    return np.clip(img, 0.0, 1.0)


def list_images(root: Path) -> List[Path]:
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


# ---------- LanceDB helpers (lazy creation, schema-safe) ----------


def get_or_create_table(
    db: lancedb.db.LanceDBConnection, table_name: str, df_first_batch: pd.DataFrame
) -> lancedb.table.LanceTable:
    """
    Lazily create (or safely recreate) a table from the first batch df.
    If an incompatible table exists, drop and recreate with the new schema.
    """
    names = set(db.table_names())
    if table_name not in names:
        return db.create_table(table_name, df_first_batch)

    tbl = db.open_table(table_name)
    try:
        # quick compatibility probe
        tbl.add(df_first_batch.iloc[0:0])
        return tbl
    except Exception:
        print(f"[lancedb] Recreating table '{table_name}' with current schema.")
        db.drop_table(table_name)
        return db.create_table(table_name, df_first_batch)


def ensure_log_table(
    db: lancedb.db.LanceDBConnection, name: str
) -> lancedb.table.LanceTable:
    names = set(db.table_names())
    if name not in names:
        return db.create_table(
            name, pd.DataFrame([{"image": "", "status": "", "n_objects": 0}])
        ).delete("n_objects == 0")


# ---------- Cellpose v3/v4 compatibility ----------


def make_cellpose_model(model_type: str, gpu: bool) -> cellpose.models.CellposeModel:
    """
    Return a Cellpose model instance that works with Cellpose v3 or v4.
    """
    # v3 API
    if hasattr(cp_models, "Cellpose"):
        return cp_models.Cellpose(model_type=model_type, gpu=gpu)

    # v4 API
    if hasattr(cp_models, "CellposeModel"):
        try:
            return cp_models.CellposeModel(gpu=gpu, model_type=model_type)
        except TypeError:
            # alt signature in some builds
            return cp_models.CellposeModel(gpu=gpu, pretrained_model=model_type)

    raise RuntimeError("No compatible Cellpose model class found (v3 or v4).")


def cellpose_eval(
    model: cellpose.models.CellposeModel,
    img_f: np.ndarray,
    channels: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, float]:
    """
    Run model.eval and return (masks, diams_scalar).
    If Cellpose returns a vector of diameters, return its nanmean.
    If missing/unavailable, return NaN.
    """
    result = model.eval(img_f, channels=list(channels))

    # Unpack results across versions
    if isinstance(result, tuple):
        if len(result) >= 4:
            masks, flows, styles, diams = result[:4]
        elif len(result) == 3:
            masks, flows, styles = result
            diams = None
        elif len(result) == 2:
            masks, _ = result
            diams = None
        else:
            masks = result[0]
            diams = None
    else:
        masks, diams = np.asarray(result), None

    masks = np.asarray(masks)

    # Normalize diams to a scalar
    if diams is None:
        diam_scalar = float("nan")
    elif np.isscalar(diams):
        diam_scalar = float(diams)
    else:
        # handle list/ndarray cases
        arr = np.asarray(diams, dtype=np.float64)
        if arr.size == 0:
            print("NEGATIVE!")
            diam_scalar = float("nan")
        else:
            with np.errstate(all="ignore"):
                diam_scalar = float(np.nanmean(arr))

    return masks, diam_scalar


def gather_profiles() -> None:  # noqa: PLR0915, C901
    # Connect LanceDB + ensure log table
    db = lancedb.connect(str(LANCE_DIR))
    ensure_log_table(db, TABLE_LOG)
    log_tbl = db.open_table(TABLE_LOG)

    # Init Cellpose (v3 or v4) + measurement funcs
    model = make_cellpose_model(MODEL_TYPE, GPU)
    measurements = get_core_measurements()

    images = list_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found under: {IMAGES_DIR.resolve()}")

    ome_tbl = ingest_ome_images_ome_arrow(
        db=db,
        col_name="ome-arrow_original",
        table_name=TABLE_IMAGES,            # reuse TABLE_IMAGES name
        image_paths=images,            # demo: first image only
        prefer_dimension_order_xyzct=False, # 2D → "XYCT" hint
    )
    print(f"[ome-arrow] stored {ome_tbl.count_rows()} image(s) in Lance")

    for img_path in [images[0]]:
        try:
            img = iio.imread(img_path)
            # match the ingestion’s channel handling
            if img.ndim == 3 and img.shape[-1] in (2, 3, 4):
                nuc_img = img[..., 0]
            else:
                nuc_img = img

            img_f = to_float01(nuc_img)

            # segment
            masks, diams = cellpose_eval(model, img_f, channels=CHANNELS)

            # store masks (kept as flattened ints in existing table)
            labels = np.unique(masks); labels = labels[labels > 0]
            n_obj = int(labels.size)
            mask_record = pd.DataFrame([{
                "filename": img_path.name,
                "algo_name": "cellpose",
                "algo_version": CP_VERSION,
                "model_type": MODEL_TYPE,
                "channels": str(CHANNELS),
                "n_objects": n_obj,
                "height": int(masks.shape[0]),
                "width": int(masks.shape[1]),
                "dtype": "int32",
                "image": masks.astype(np.int32, copy=False).ravel(order="C").tolist(),
            }])
            get_or_create_table(db, TABLE_MASKS, mask_record)

            # features
            if n_obj < MIN_OBJECTS_TO_SAVE:
                log_tbl.add([{
                    "image_filename": str(img_path),
                    "status": "no_objects",
                    "n_objects": n_obj,
                }])
                print(f"- {img_path.name}: no objects")
                continue

            feature_arrays: Dict[str, np.ndarray] = {}
            for _, fn in measurements.items():
                res = fn(masks, img_f)
                feature_arrays.update(res)

            rows = []
            for i, obj_id in enumerate(labels, start=0):
                row = {"image_filename": img_path.name, "nuclei_object_number": int(obj_id)}
                for feat_name, arr in feature_arrays.items():
                    if i < len(arr):
                        val = arr[i]
                        try:
                            row[feat_name] = float(val) if np.ndim(val) == 0 else float(np.array(val).item())
                        except Exception:
                            row[feat_name] = float("nan")
                rows.append(row)
            df_features = pd.DataFrame(rows)
            get_or_create_table(db, TABLE_FEATURES, df_features)

            log_tbl.add([{"image": str(img_path), "status": "ok", "n_objects": n_obj}])
            print(f"✓ {img_path.name}: {n_obj} objects")

        except Exception as ex:
            log_tbl.add([{
                "image": str(img_path),
                "status": f"error: {type(ex).__name__}: {ex}",
                "n_objects": 0,
            }])
            print(f"✗ {img_path.name}: {ex}")


# -

# %%time
# run the pipeline and show the time duration
gather_profiles()

db = lancedb.connect(LANCE_DIR)
db.table_names()

# show nuclei features
db.open_table("compartment_nuclei").to_pandas()

# show the images table
db.open_table("images").to_pandas()

# show the images table
db.open_table("masks").to_pandas()

show_images_from_lance(db_path=LANCE_DIR, table_name="images", col_name="ome-arrow_original", max_images=4, pick="first", cmap="gray")

for _, row in db.open_table("masks").to_pandas().iterrows():
    mask = np.array(row["image"], dtype=np.int32).reshape(row["height"], row["width"])
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Mask ({row['algo_name']} {row['algo_version']}): {row['filename']}")
    plt.axis("off")
    plt.show()
