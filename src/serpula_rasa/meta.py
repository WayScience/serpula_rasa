import numpy as np
import pyarrow as pa

# Column name:  "ome_arrow"       (single column table)
# Type tag:     type="ome.arrow"  (quick validation)
# Version tag:  "0.0.1"           (bump on schema changes)

OME_ARROW_TAG_TYPE = "ome.arrow"
OME_ARROW_TAG_VERSION = "0.0.1"

# GLOBAL Arrow types

# OME_ARROW_STRUCT: ome-arrow record (describes one image/value).
#  - type/version: quick identity & evolution.
#  - id/name/acquisition_datetime: identity & provenance.
#  - pixels_meta: pixels struct (sizes, units, channels).
#  - planes: list of planes struct entries, one per (t,c,z).
#  - masks: reserved for future labels/ROIs (placeholder).
OME_ARROW_STRUCT: pa.StructType = pa.struct(
    [
        pa.field("type", pa.string()),  # must be "ome.arrow"
        pa.field("version", pa.string()),  # e.g., "1.0.0"
        pa.field("id", pa.string()),  # stable image identifier
        pa.field("name", pa.string()),  # human label
        pa.field("acquisition_datetime", pa.timestamp("us")),
        # PIXELS: OME-like "Pixels" header summarizing shape & scale.
        #  - dimension_order: hint like "XYZCT" (or "XYCT" when Z==1).
        #  - type: numeric storage type (e.g., "uint16").
        #  - size_*: axis lengths.
        #  - physical_size_* (+ *_unit): microscope scale in micrometers.
        #  - channels: list of channel struct entries (one per channel).
        pa.field(
            "pixels_meta",
            pa.struct(
                [
                    pa.field("dimension_order", pa.string()),  # "XYZCT" / "XYCT"
                    pa.field("type", pa.string()),  # "uint8","uint16","float",...
                    pa.field("size_x", pa.int32()),  # width  (pixels)
                    pa.field("size_y", pa.int32()),  # height (pixels)
                    pa.field("size_z", pa.int32()),  # z-slices (1 for 2D)
                    pa.field("size_c", pa.int16()),  # channels
                    pa.field("size_t", pa.int32()),  # time points
                    pa.field("physical_size_x", pa.float32()),  # µm per pixel (X)
                    pa.field("physical_size_y", pa.float32()),  # µm per pixel (Y)
                    pa.field("physical_size_z", pa.float32()),  # µm per z-step
                    pa.field("physical_size_x_unit", pa.string()),  # usually "µm"
                    pa.field("physical_size_y_unit", pa.string()),
                    pa.field("physical_size_z_unit", pa.string()),
                    
                    pa.field(
                        "channels",
                        pa.list_(
                            # CHANNELS: one entry per channel (e.g., DNA, Mito, ER).
                            #  - emission_um / excitation_um: wavelengths (micrometers) for context.
                            #  - illumination: modality (e.g., "Epifluorescence").
                            #  - color_rgba: preferred display color (packed 0xRRGGBBAA).
                            pa.struct(
                                [
                                    pa.field("id", pa.string()),
                                    pa.field("name", pa.string()),
                                    pa.field("emission_um", pa.float32()),
                                    pa.field("excitation_um", pa.float32()),
                                    pa.field("illumination", pa.string()),
                                    pa.field("color_rgba", pa.uint32()),
                                ]
                            )
                        ),
                    ),
                ]
            ),
        ),
        # PLANES: one 2D image plane for a specific (t, c, z).
        #  - pixels: flattened numeric list (Y*X) for analysis-ready computation.
        pa.field(
            "planes",
            pa.list_(
                pa.struct(
                    [
                        pa.field("z", pa.int32()),
                        pa.field("t", pa.int32()),
                        pa.field("c", pa.int16()),
                        pa.field(
                            "pixels", pa.list_(pa.uint16())
                        ),  # keep numeric (not PNG/JPEG)
                    ]
                )
            ),
        ),
        pa.field("masks", pa.null()),  # reserved for future annotations
    ]
)

# OME_ARROW_SCHEMA: a one-column table named 'ome_arrow' holding the struct.
OME_ARROW_SCHEMA: pa.Schema = pa.schema([pa.field("ome_arrow", OME_ARROW_STRUCT)])


# -----------------------------------------------------------------------------
# Builders & IO (unchanged behavior, now referencing the globals above)
# -----------------------------------------------------------------------------

OME_DTYPE_MAP = {
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("int16"): "int16",
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
}
