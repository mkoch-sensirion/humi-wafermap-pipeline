from itertools import chain

from pyspark import pipelines as dp
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType


# ============================================================================
# SILVER LAYER: Read and aggregate measurements into wafermap dicts
# ============================================================================

@dp.table(
    name="silver_wafermap_raw",
)
def silver_wafermap_raw():
    """
    Read measurement data from ops_prod.dk_silver_data.t_data_measurement
    incrementally
    and aggregate by substrate and measurement series (L3).

    Filters only Castor/Pollux, Process=Prober, SeriesSuffix=L3,
    MeasurementName=CSR_40_01_01. Update column names if they differ
    in your table.
    """

    # Product -> MeasurementName mapping (extend as new products are added)
    product_measurement_map = {
        "Castor": "CSR_40_01_01",
        "Pollux": "PLX_40_01_01",
    }

    measurement_name_expr = F.create_map(
        *chain.from_iterable(
            (F.lit(k), F.lit(v)) for k, v in product_measurement_map.items()
        )
    )

    TABLE_NAME = "ops_prod.dk_silver_data.t_data_measurement"

    raw_data = (
        spark.readStream.table(TABLE_NAME)  # noqa: F821
        .filter(F.col("ValueReal").isNotNull())
        .filter(F.col("Product").isin(list(product_measurement_map.keys())))
        .filter(F.col("Process") == F.lit("Prober"))
        .filter(F.col("SeriesSuffix") == F.lit("L3"))
        .withColumn(
            "expected_measurement_name",
            measurement_name_expr[F.col("Product")]
        )
        .filter(F.col("MeasurementName") == F.col("expected_measurement_name"))
        .filter(
            (
                F.col("MeasurementSeriesStartTime")
                >= F.lit("2026-01-01")
            )
        )  # temporal filter to limit data for testing - adjust as needed
        .select(
            F.col("InputSubstrateName").alias("WaferNr"),
            F.col("SerialNumber").alias("Serial"),
            F.col("MeasurementSession").alias("MeasurementSession"),
            F.col("MeasurementSeries").alias("MeasurementSeries"),
            F.col("SeriesSuffix").alias("SeriesSuffix"),
            F.col("MeasurementName").alias("MeasurementName"),
            F.col("Product").alias("Chip"),
            F.col("X_InputSubstrate").alias("X_OnWafer"),
            F.col("Y_InputSubstrate").alias("Y_OnWafer"),
            F.col("ValueReal").alias("RhMeasurement"),
            F.col("MeasurementSeriesStartTime")
        )
    )

    wafermap_data = (
        raw_data
        .withWatermark("MeasurementSeriesStartTime", "30 days")
        .groupBy(
            "WaferNr",
            "MeasurementSession",
            "MeasurementSeries",
            "MeasurementName",
            "Chip"
        )
        .agg(
            F.map_from_entries(
                F.collect_list(
                    F.struct(
                        F.struct(
                            F.col("X_OnWafer"),
                            F.col("Y_OnWafer"),
                        ).alias("coord"),
                        F.col("RhMeasurement").alias("value")
                    )
                )
            ).alias("WafermapDict"),
            F.count("RhMeasurement").alias("MeasurementCount"),
            F.first("MeasurementSeriesStartTime").alias(
                "MeasurementSeriesStartTime"
            )
        )
    )

    return (
        wafermap_data
        .select(
            F.col("WaferNr"),
            F.col("MeasurementSession"),
            F.col("MeasurementSeries"),
            F.col("MeasurementName"),
            F.col("Chip"),
            F.col("WafermapDict"),
            F.col("MeasurementCount"),
            F.col("MeasurementSeriesStartTime"),
            F.current_timestamp().alias("GeneratedAt"),
            F.lit("silver").alias("layer")
        )
    )


# ===================================================================
# SILVER LAYER: Rasterize wafermap dict to 256x256 image
# ===================================================================

@dp.table(
    name="silver_wafermap_rasterized",
)
def silver_wafermap_rasterized():
    """
    Convert WafermapDict to a 256x256 rasterized numpy-like array.

    - Center values by wafer median
    - Normalize to 0..1
    - Interpolate onto grid defined by chip limits
    - Resize to 256x256
    """
    import io
    import numpy as np
    from scipy.interpolate import griddata
    from PIL import Image

    chip_limits = {
        "Castor": {"xlim": [10, 232], "ylim": [-5, 439]},
        "Pollux": {"xlim": [4, 224], "ylim": [8, 444]},
        "Helena": {"xlim": [6, 250], "ylim": [6, 336]},
        "Monsun": {"xlim": [2, 107], "ylim": [0, 170]},
    }

    # Optional global min/max for normalization. Set to None to compute per wafer.
    NORMALIZE_VMIN = None
    NORMALIZE_VMAX = None

    @udf(BinaryType())
    def rasterize_udf(wafermap, chip):
        if wafermap is None or chip not in chip_limits:
            return None

        limits = chip_limits[chip]
        xlim = limits["xlim"]
        ylim = limits["ylim"]

        coords = list(wafermap.keys())
        values = np.array(list(wafermap.values()), dtype=float)

        if len(coords) == 0:
            return None

        xs = []
        ys = []
        for c in coords:
            try:
                xs.append(float(c[0]))
                ys.append(float(c[1]))
            except Exception:
                xs.append(float(c["x"]))
                ys.append(float(c["y"]))

        xs = np.array(xs)
        ys = np.array(ys)

        median_val = np.median(values)
        centered = values - median_val

        vmin = NORMALIZE_VMIN
        vmax = NORMALIZE_VMAX
        if vmin is None:
            vmin = float(np.min(centered))
        if vmax is None:
            vmax = float(np.max(centered))
        if vmax == vmin:
            normalized = np.zeros_like(centered)
        else:
            normalized = (centered - vmin) / (vmax - vmin)

        x_grid, y_grid = np.meshgrid(
            np.arange(xlim[0], xlim[1] + 1),
            np.arange(ylim[0], ylim[1] + 1),
        )

        image_raw = griddata(
            (xs, ys),
            normalized,
            (x_grid, y_grid),
            method="linear",
            fill_value=0.0,
        )

        image_uint8 = (image_raw * 255.0).astype(np.uint8)
        image_pil = Image.fromarray(image_uint8, mode="L")
        image_resized = image_pil.resize((256, 256), resample=Image.BILINEAR)

        buffer = io.BytesIO()
        image_resized.save(buffer, format="PNG")
        return buffer.getvalue()

    source = spark.read.table("silver_wafermap_raw")  # noqa: F821

    return (
        source
        .select(
            F.col("WaferNr"),
            F.col("MeasurementSession"),
            F.col("MeasurementSeries"),
            F.col("MeasurementName"),
            F.col("Chip"),
            rasterize_udf(
                F.col("WafermapDict"),
                F.col("Chip"),
            ).alias("WafermapRasterPng"),
            F.col("MeasurementCount"),
            F.col("MeasurementSeriesStartTime"),
            F.current_timestamp().alias("GeneratedAt"),
            F.lit("silver").alias("layer"),
        )
    )
