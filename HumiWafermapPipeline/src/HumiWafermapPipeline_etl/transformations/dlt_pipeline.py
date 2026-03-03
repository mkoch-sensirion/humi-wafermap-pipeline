from itertools import chain

from pyspark import pipelines as dp
import pyspark.sql.functions as F


# ============================================================================
# SILVER LAYER: Read and aggregate measurements into wafermap dicts
# ============================================================================

@dp.table(
    name="silver_wafermap_raw",
    description=(
        "Aggregated wafermap measurements by session + L3 series, with XY "
        "coordinates as dict keys"
    ),
    comment=(
        "One row per substrate per MeasurementSession/SeriesSuffix with "
        "wafermap: {(X,Y): meas_value}"
    ),
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
                >= F.lit("2026-02-01")
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
            F.first("Serial").alias("serial"),
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
            F.first("MeasurementSeriesStartTime")
        )
    )

    return (
        wafermap_data
        .select(
            F.col("WaferNr"),
            F.col("Serial"),
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


# # ============================================================================
# # BRONZE LAYER: Wafermap normalization and feature extraction
# # ============================================================================

# @dp.materialized_view(
#     name="bronze_wafermap_normalized",
#     description=(
#         "Normalized wafermap data with standardized values and metadata"
#     ),
# )
# def bronze_wafermap_normalized():
#     """
#     Process and normalize wafermap data from silver layer.

#     - Normalize measurement values to 0-1 range per substrate
#     - Extract statistical features
#     - Prepare for image processing or feature extraction
#     """
#     source = dp.read_table("silver_wafermap_raw")

#     values = F.map_values(F.col("wafermap"))
#     min_val = F.array_min(values)
#     max_val = F.array_max(values)
#     norm_vals = F.transform(
#         values,
#         lambda v: F.when(
#             max_val != min_val,
#             (v - min_val) / (max_val - min_val)
#         ).otherwise(F.lit(0.0))
#     )

#     return (
#         source
#         .select(
#             F.col("substrate_id"),
#             F.col("serial"),
#             F.col("measurement_session"),
#             F.col("series_suffix"),
#             F.col("measurement_name"),
#             F.col("wafermap"),
#             F.map_from_arrays(
#                 F.map_keys(F.col("wafermap")),
#                 norm_vals,
#             ).alias("wafermap_normalized"),
#             F.col("measurement_count"),
#             F.col("latest_measurement_time"),
#             F.when(
#                 F.size(values) > 0,
#                 F.aggregate(
#                     values,
#                     F.lit(0.0),
#                     lambda acc, x: acc + x
#                 ) / F.size(values)
#             ).otherwise(F.lit(0.0)).alias("mean_value"),
#             min_val.alias("min_value"),
#             max_val.alias("max_value"),
#             F.current_timestamp().alias("normalized_at"),
#             F.lit("bronze").alias("layer")
#         )
#     )

