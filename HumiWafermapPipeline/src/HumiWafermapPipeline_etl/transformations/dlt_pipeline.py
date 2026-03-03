from itertools import chain

from pyspark import pipelines as dp
import pyspark.sql.functions as F


# ============================================================================
# SILVER LAYER: Read and aggregate measurements into wafermap dicts
# ============================================================================

@dp.table(
    name="silver_wafermap_raw",
    description="Aggregated wafermap measurements by session + L3 series, with XY coordinates as dict keys",
    comment="One row per substrate per MeasurementSession/SeriesSuffix with wafermap: {(X,Y): meas_value}"
)
def silver_wafermap_raw():
    """
    Read measurement data from ops_prod.dk_silver_data.t_data_measurement incrementally
    and aggregate by substrate and measurement series (L3).

    Filters only Castor/Pollux, Process=Prober, SeriesSuffix=L3, MeasurementName=CSR_40_01_01.
    Update column names if they differ in your table.
    """

    # Product -> MeasurementName mapping (extend this dict as new products are added)
    product_measurement_map = {
        "Castor": "CSR_40_01_01",
        "Pollux": "PLX_40_01_01",
    }

    measurement_name_expr = F.create_map(
        *chain.from_iterable((F.lit(k), F.lit(v)) for k, v in product_measurement_map.items())
    )

    raw_data = (
        spark.readStream.table("ops_prod.dk_silver_data.t_data_measurement")
        .filter(F.col("ValueReal").isNotNull())
        .filter(F.col("Product").isin(list(product_measurement_map.keys())))
        .filter(F.col("Process") == F.lit("Prober"))
        .filter(F.col("SeriesSuffix") == F.lit("L3"))
        .withColumn("expected_measurement_name", measurement_name_expr[F.col("Product")])
        .filter(F.col("MeasurementName") == F.col("expected_measurement_name"))
        .select(
            F.col("InputSubstrateName").alias("WaferNr"),
            F.col("SerialNumber").alias("Serial"),
            F.col("MeasurementSession").alias("MeasurementSession"),
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
            "substrate_id",
            "measurement_session",
            "series_suffix",
            "measurement_name",
            "product"
        )
        .agg(
            F.first("serial").alias("serial"),
            F.map_from_entries(
                F.collect_list(
                    F.struct(
                        F.struct(F.col("x_on_substrate"), F.col("y_on_substrate")).alias("coord"),
                        F.col("meas_value").alias("value")
                    )
                )
            ).alias("wafermap"),
            F.count("meas_value").alias("measurement_count"),
            F.max("measurement_ts").alias("latest_measurement_time")
        )
    )

    return (
        wafermap_data
        .select(
            F.col("substrate_id"),
            F.col("serial"),
            F.col("measurement_session"),
            F.col("series_suffix"),
            F.col("measurement_name"),
            F.col("product"),
            F.col("wafermap"),
            F.col("measurement_count"),
            F.col("latest_measurement_time"),
            F.current_timestamp().alias("processed_at"),
            F.lit("silver").alias("layer")
        )
    )


# ============================================================================
# BRONZE LAYER: Wafermap normalization and feature extraction
# ============================================================================

@dp.materialized_view(
    name="bronze_wafermap_normalized",
    description="Normalized wafermap data with standardized values and metadata",
)
def bronze_wafermap_normalized():
    """
    Process and normalize wafermap data from silver layer.
    
    - Normalize measurement values to 0-1 range per substrate
    - Extract statistical features
    - Prepare for image processing or feature extraction
    """
    source = dp.read_table("silver_wafermap_raw")

    values = F.map_values(F.col("wafermap"))
    min_val = F.array_min(values)
    max_val = F.array_max(values)
    norm_vals = F.transform(
        values,
        lambda v: F.when(max_val != min_val, (v - min_val) / (max_val - min_val)).otherwise(F.lit(0.0))
    )

    return (
        source
        .select(
            F.col("substrate_id"),
            F.col("serial"),
            F.col("measurement_session"),
            F.col("series_suffix"),
            F.col("measurement_name"),
            F.col("wafermap"),
            F.map_from_arrays(F.map_keys(F.col("wafermap")), norm_vals).alias("wafermap_normalized"),
            F.col("measurement_count"),
            F.col("latest_measurement_time"),
            F.when(F.size(values) > 0, F.aggregate(values, F.lit(0.0), lambda acc, x: acc + x) / F.size(values))
            .otherwise(F.lit(0.0)).alias("mean_value"),
            min_val.alias("min_value"),
            max_val.alias("max_value"),
            F.current_timestamp().alias("normalized_at"),
            F.lit("bronze").alias("layer")
        )
    )


# ============================================================================
# GOLD LAYER: Predictions based on wafermap analysis
# ============================================================================

@dp.materialized_view(
    name="gold_wafermap_predictions",
    description="Prediction layer with wafer quality scores and defect classification",
)
def gold_wafermap_predictions():
    """
    Generate predictions from normalized wafermap data.
    
    For now: hardcoded prediction logic based on measurement statistics.
    TODO: Replace with actual ML model inference on wafermap patterns
    """
    source = dp.read_table("bronze_wafermap_normalized")
    
    # HARDCODED PREDICTIONS - Replace with actual model inference
    return (
        source
        .select(
            F.col("substrate_id"),
            F.col("serial"),
            F.col("measurement_session"),
            F.col("series_suffix"),
            F.col("measurement_name"),
            F.col("wafermap"),
            F.col("wafermap_normalized"),
            # Placeholder predictions based on measurement statistics
            F.when(
                (F.col("max_value") - F.col("min_value")) < 10,  # Low variance = good?
                "PASS"
            ).otherwise("FAIL").alias("wafer_status"),
            F.round(
                F.when(F.col("max_value") > 0, F.col("min_value") / F.col("max_value") * 100)
                .otherwise(0.0),
                2
            ).alias("quality_score"),  # Simple metric: min/max ratio
            F.col("mean_value"),
            F.col("min_value"),
            F.col("max_value"),
            F.col("measurement_count"),
            F.lit("PLACEHOLDER_MODEL_V1").alias("model_version"),
            F.current_timestamp().alias("prediction_timestamp"),
            F.lit("gold").alias("layer")
        )
    )


# ============================================================================
# GOLD LAYER: Summary statistics for monitoring
# ============================================================================

@dp.materialized_view(
    name="gold_wafermap_summary",
    description="Summary statistics of wafer predictions by date",
)
def gold_wafermap_summary():
    """
    Create summary statistics for monitoring and dashboards.
    """
    predictions = dp.read_table("gold_wafermap_predictions")
    
    return (
        predictions
        .groupBy(F.date(F.col("prediction_timestamp")).alias("prediction_date"))
        .agg(
            F.count("substrate_id").alias("total_substrates"),
            F.sum(F.when(F.col("wafer_status") == "PASS", 1).otherwise(0)).alias("passed"),
            F.sum(F.when(F.col("wafer_status") == "FAIL", 1).otherwise(0)).alias("failed"),
            F.round(F.avg(F.col("quality_score")), 2).alias("avg_quality_score"),
            F.round(F.avg(F.col("mean_value")), 2).alias("avg_measurement_value"),
            F.round(F.avg(F.col("measurement_count")), 0).alias("avg_points_per_wafer")
        )
        .orderBy(F.col("prediction_date").desc())
    )
