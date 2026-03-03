from pyspark import pipelines as dp
import pyspark.sql.functions as F


# ============================================================================
# GOLD LAYER: Predictions based on wafermap analysis
# ============================================================================

@dp.materialized_view(
    name="gold_wafermap_predictions",
    description=(
        "Prediction layer with wafer quality scores and defect classification"
    ),
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
                (F.col("max_value") - F.col("min_value")) < 10,
                "PASS"
            ).otherwise("FAIL").alias("wafer_status"),
            F.round(
                F.when(
                    F.col("max_value") > 0,
                    F.col("min_value") / F.col("max_value") * 100
                )
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
        .groupBy(
            F.date(F.col("prediction_timestamp")).alias("prediction_date")
        )
        .agg(
            F.count("substrate_id").alias("total_substrates"),
            F.sum(
                F.when(F.col("wafer_status") == "PASS", 1).otherwise(0)
            ).alias("passed"),
            F.sum(
                F.when(F.col("wafer_status") == "FAIL", 1).otherwise(0)
            ).alias("failed"),
            F.round(
                F.avg(F.col("quality_score")),
                2,
            ).alias("avg_quality_score"),
            F.round(
                F.avg(F.col("mean_value")),
                2,
            ).alias("avg_measurement_value"),
            F.round(
                F.avg(F.col("measurement_count")),
                0,
            ).alias("avg_points_per_wafer"),
        )
        .orderBy(F.col("prediction_date").desc())
    )