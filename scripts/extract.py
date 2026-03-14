import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("NYC Taxi Data Pipeline") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / "data" / "raw" / "yellow_tripdata_2025-11.parquet"
file_path_out = base_dir / "data" / "processed" / "clean_taxi_data_2025-11.csv"

df = spark.read.parquet(str(file_path))

df = df.withColumn(
    "trip_duration_minutes",
    round((unix_timestamp(col("tpep_dropoff_datetime")) -
     unix_timestamp(col("tpep_pickup_datetime"))) / 60,2)
)

# Filter invalid trips
df_clean = df.filter(
    (col("trip_distance") > 0) &
    (col("passenger_count") > 0) &
    (col("trip_duration_minutes") > 0) &
    (col("trip_duration_minutes") < 180) &
    (col("fare_amount") > 0) &
    (col("tip_amount") >= 0)
)

# Exploration

# Trip distance breakdown
df_clean.groupBy("passenger_count") \
    .agg(round(avg("trip_distance"), 2).alias("avg_distance")) \
    .orderBy("passenger_count") \
    .show()

# Average fare and tip by hour of day
df_clean.withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
    .groupBy("pickup_hour") \
    .agg(
        round(avg("fare_amount"), 2).alias("avg_fare"),
        round(avg("tip_amount"), 2).alias("avg_tip")
    ) \
    .orderBy("pickup_hour") \
    .show(24)

# Payment type distribution
df_clean.groupBy("payment_type") \
    .count() \
    .orderBy(desc("count")) \
    .show()

# Top 10 longest trips by distance
df_clean.orderBy(desc("trip_distance")) \
    .select("trip_distance", "fare_amount", "trip_duration_minutes") \
    .show(10)

# Window function — rank busiest hours within each day
df_clean.withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
    .withColumn("pickup_date", col("tpep_pickup_datetime").cast("date")) \
    .groupBy("pickup_date", "pickup_hour") \
    .count() \
    .withColumn("rank", rank().over(Window.partitionBy("pickup_date").orderBy(desc("count")))) \
    .filter(col("rank") <= 3) \
    .orderBy("pickup_date", "rank") \
    .show(20)

# Multi-aggregation in one call
df_clean.groupBy("payment_type") \
    .agg(
        count("*").alias("trip_count"),
        round(avg("fare_amount"), 2).alias("avg_fare"),
        round(avg("tip_amount"), 2).alias("avg_tip"),
        round(avg("trip_distance"), 2).alias("avg_distance")
    ) \
    .orderBy(desc("trip_count")) \
    .show()

# SQL interface
df_clean.createOrReplaceTempView("trips")
spark.sql("""
    SELECT
        HOUR(tpep_pickup_datetime) AS pickup_hour,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS avg_fare
    FROM trips
    WHERE fare_amount > 0
    GROUP BY pickup_hour
    ORDER BY trip_count DESC
    LIMIT 10
""").show()

# Bin trips into distance buckets
from pyspark.ml.feature import Bucketizer

bucketizer = Bucketizer(
    splits=[0, 1, 5, 10, float("inf")],
    inputCol="trip_distance",
    outputCol="distance_bucket"
)
bucket_labels = {0.0: "Short (<1mi)", 1.0: "Medium (1-5mi)", 2.0: "Long (5-10mi)", 3.0: "Very Long (10mi+)"}
df_bucketed = bucketizer.transform(df_clean)
df_bucketed.groupBy("distance_bucket") \
    .agg(
        count("*").alias("trip_count"),
        round(avg("fare_amount"), 2).alias("avg_fare"),
        round(avg("tip_amount"), 2).alias("avg_tip")
    ) \
    .orderBy("distance_bucket") \
    .show()

# Save the filtered data
df_clean.sample(fraction=0.3, seed=42) \
    .toPandas() \
    .to_csv(str(file_path_out), index=False)

print("Rows:", df.count())
print("Columns:", len(df.columns))
df.select(round(avg("total_amount"),2)).show()

