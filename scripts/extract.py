import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("NYC Taxi Data Pipeline").getOrCreate()

base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / "data" / "raw" / "yellow_tripdata_2025-08.parquet"
file_path_out = base_dir / "data" / "processed" / "clean_taxi_data_2025-08"

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

df_clean.write.mode("overwrite").option("header", True).csv(str(file_path_out))

df.show(5)

print("Rows:", df.count())
print("Columns:", len(df.columns))
df.select(round(avg("total_amount"),2)).show()

