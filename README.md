# NYC Taxi Data Project

An end-to-end data pipeline and analysis project using **PySpark** and **Python** to process, explore, and visualize NYC Yellow Taxi trip data from August to November 2025.

---

## Project Structure
```
NYC-Taxi-Data-Project/
├── data/
│   ├── raw/                  # Original .parquet files
│   └── processed/            # Cleaned .csv outputs
├── scripts/
│   ├── extract.py            # PySpark pipeline: load, clean, explore, save
│   └── visualization.py      # Cross-month visualizations using Matplotlib & Seaborn
├── outputs/
│   └── charts/               # Generated chart images
└── README.md
```

---

## Pipeline Overview (`extract.py`)

The pipeline loads raw `.parquet` files using PySpark and performs the following steps:

- Computes `trip_duration_minutes` from pickup and dropoff timestamps
- Filters out invalid trips (zero distance, negative fares, unrealistic durations)
- Explores the cleaned data using PySpark aggregations, window functions, and SQL queries
- Saves a 30% sample of the cleaned data as a `.csv` for visualization

### PySpark Exploration Highlights

**Basic aggregations** — average trip distance by passenger count, fare and tip by hour of day, payment type distribution, and top 10 longest trips.

**Window functions** — ranks the top 3 busiest pickup hours within each day using `rank()` over a date partition. This identifies peak demand periods at a daily level rather than just an overall average.

**Multi-aggregation** — a single `groupBy` on payment type returning trip count, average fare, average tip, and average distance together.

**SQL interface** — registers the DataFrame as a temp view and queries it with `spark.sql()`, identifying the top 10 busiest hours across all trips.

**Bucketizer** — bins trips into distance categories (Short / Medium / Long / Very Long) using `pyspark.ml.feature.Bucketizer` and compares average fares and tips across buckets.

### Notes on Running the Pipeline

- PySpark is configured with `spark.driver.memory = 4g` to handle large datasets on a local machine
- The full dataset exceeds Excel's row limit (~1M rows) — this is expected and intentional
- A 30% sample is saved for visualization; the full pipeline processes all rows
- On Windows, the output is saved via `toPandas().to_csv()` to avoid Hadoop/winutils requirements

---

## Visualizations (`visualization.py`)

Loads cleaned `.csv` files for all four months and generates cross-month comparisons.

### Average Fare Trend
![Average Fare Trend](outputs/charts/fare_trend.png)

A line chart with annotated values showing how average fare amounts shift across the four months. The y-axis is scaled tightly to the actual data range to make small fluctuations visible.

### Total Trip Volume by Month
![Trip Volume](outputs/charts/trip_volume_by_month.png)

Bar chart showing total trip counts per month. Useful for identifying seasonal demand shifts heading into fall and winter.

### Average Trip Duration Heatmap
![Duration Heatmap](outputs/charts/duration_heatmap.png)

A heatmap of average trip duration broken down by hour of day and month. Highlights how rush hour patterns and trip lengths shift across months — longer trips tend to cluster around morning and evening commute windows.

### Hourly Trip Demand
![Hourly Demand](outputs/charts/hourly_demand_by_month.png)

Overlapping line chart showing trip volume by hour for each month. The morning and evening rush hours are clearly visible, and the lines show how demand patterns remain consistent month over month with minor variations.

### Tip Percentage Distribution
![Tip Distribution](outputs/charts/tip_distribution_by_month.png)

Violin plot showing the distribution of tip amounts as a percentage of fare across months. The `inner="quartile"` setting shows quartile lines inside each violin, combining the readability of a box plot with the distributional detail of a density plot. Chosen over a box plot because the large dataset (millions of rows) makes whisker-based plots less informative.

---

## Requirements
```
pyspark
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install pyspark pandas matplotlib seaborn
```

---

## Windows Notes

- Saving output uses `toPandas().to_csv()` to avoid the need for Hadoop or `winutils.exe`
- If using PySpark on Windows and Hadoop errors appear, either install winutils or use the `toPandas()` approach as done here
- For datasets over 1GB, ensure `spark.driver.memory` is set to at least `4g`

## Data & Output Files

The processed `.csv` files for August, September, and November 2025 are included in this repository. The October 2025 output (`clean_taxi_data_2025-10.csv`) exceeds GitHub's 100MB file size limit and is excluded from the repo, but can be regenerated locally by running `extract.py` with the corresponding raw `.parquet` file placed in `data/raw/`.
