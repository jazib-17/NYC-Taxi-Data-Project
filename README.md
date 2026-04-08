# NYC Taxi Data Project

An end-to-end data pipeline, analysis, and predictive modelling project using **PySpark**, **Python**, and **R** to process, explore, visualize, and model NYC Yellow Taxi trip data.

The project is split into two independent components:
- **PySpark Pipeline** — large-scale data processing and visualization across four months of raw trip data (August–November 2025)
- **Kaggle Competition** — predictive modelling to forecast NYC taxi fare amounts, evaluated on RMSE via a held-out Kaggle test set

---

## Project Structure
```
NYC-Taxi-Data-Project/
├── data/
│   ├── raw/                        # Original .parquet files
│   └── processed/                  # Cleaned .csv outputs
├── scripts/
│   ├── extract.py                  # PySpark pipeline: load, clean, explore, save
│   └── visualization.py            # Cross-month visualizations using Matplotlib & Seaborn
├── outputs/
│   └── charts/                     # Generated chart images
├── kaggle_competition/
│   ├── W26P1_Lasso.R               # Lasso regression (λ_1se)
│   ├── W26P1_Lassomin.R            # Lasso regression (λ_min)
│   ├── W26P1_GAM.R                 # Generalized Additive Model with cubic splines
│   ├── W26P1_Ranger.R              # Random Forest using the ranger package
│   ├── W26P1_RandomForests.R       # Random Forest using the randomForest package
│   ├── W26P1_XGBoost.R             # Gradient boosting using XGBoost
│   ├── W26P1_diagrams.R            # EDA plots (fare vs hour, dow, passenger count, etc.)
│   └── submissions/                # Kaggle submission .csv files
└── README.md
```

---

## PySpark Pipeline (`scripts/extract.py`)

The pipeline loads raw `.parquet` files using PySpark and performs the following steps:

- Computes `trip_duration_minutes` from pickup and dropoff timestamps
- Filters out invalid trips (zero distance, negative fares, unrealistic durations)
- Explores the cleaned data using PySpark aggregations, window functions, and SQL queries
- Saves a 30% sample of the cleaned data as a `.csv` for visualization

### PySpark Exploration Highlights

**Basic aggregations:** average trip distance by passenger count, fare and tip by hour of day, payment type distribution, and top 10 longest trips.

**Window functions:** ranks the top 3 busiest pickup hours within each day using `rank()` over a date partition. This identifies peak demand periods at a daily level rather than just an overall average.

**Multi-aggregation:** a single `groupBy` on payment type returning trip count, average fare, average tip, and average distance together.

**SQL interface:** registers the DataFrame as a temp view and queries it with `spark.sql()`, identifying the top 10 busiest hours across all trips.

**Bucketizer:** bins trips into distance categories (Short / Medium / Long / Very Long) using `pyspark.ml.feature.Bucketizer` and compares average fares and tips across buckets.

### Notes on Running the Pipeline

The full NYC taxi dataset for a single month contains several million rows, which creates two practical challenges when working locally on Windows.

- **Memory limitations:** calling `toPandas()` on the full cleaned DataFrame caused a `java.lang.OutOfMemoryError: Java heap space` crash, meaning the JVM ran out of memory trying to collect all rows at once. To address this, PySpark is configured with `spark.driver.memory = 4g` and the output is sampled at 30% using `df_clean.sample(fraction=0.3, seed=42)` before converting to Pandas. A 30% sample still represents millions of rows and is more than sufficient for meaningful analysis and visualization, while keeping memory usage manageable on a standard laptop.

- **GitHub file size limits:** even at 30%, the processed CSV files are large. GitHub enforces a 100MB limit per file, which the October output exceeded and had to be excluded from the repo. The other three months fall under the limit and are included. Any excluded file can be regenerated locally by running `extract.py` with the corresponding raw `.parquet` file in `data/raw/`.

- **Windows CSV output:** saving is done via `toPandas().to_csv()` rather than Spark's native `.write.csv()` to avoid the need for a Hadoop installation or `winutils.exe` setup on Windows. This produces a single clean CSV file rather than Spark's default folder of part files, which is also more convenient for sharing and loading downstream.

---

## Visualizations (`scripts/visualization.py`)

Loads cleaned `.csv` files for all four months and generates cross-month comparisons.

### Average Fare Trend
![Average Fare Trend](outputs/charts/fare_trend.png)

Average fares remain remarkably stable across all four months, ranging narrowly from $20.27 to $21.03. Fares peak in September at $21.00 before gradually declining through October and November. The consistency suggests NYC taxi pricing is not heavily influenced by season, at least over this four-month window.

### Total Trip Volume by Month
![Trip Volume](outputs/charts/trip_volume_by_month.png)

Trip volume peaks in October and is notably lower in August. One possible explanation is that August is a summer month where people may be more inclined to walk or bike, while the cooler fall months push more riders toward taxis. It is also worth noting that October's volume makes it the busiest month in this dataset despite fares being slightly lower than September.

### Average Trip Duration Heatmap
![Duration Heatmap](outputs/charts/duration_heatmap.png)

The longest average trip durations consistently appear around **hour 5** and again around **hours 15–16** across all four months. The early morning spike at hour 5 likely reflects airport runs or long cross-borough trips with very little traffic. The mid-afternoon spike around 3–4 PM aligns with the start of rush hour. The pattern is consistent month over month, suggesting it is structural rather than seasonal.

### Hourly Trip Demand
![Hourly Demand](outputs/charts/hourly_demand_by_month.png)

Demand follows a very predictable daily pattern across all four months. Trip volume bottoms out overnight between hours 1–6, then climbs steadily through the morning. Demand peaks sharply in the **late afternoon between hours 15–19**, coinciding with the end of the standard workday.

### Tip Percentage Distribution
![Tip Distribution](outputs/charts/tip_distribution_by_month.png)

Tip percentages are largely consistent across all four months. August shows a slightly narrower body in the violin shape, suggesting marginally less variability in tipping behaviour during the summer. A notable finding is the number of trips with tip percentages between 60–100%, which likely reflect short, cheap rides where even a small dollar tip becomes a high percentage of the fare. A violin plot was used here over a box plot because the volume of rows caused the outlier markers to overwhelm the chart.

---

## Kaggle Competition (`kaggle_competition/`)

This component focuses on predicting NYC taxi fare amounts using ~30,000 labelled training trips. Each trip includes pickup and dropoff coordinates, pickup datetime, and passenger count. Submissions are evaluated on **Root Mean Squared Error (RMSE)** on a held-out test set.

### Feature Engineering

Since the raw dataset contains limited features, a key focus was constructing meaningful predictors:

- **Haversine distance** — direct GPS distance in kilometres between pickup and dropoff; the strongest single predictor of fare amount
- **Hour of day / Day of week** — extracted from pickup datetime to capture rush hour and weekend effects
- **Direction (bearing)** — the bearing from pickup to dropoff, capturing route directionality which affects tolls and routing
- **Airport distances** — straight-line distances to JFK and LGA airports, since airport trips carry flat fares and follow predictable patterns
- **Distance from Midtown Manhattan** — trips originating further from the city centre tend to be longer and pricier
- **Cyclical time encoding** (`sin_time`, `cos_time`) — ensures the model recognizes that hour 23 and hour 0 are adjacent rather than numerically distant

### Exploratory Data Analysis

Key findings from the EDA (`W26P1_diagrams.R`):

- Most fares fall between $0–$20 and most trips do not exceed 25 km, consistent with short urban rides
- The busiest pickup location (rounded coordinates) is **longitude -73.982, latitude 40.756** — central Manhattan, as expected
- **Sunday** has the highest average trip fare (~$12.50); **Friday and Saturday** see the highest trip volumes
- **5 AM** has the highest average fare (~$17), likely reflecting long-distance airport runs with minimal traffic
- Passenger count has no meaningful linear relationship with fare, and variable importance plots consistently rank it last across all models

### Models

Four fundamentally different approaches were explored and compared, all tuned using **10-fold cross-validation** on the training set.

---

#### 1. Lasso Regression (`W26P1_Lasso.R`, `W26P1_Lassomin.R`)

A linear baseline using the `glmnet` package in R. Two standard choices of regularization strength were compared:

- **λ_1se** — the largest λ within one standard error of the CV minimum; produces a simpler, more regularized model. With this setting, distance was the only feature with a meaningful coefficient, reflecting that taxi fares are primarily distance-driven. **Kaggle RMSE: 4.83**
- **λ_min** — the λ minimizing CV error; retains more features and captures extra variation from pickup/dropoff coordinates and day of week. **Kaggle RMSE: 4.65**

The trade-off is clear: λ_1se is more interpretable but less accurate; λ_min extracts additional signal from location at the cost of a more complex model.

---

#### 2. Generalized Additive Model — GAM (`W26P1_GAM.R`)

After establishing a linear baseline, a GAM with **cubic regression splines** was fitted using the `mgcv` package to allow non-linear relationships between each predictor and fare amount. Smoothing splines were applied to distance, hour, and pickup/dropoff coordinates; day of week and passenger count were kept as linear terms.

All smooth terms were highly significant (p < 2e-16) with estimated degrees of freedom well above 1, confirming that the linear structure in Lasso was too restrictive. Passenger count remained non-significant, consistent with the Lasso results.

GAM improves on Lasso by capturing non-linear effects per feature, but still assumes an additive structure with no interactions between variables. **Kaggle RMSE: 4.23**

---

#### 3. Random Forest — Ranger (`W26P1_Ranger.R`, `W26P1_RandomForests.R`)

Random Forests build an ensemble of decision trees, each trained on a bootstrap sample of the data with a random subset of features considered at each split (`mtry`). Predictions are made by averaging across all trees, reducing variance relative to a single tree.

The initial implementation used the `randomForest` package, but at 30,000 rows and 500 trees, runtimes were prohibitive (~30 min per run). The implementation was switched to **`ranger`**, which supports native parallelism and runs significantly faster while producing equivalent results. The `mtry` parameter was tuned via 10-fold cross-validation using `caret` over a grid of 1–8; **mtry = 3** was selected as optimal on the original feature set, giving a **Kaggle RMSE of 3.64**.

After adding the engineered features described above, the expanded model was re-tuned and selected **mtry = 5**, achieving the best overall result of **Kaggle RMSE: 3.53**.

Variable importance confirmed the feature engineering was meaningful — distance dominated at 100%, followed by coordinates and the new airport distance features. Cyclical time encoding, hour, day of week, and passenger count contributed the least.

<!-- Optional: add variable importance plot here -->
![Ranger Variable Importance](kaggle_competition/RangerVarImportance.png)

---

#### 4. Gradient Boosting — GBM & XGBoost (`W26P1_XGBoost.R`)

Boosting builds trees **sequentially**, with each new tree correcting the residuals of the previous ones, unlike Random Forest which builds trees independently in parallel.

**GBM** (`gbm` package via `caret`) was tuned over a grid of interaction depths (5, 6, 7), number of trees (200–1000), and shrinkage values (0.01, 0.05, 0.10). Optimal parameters were `interaction.depth = 5`, `shrinkage = 0.05`, `n.trees = 800`. Using only the original feature set, this produced a **Kaggle RMSE of 3.66** — marginally worse than Random Forest's 3.64 at that stage, both models likely hitting the same feature ceiling.

**XGBoost** was then adopted for its speed and flexibility, and run with the full expanded feature set. Key tuning decisions:
- `eta` reduced from 0.1 to **0.05** to make more conservative corrections and reduce overfitting with the richer feature set
- `max_depth` increased from 6 to **7** to capture additional feature interactions
- Optimal tree count determined via `xgb.cv` with 10-fold CV and early stopping: **206 trees**

XGBoost variable importance mirrored the Random Forest results — distance dominant, followed by dropoff longitude and direction, with airport distances contributing meaningfully. **Kaggle RMSE: 3.56**

<!-- Optional: add XGBoost variable importance plot here -->
![XGBoost Variable Importance](kaggle_competition/XGvarImportance.png)

---

### Final Results

| Model | Kaggle RMSE |
|---|---|
| Lasso (λ_min) | 4.65031 |
| GAM with Cubic Splines | 4.23407 |
| Gradient Boosting (XGBoost) | 3.56189 |
| **Random Forest (Ranger)** | **3.53813** |

Random Forest with the expanded feature set achieved the best score. The progression from Lasso → GAM → tree-based methods illustrates the value of both model flexibility and feature engineering — each step addressed a limitation of the previous approach.

---

## Requirements

### Python (PySpark pipeline)
```
pyspark
pandas
matplotlib
seaborn
```
```bash
pip install pyspark pandas matplotlib seaborn
```

### R (Kaggle modelling)
```
glmnet
mgcv
randomForest
ranger
caret
doParallel
gbm
xgboost
```

---

## Windows Notes

- Saving output uses `toPandas().to_csv()` to avoid the need for Hadoop or `winutils.exe`
- If using PySpark on Windows and Hadoop errors appear, either install winutils or use the `toPandas()` approach as done here
- For datasets over 1GB, ensure `spark.driver.memory` is set to at least `4g`

## Data & Output Files

The processed `.csv` files for August, September, and November 2025 are included in this repository. The October 2025 output (`clean_taxi_data_2025-10.csv`) exceeds GitHub's 100MB file size limit and is excluded from the repo, but can be regenerated locally by running `extract.py` with the corresponding raw `.parquet` file placed in `data/raw/`.
