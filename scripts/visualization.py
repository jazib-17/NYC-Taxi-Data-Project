import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# Paths
base_dir = Path(__file__).resolve().parent.parent
processed_dir = base_dir / "data" / "processed"
output_dir = base_dir / "outputs" / "charts"
output_dir.mkdir(parents=True, exist_ok=True)

MONTHS = {
    "Aug 2025": "clean_taxi_data_2025-08.csv",
    "Sep 2025": "clean_taxi_data_2025-09.csv",
    "Oct 2025": "clean_taxi_data_2025-10.csv",
    "Nov 2025": "clean_taxi_data_2025-11.csv",
}

# Load & tag each month
dfs = []
for label, filename in MONTHS.items():
    path = processed_dir / filename
    df = pd.read_csv(path, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df["month"] = label
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all["pickup_hour"] = df_all["tpep_pickup_datetime"].dt.hour

print(f"Total rows loaded: {len(df_all):,}")

#Style
COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
MONTH_ORDER = list(MONTHS.keys())
sns.set_theme(style="darkgrid", font_scale=1.1)

# 1. Average Fare Trend with Spread
fig, ax = plt.subplots(figsize=(8, 5))
avg_fare = df_all.groupby("month")["fare_amount"].mean().reindex(MONTH_ORDER)
ax.plot(MONTH_ORDER, avg_fare, color=COLORS[0], linewidth=2.5, marker="o", markersize=8)
for i, (month, val) in enumerate(avg_fare.items()):
    ax.annotate(f"${val:.2f}", xy=(i, val), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=10)
ax.set_title("Average Fare by Month", fontsize=14, fontweight="bold")
ax.set_ylabel("Fare Amount ($)")
ax.set_ylim(avg_fare.min() * 0.95, avg_fare.max() * 1.1)
plt.tight_layout()
plt.savefig(output_dir / "fare_trend.png", dpi=150)
plt.close()
print("Saved: fare_trend.png")

# 2. Trip Volume by Month (bar chart)
fig, ax = plt.subplots(figsize=(8, 5))
trip_counts = df_all.groupby("month").size().reindex(MONTH_ORDER)
bars = ax.bar(MONTH_ORDER, trip_counts, color=COLORS, edgecolor="white", linewidth=0.8)
trip_labels = [f"{v:,.0f}" for v in trip_counts]
ax.bar_label(bars, labels=trip_labels, padding=4, fontsize=10)
ax.set_title("Total Trip Volume by Month", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_ylim(0, trip_counts.max() * 1.2)
plt.tight_layout()
plt.savefig(output_dir / "trip_volume_by_month.png", dpi=150)
plt.close()
print("Saved: trip_volume_by_month.png")

# 3. Avg Trip Duration Heatmap by Hour × Month
pivot = df_all.groupby(["month", "pickup_hour"])["trip_duration_minutes"] \
              .mean().unstack().reindex(MONTH_ORDER)
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
            annot=False, fmt=".1f", cbar_kws={"label": "Avg Duration (min)"})
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_title("Average Trip Duration by Hour and Month", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(output_dir / "duration_heatmap.png", dpi=150)
plt.close()
print("Saved: duration_heatmap.png")

# 4. Hourly Trip Demand — all months overlaid (line chart)
fig, ax = plt.subplots(figsize=(10, 5))
for i, month in enumerate(MONTH_ORDER):
    hourly = df_all[df_all["month"] == month].groupby("pickup_hour").size()
    ax.plot(hourly.index, hourly.values, label=month, color=COLORS[i],
            linewidth=2.5, marker="o", markersize=4)
ax.set_title("Hourly Trip Demand by Month", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Trips")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / "hourly_demand_by_month.png", dpi=150)
plt.close()
print("Saved: hourly_demand_by_month.png")

# 5. Tip % Distribution by Month (violin plot)
df_all["tip_pct"] = (df_all["tip_amount"] / df_all["fare_amount"] * 100).clip(0, 100)
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=df_all, x="month", y="tip_pct", order=MONTH_ORDER,
               palette=COLORS, ax=ax, inner="quartile")
ax.set_title("Tip % Distribution by Month", fontsize=14, fontweight="bold")
ax.set_ylabel("Tip as % of Fare")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(output_dir / "tip_distribution_by_month.png", dpi=150)
plt.close()
print("Saved: tip_distribution_by_month.png")