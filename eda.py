"""
car-health-ai - exploratory data analysis
==========================================
Run after generate_data.py. Produces summary stats + saved plots in
eda_output/ covering:
  1. Feature distributions (age, mileage, sensor readings)
  2. Correlation between numeric features and failure labels
  3. Per-component failure rates
  4. Failure rate by driving conditions / climate (categorical breakdown)
  5. Key feature vs failure boxplots (e.g. brake pad depth vs brake failure)

Usage:
    python eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/car_health_ai_dataset.csv"
OUT_DIR = "eda_output"
COMPONENTS = ["engine", "brakes", "battery", "transmission", "suspension", "tires"]

os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\nMissing values:\n{missing if len(missing) else 'None'}")
    return df


def basic_summary(df):
    print("\n" + "=" * 60)
    print("BASIC SUMMARY")
    print("=" * 60)
    numeric_cols = ["age_years", "mileage_km", "avg_daily_km", "service_gap_months",
                     "oil_quality_index", "engine_temp_c", "brake_pad_depth_mm",
                     "battery_voltage", "transmission_fluid_quality",
                     "vibration_score", "tire_tread_depth_mm"]
    print(df[numeric_cols].describe().T[["mean", "std", "min", "max"]].round(2))

    print("\nCategorical breakdown:")
    print(df["driving_conditions"].value_counts(normalize=True).round(3))
    print(df["climate"].value_counts(normalize=True).round(3))


def plot_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fields = ["age_years", "mileage_km", "service_gap_months", "oil_quality_index",
              "engine_temp_c", "brake_pad_depth_mm", "battery_voltage",
              "transmission_fluid_quality", "tire_tread_depth_mm"]
    for ax, field in zip(axes.flat, fields):
        sns.histplot(df[field], kde=True, ax=ax, color="#FF6B35")
        ax.set_title(field)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_feature_distributions.png")
    plt.close()
    print(f"Saved {OUT_DIR}/01_feature_distributions.png")


def plot_failure_rates(df):
    rates = [df[f"{c}_fail_within_10k"].mean() * 100 for c in COMPONENTS]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(COMPONENTS, rates, color="#FF6B35")
    ax.set_ylabel("% of cars flagged (fail within 10,000 km)")
    ax.set_title("Component Failure Flag Rate")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.5, f"{rate:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_component_failure_rates.png")
    plt.close()
    print(f"Saved {OUT_DIR}/02_component_failure_rates.png")


def plot_correlation_heatmap(df):
    numeric_cols = ["age_years", "mileage_km", "avg_daily_km", "service_gap_months",
                     "missed_services", "oil_quality_index", "engine_temp_c",
                     "brake_pad_depth_mm", "battery_voltage",
                     "transmission_fluid_quality", "vibration_score",
                     "tire_tread_depth_mm"] + [f"{c}_fail_within_10k" for c in COMPONENTS]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, ax=ax, linewidths=0.4)
    ax.set_title("Feature Correlation Matrix (incl. failure labels)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_correlation_heatmap.png")
    plt.close()
    print(f"Saved {OUT_DIR}/03_correlation_heatmap.png")


def plot_key_relationships(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sns.boxplot(x="brakes_fail_within_10k", y="brake_pad_depth_mm", hue="brakes_fail_within_10k", data=df, ax=axes[0, 0], palette=["#3FB68B", "#FF6B35"], legend=False)
    axes[0, 0].set_title("Brake Pad Depth vs Brake Failure")

    sns.boxplot(x="battery_fail_within_10k", y="battery_voltage", hue="battery_fail_within_10k", data=df, ax=axes[0, 1], palette=["#3FB68B", "#FF6B35"], legend=False)
    axes[0, 1].set_title("Battery Voltage vs Battery Failure")

    sns.boxplot(x="engine_fail_within_10k", y="oil_quality_index", hue="engine_fail_within_10k", data=df, ax=axes[0, 2], palette=["#3FB68B", "#FF6B35"], legend=False)
    axes[0, 2].set_title("Oil Quality vs Engine Failure")

    sns.boxplot(x="transmission_fail_within_10k", y="transmission_fluid_quality", hue="transmission_fail_within_10k", data=df, ax=axes[1, 0], palette=["#3FB68B", "#FF6B35"], legend=False)
    axes[1, 0].set_title("Transmission Fluid Quality vs Failure")

    sns.boxplot(x="suspension_fail_within_10k", y="vibration_score", hue="suspension_fail_within_10k", data=df, ax=axes[1, 1], palette=["#3FB68B", "#FF6B35"], legend=False)
    axes[1, 1].set_title("Vibration Score vs Suspension Failure")

    sns.boxplot(x="tires_fail_within_10k", y="tire_tread_depth_mm", hue="tires_fail_within_10k", data=df, ax=axes[1, 2], palette=["#3FB68B", "#FF6B35"], legend=False)
    axes[1, 2].set_title("Tire Tread Depth vs Tire Failure")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_key_feature_relationships.png")
    plt.close()
    print(f"Saved {OUT_DIR}/04_key_feature_relationships.png")


def plot_categorical_breakdown(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    driving_rates = df.groupby("driving_conditions")[[f"{c}_fail_within_10k" for c in COMPONENTS]].mean().mean(axis=1) * 100
    driving_rates.plot(kind="bar", ax=axes[0], color="#FF6B35")
    axes[0].set_title("Avg Failure Rate by Driving Conditions")
    axes[0].set_ylabel("% flagged (avg across components)")
    axes[0].tick_params(axis="x", rotation=0)

    climate_rates = df.groupby("climate")[[f"{c}_fail_within_10k" for c in COMPONENTS]].mean().mean(axis=1) * 100
    climate_rates.plot(kind="bar", ax=axes[1], color="#3FB68B")
    axes[1].set_title("Avg Failure Rate by Climate")
    axes[1].set_ylabel("% flagged (avg across components)")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_categorical_breakdown.png")
    plt.close()
    print(f"Saved {OUT_DIR}/05_categorical_breakdown.png")


if __name__ == "__main__":
    df = load_data()
    basic_summary(df)
    plot_distributions(df)
    plot_failure_rates(df)
    plot_correlation_heatmap(df)
    plot_key_relationships(df)
    plot_categorical_breakdown(df)
    print(f"\nAll plots saved to {OUT_DIR}/")
