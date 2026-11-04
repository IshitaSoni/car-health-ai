"""
carsentinel - synthetic dataset generator
==========================================
Generates a realistic used-car inspection dataset with per-component
failure-risk labels, based on commonly documented degradation patterns used in
standard pre-purchase inspection checklists rather than arbitrary randomness.

Why synthetic: no public dataset pairs (age, mileage, service history,
sensor readings) with *actual* component failure outcomes -- real fleet
telemetry of this kind is proprietary. Instead we encode known wear
relationships (e.g. brake wear scales with mileage AND urban/stop-go
driving; battery degradation scales with age AND heat exposure) and
generate labeled data from those relationships plus realistic noise.
This is a standard approach in predictive-maintenance research when
real failure-labeled data isn't accessible, and is disclosed as such
in the project report.

Components modeled (matches standard PPI checklist categories):
    Engine, Brakes, Battery, Transmission, Suspension, Tires
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CARS = 6000


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate_base_features(n):
    age_years = RNG.gamma(shape=2.2, scale=2.0, size=n).clip(0.2, 18)
    avg_daily_km = RNG.normal(35, 15, n).clip(5, 150)
    mileage_km = (age_years * avg_daily_km * 365 * RNG.normal(1, 0.15, n)).clip(500, 300000)

    # service behaviour
    service_gap_months = RNG.gamma(shape=2.0, scale=3.0, size=n).clip(1, 36)
    missed_services = (service_gap_months / 6 - 1).clip(0, None) + RNG.poisson(0.3, n)

    driving_conditions = RNG.choice(["urban", "highway", "mixed"], size=n, p=[0.45, 0.25, 0.30])
    climate = RNG.choice(["hot", "temperate", "coastal"], size=n, p=[0.4, 0.4, 0.2])

    return pd.DataFrame({
        "age_years": age_years.round(2),
        "mileage_km": mileage_km.round(0),
        "avg_daily_km": avg_daily_km.round(1),
        "service_gap_months": service_gap_months.round(1),
        "missed_services": missed_services.round(0).clip(0, None),
        "driving_conditions": driving_conditions,
        "climate": climate,
    })


def generate_sensor_readings(df):
    n = len(df)
    urban_frac = np.where(df.driving_conditions == "urban", 1.0,
                  np.where(df.driving_conditions == "mixed", 0.5, 0.1))
    hot_climate = (df.climate == "hot").astype(float)
    coastal = (df.climate == "coastal").astype(float)

    mileage_n = df.mileage_km / 200000       # normalized wear driver
    age_n = df.age_years / 15
    neglect = clip01(df.service_gap_months / 24 + df.missed_services * 0.05)

    # --- Engine ---
    oil_quality_index = clip01(1 - (neglect * 0.6 + mileage_n * 0.4)) * 100
    oil_quality_index = (oil_quality_index + RNG.normal(0, 5, n)).clip(0, 100)
    engine_temp_c = 88 + (1 - oil_quality_index / 100) * 25 + hot_climate * 5 + RNG.normal(0, 3, n)

    # --- Brakes (urban / stop-go driving accelerates pad wear) ---
    brake_wear = clip01(mileage_n * (0.5 + 0.5 * urban_frac) + neglect * 0.2)
    brake_pad_depth_mm = (10 * (1 - brake_wear)).clip(0.5, 10) + RNG.normal(0, 0.4, n)
    brake_pad_depth_mm = brake_pad_depth_mm.clip(0.5, 10)

    # --- Battery (age + heat are the dominant killers) ---
    battery_stress = clip01(age_n * 0.6 + hot_climate * 0.3 + neglect * 0.1)
    battery_voltage = (12.6 - battery_stress * 2.2 + RNG.normal(0, 0.15, n)).clip(9.5, 12.9)

    # --- Transmission ---
    trans_stress = clip01(mileage_n * 0.55 + neglect * 0.35 + urban_frac * 0.1)
    transmission_fluid_quality = (clip01(1 - trans_stress) * 100 + RNG.normal(0, 5, n)).clip(0, 100)

    # --- Suspension (road conditions proxy via urban fraction + mileage) ---
    suspension_stress = clip01(mileage_n * 0.5 + urban_frac * 0.3 + coastal * 0.1)
    vibration_score = (suspension_stress * 10 + RNG.normal(0, 0.8, n)).clip(0, 10)

    # --- Tires ---
    tire_stress = clip01(mileage_n * 0.7 + urban_frac * 0.2)
    tire_tread_depth_mm = (8 * (1 - tire_stress)).clip(0.5, 8) + RNG.normal(0, 0.3, n)
    tire_tread_depth_mm = tire_tread_depth_mm.clip(0.5, 8)

    sensors = pd.DataFrame({
        "oil_quality_index": oil_quality_index.round(1),
        "engine_temp_c": engine_temp_c.round(1),
        "brake_pad_depth_mm": brake_pad_depth_mm.round(2),
        "battery_voltage": battery_voltage.round(2),
        "transmission_fluid_quality": transmission_fluid_quality.round(1),
        "vibration_score": vibration_score.round(2),
        "tire_tread_depth_mm": tire_tread_depth_mm.round(2),
    })

    # stash intermediate stress terms for label generation
    stress = pd.DataFrame({
        "engine_stress": clip01(1 - oil_quality_index / 100) * 0.6 + mileage_n * 0.4,
        "brake_stress": brake_wear,
        "battery_stress": battery_stress,
        "trans_stress": trans_stress,
        "suspension_stress": suspension_stress,
        "tire_stress": tire_stress,
    })
    return sensors, stress


def stress_to_label(stress, n, k=8, midpoint=0.55):
    """Logistic mapping from stress (0-1) to failure-within-10k-km probability,
    with added noise so labels aren't a deterministic function of stress
    (mirrors real-world unpredictability of failures)."""
    noise = RNG.normal(0, 0.08, n)
    prob = 1 / (1 + np.exp(-k * (stress + noise - midpoint)))
    label = (RNG.uniform(0, 1, n) < prob).astype(int)
    return prob.round(3), label


def build_dataset(n=N_CARS):
    df = generate_base_features(n)
    sensors, stress = generate_sensor_readings(df)
    df = pd.concat([df, sensors], axis=1)

    for comp, col in [
        ("engine", "engine_stress"),
        ("brakes", "brake_stress"),
        ("battery", "battery_stress"),
        ("transmission", "trans_stress"),
        ("suspension", "suspension_stress"),
        ("tires", "tire_stress"),
    ]:
        prob, label = stress_to_label(stress[col].values, n)
        df[f"{comp}_fail_risk_prob"] = prob
        df[f"{comp}_fail_within_10k"] = label

    df.insert(0, "car_id", [f"C{100000+i}" for i in range(n)])
    return df


if __name__ == "__main__":
    dataset = build_dataset()
    os.makedirs("data", exist_ok=True)
    out_path = "data/car_health_ai_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Generated {len(dataset)} rows -> {out_path}")
    print("\nColumn summary:")
    print(dataset.describe(include='all').T[["count", "mean", "std", "min", "max"]].head(20))
    print("\nComponent failure base rates:")
    for comp in ["engine", "brakes", "battery", "transmission", "suspension", "tires"]:
        rate = dataset[f"{comp}_fail_within_10k"].mean()
        print(f"  {comp:14s}: {rate:.1%}")