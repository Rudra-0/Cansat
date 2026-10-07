"""
================================================================================
Cognitive CanSat - Descent Aerodynamics & Touchdown Prognostics Pipeline
Module: ml/train_touchdown_prognostics.py
Author: Flight Dynamics & Machine Learning Subsystem
--------------------------------------------------------------------------------
Trains:
1. Time-to-Touchdown (TTD) Point Regressor (Gradient Boosting Regressor)
2. Calibrated Uncertainty Estimators (10th and 90th Quantile Regressors)
3. Descent Aerodynamic Regime Classifier (Random Forest Classifier)
4. Comprehensive Statistical Analytics & High-Resolution Visualizations
================================================================================
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import joblib

# Headless plotting setup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, accuracy_score, confusion_matrix
)

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "test_cases")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")
FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "model_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [
    'altitude', 'vSpd', 'vAcc', 'pressure', 'temp', 'humidity',
    'air_density', 'dynamic_pressure', 'kinetic_energy',
    'ax', 'ay', 'az', 'accelMag', 'gx', 'gy', 'gz', 'gyroMag',
    'pitch', 'roll', 'dAlt_pad', 'batteryVoltage'
]

REGIME_CLASSES = [
    'NOMINAL_CANOPY',
    'PENDULUM_OSCILLATION',
    'BALLISTIC_FREEFALL',
    'TERMINAL_FLARE_TOUCHDOWN'
]

# -----------------------------------------------------------------------------
# 1. Physics Ingestion & Descent Feature Extraction
# -----------------------------------------------------------------------------
def process_flight_profile(filepath):
    """
    Ingests raw flight CSV, computes aerodynamic properties,
    and isolates the descent trajectory with ground-truth TTD.
    """
    df = pd.read_csv(filepath)
    filename = os.path.basename(filepath)
    n = len(df)
    dt = 0.8  # Telemetry broadcast interval in seconds

    if n < 10 or '10_ground_pad' in filename:
        return None

    # Numerical differentiation for vertical velocity (vSpd) and vertical acceleration (vAcc)
    vz = np.gradient(df['altitude'].values, dt)
    df['vSpd'] = vz
    df['vAcc'] = np.gradient(vz, dt)

    # Accelerometer & Gyroscope 3D Vector Norms
    df['accelMag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyroMag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    # Ideal Gas Law Atmospheric Air Density (kg/m^3)
    # rho = P_pa / (R_specific * T_kelvin)
    df['air_density'] = (df['pressure'] * 100.0) / (287.058 * (df['temp'] + 273.15))

    # Dynamic Pressure Proxy: q = 0.5 * rho * v_z^2 (Pa)
    df['dynamic_pressure'] = 0.5 * df['air_density'] * (df['vSpd']**2)

    # Kinetic Energy Proxy for 350g CanSat: 0.5 * m * v_z^2 (Joules)
    can_sat_mass_kg = 0.35
    df['kinetic_energy'] = 0.5 * can_sat_mass_kg * (df['vSpd']**2)

    # Pad baseline
    pad_alt = df['altitude'].iloc[0]
    df['dAlt_pad'] = df['altitude'] - pad_alt

    # Isolate Descent Stage (from Apogee peak down to ground touchdown)
    apogee_idx = df['altitude'].idxmax()
    descent_df = df.iloc[apogee_idx:].copy().reset_index(drop=True)

    # Locate touchdown index: first frame where altitude returns near pad level and stabilizes
    touch_idx = len(descent_df) - 1
    for i in range(len(descent_df) - 1, 0, -1):
        if descent_df.loc[i, 'altitude'] > pad_alt + 3.5:
            touch_idx = min(len(descent_df) - 1, i + 2)
            break

    # Ground-truth Time-to-Touchdown (TTD) in seconds
    ttd = (touch_idx - np.arange(len(descent_df))) * dt
    descent_df['ttd_seconds'] = np.maximum(0.0, ttd)

    # Aerodynamic Descent Stability Regime Labeling
    regimes = []
    for i in range(len(descent_df)):
        alt_pad = descent_df.loc[i, 'dAlt_pad']
        v = descent_df.loc[i, 'vSpd']
        gyro = descent_df.loc[i, 'gyroMag']

        if alt_pad <= 12.0:
            regimes.append('TERMINAL_FLARE_TOUCHDOWN')
        elif v < -13.0:
            regimes.append('BALLISTIC_FREEFALL')
        elif gyro > 42.0 and v < -3.0:
            regimes.append('PENDULUM_OSCILLATION')
        else:
            regimes.append('NOMINAL_CANOPY')

    descent_df['descent_regime'] = regimes
    descent_df['flight_scenario'] = filename
    return descent_df


# -----------------------------------------------------------------------------
# 2. Data Analytics & Visualization Engine
# -----------------------------------------------------------------------------
def run_descent_analytics(dataset):
    """
    Executes full statistical profiling, correlation analysis,
    and generates presentation-ready analytics figures.
    """
    print("\n" + "=" * 70)
    print("STEP 1: FULL DESCENT DATA ANALYTICS & STATISTICAL PROFILING")
    print("=" * 70)

    print(f"Total Descent Frames Analyzed: {len(dataset)}")
    print(f"Descent Altitude Range: {dataset['altitude'].min():.1f}m to {dataset['altitude'].max():.1f}m")
    print(f"Descent Velocity (vSpd) Range: {dataset['vSpd'].min():.2f}m/s to {dataset['vSpd'].max():.2f}m/s")
    print(f"Air Density Range: {dataset['air_density'].min():.4f} to {dataset['air_density'].max():.4f} kg/m^3")
    print(f"Maximum Gyroscope Oscillation: {dataset['gyroMag'].max():.1f} deg/s")

    print("\nDescent Aerodynamic Regime Distribution:")
    print(dataset['descent_regime'].value_counts())

    # --- FIGURE 1: Feature Correlation Heatmap ---
    corr_features = ['altitude', 'vSpd', 'pressure', 'temp', 'air_density',
                     'dynamic_pressure', 'kinetic_energy', 'gyroMag', 'ttd_seconds']
    corr_matrix = dataset[corr_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr_features)))
    ax.set_yticks(range(len(corr_features)))
    ax.set_xticklabels(corr_features, rotation=45, ha='left', fontsize=10, fontweight='bold')
    ax.set_yticklabels(corr_features, fontsize=10, fontweight='bold')

    for i in range(len(corr_features)):
        for j in range(len(corr_features)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    plt.title("CanSat Descent Telemetry - Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=25)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "01_correlation_matrix.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"[+] Saved Figure 1: {fig1_path}")

    # --- FIGURE 2: Aerodynamics & Atmospheric Profiling ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=200)

    # Subplot A: Altitude vs Time-to-Touchdown
    for scenario, grp in dataset.groupby('flight_scenario'):
        axs[0, 0].plot(grp['ttd_seconds'], grp['altitude'], label=scenario[:18], alpha=0.7, lw=1.5)
    axs[0, 0].set_xlabel("Time to Touchdown (s)", fontweight='bold')
    axs[0, 0].set_ylabel("Altitude (m)", fontweight='bold')
    axs[0, 0].set_title("Altitude Trajectory vs TTD (Descent Phase)", fontweight='bold')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Subplot B: Air Density vs Altitude
    axs[0, 1].scatter(dataset['altitude'], dataset['air_density'], c=dataset['temp'], cmap='plasma', s=6, alpha=0.6)
    axs[0, 1].set_xlabel("Altitude (m)", fontweight='bold')
    axs[0, 1].set_ylabel("Air Density (kg/m³)", fontweight='bold')
    axs[0, 1].set_title("Atmospheric Air Density Profile (Ideal Gas Law)", fontweight='bold')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Subplot C: Dynamic Pressure vs Vertical Speed
    axs[1, 0].scatter(dataset['vSpd'], dataset['dynamic_pressure'], c=dataset['altitude'], cmap='viridis', s=6, alpha=0.6)
    axs[1, 0].set_xlabel("Vertical Velocity vSpd (m/s)", fontweight='bold')
    axs[1, 0].set_ylabel("Dynamic Pressure q (Pa)", fontweight='bold')
    axs[1, 0].set_title("Dynamic Pressure vs Descent Velocity", fontweight='bold')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Subplot D: Gyroscope Oscillation Energy vs Regime
    regime_counts = dataset['descent_regime'].value_counts()
    axs[1, 1].bar(regime_counts.index, regime_counts.values, color=['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4'])
    axs[1, 1].set_xticklabels(regime_counts.index, rotation=25, ha='right', fontsize=9, fontweight='bold')
    axs[1, 1].set_ylabel("Frame Count", fontweight='bold')
    axs[1, 1].set_title("Aerodynamic Regime Class Balance", fontweight='bold')
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Cognitive CanSat Descent Aerodynamics & Atmospheric Profiling", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "02_aerodynamic_drag_profile.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"[+] Saved Figure 2: {fig2_path}")


# -----------------------------------------------------------------------------
# 3. Model Training & Uncertainty Quantification Engine
# -----------------------------------------------------------------------------
def train_touchdown_prognostics(dataset):
    """
    Trains:
    - Median Point TTD Regressor (GBR)
    - 10th & 90th Quantile Uncertainty Bounds (GBR with pinball/quantile loss)
    - Aerodynamic Regime Classifier (Random Forest)
    """
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING TOUCHDOWN PROGNOSTICS & UNCERTAINTY MODELS")
    print("=" * 70)

    X = dataset[FEATURE_COLS]
    y_ttd = dataset['ttd_seconds']
    y_regime = dataset['descent_regime']

    # Stratified split on regime for robust generalization
    X_train, X_test, y_train_ttd, y_test_ttd, y_train_reg, y_test_reg = train_test_split(
        X, y_ttd, y_regime, test_size=0.25, random_state=42, stratify=y_regime
    )

    print(f"Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Point Prediction Model (Mean/Median TTD Regressor)
    print("\n[1/3] Training Median TTD Regressor (Gradient Boosting)...")
    gbr_median = GradientBoostingRegressor(
        loss='squared_error',
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        random_state=42
    )
    gbr_median.fit(X_train_scaled, y_train_ttd)
    y_pred_median = gbr_median.predict(X_test_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_ttd, y_pred_median))
    mae = mean_absolute_error(y_test_ttd, y_pred_median)
    r2 = r2_score(y_test_ttd, y_pred_median)

    print(f"  -> TTD Regressor R^2 Score : {r2:.4f}")
    print(f"  -> TTD Regressor RMSE      : {rmse:.2f} seconds")
    print(f"  -> TTD Regressor MAE       : {mae:.2f} seconds")

    # Naive Baseline Physics Model: t = delta_h / |v_z|
    v_clamped = np.maximum(1.0, np.abs(X_test['vSpd'].values))
    y_pred_physics = np.maximum(0.0, X_test['dAlt_pad'].values / v_clamped)
    baseline_rmse = np.sqrt(mean_squared_error(y_test_ttd, y_pred_physics))
    baseline_mae = mean_absolute_error(y_test_ttd, y_pred_physics)
    print(f"  -> Naive Physics Baseline  : RMSE = {baseline_rmse:.2f}s, MAE = {baseline_mae:.2f}s")
    print(f"  -> ML Improvement over Naive: {(1.0 - rmse / baseline_rmse) * 100:.1f}% reduction in error!")

    # 2. Uncertainty Quantification: 10th and 90th Quantile Regressors
    print("\n[2/3] Training Uncertainty Bounds (10th & 90th Quantile Regressors)...")
    gbr_q10 = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.10,
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        random_state=42
    )
    gbr_q10.fit(X_train_scaled, y_train_ttd)
    y_pred_q10 = gbr_q10.predict(X_test_scaled)

    gbr_q90 = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.90,
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        random_state=42
    )
    gbr_q90.fit(X_train_scaled, y_train_ttd)
    y_pred_q90 = gbr_q90.predict(X_test_scaled)

    # Prediction Interval Coverage Probability (PICP)
    coverage = np.mean((y_test_ttd.values >= y_pred_q10) & (y_test_ttd.values <= y_pred_q90))
    print(f"  -> 80% Prediction Interval Coverage (PICP): {coverage * 100:.2f}% (Target: ~80%)")

    # 3. Aerodynamic Regime Classifier
    print("\n[3/3] Training Aerodynamic Regime Classifier (Random Forest)...")
    rf_regime = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    rf_regime.fit(X_train_scaled, y_train_reg)
    y_pred_reg = rf_regime.predict(X_test_scaled)
    reg_acc = accuracy_score(y_test_reg, y_pred_reg)
    reg_report = classification_report(y_test_reg, y_pred_reg, output_dict=True)

    print(f"  -> Regime Classifier Accuracy: {reg_acc * 100:.2f}%")

    # --- FIGURE 3: Residuals & Uncertainty Confidence Bands ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Actual vs Predicted with Uncertainty
    sort_idx = np.argsort(y_test_ttd.values)
    x_axis = np.arange(len(sort_idx))

    axs[0].scatter(y_test_ttd.values, y_pred_median, c='#1f77b4', s=12, alpha=0.6, label='Predicted TTD (Median)')
    ideal_line = [0, y_test_ttd.max()]
    axs[0].plot(ideal_line, ideal_line, 'r--', lw=2, label='Perfect 1:1 Parity')
    axs[0].set_xlabel("True Time-to-Touchdown (s)", fontweight='bold')
    axs[0].set_ylabel("Predicted Time-to-Touchdown (s)", fontweight='bold')
    axs[0].set_title(f"TTD Regressor Parity (R² = {r2:.4f}, RMSE = {rmse:.2f}s)", fontweight='bold')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Sorted Uncertainty Interval Demonstration
    sample_sub = sort_idx[::5]  # Downsample for clear visual intervals
    axs[1].fill_between(
        range(len(sample_sub)),
        y_pred_q10[sample_sub],
        y_pred_q90[sample_sub],
        color='#a6cee3',
        alpha=0.6,
        label='10th-90th Percentile Prediction Interval'
    )
    axs[1].plot(range(len(sample_sub)), y_test_ttd.values[sample_sub], 'k.', markersize=4, label='True TTD')
    axs[1].plot(range(len(sample_sub)), y_pred_median[sample_sub], 'b-', lw=1.2, label='Predicted Median TTD')
    axs[1].set_xlabel("Sorted Test Sample Index", fontweight='bold')
    axs[1].set_ylabel("Seconds Remaining", fontweight='bold')
    axs[1].set_title(f"Calibrated 80% Uncertainty Envelopes (Coverage = {coverage*100:.1f}%)", fontweight='bold')
    axs[1].legend(loc='upper left')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "03_model_benchmark_ttd.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"[+] Saved Figure 3: {fig3_path}")

    # --- FIGURE 4: Feature Importance Ranking ---
    importances = gbr_median.feature_importances_
    feat_order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    ax.barh([FEATURE_COLS[i] for i in feat_order], importances[feat_order], color='#2b83ba')
    ax.set_xlabel("Relative Feature Importance (Gradient Boosting)", fontweight='bold')
    ax.set_title("Touchdown Prognostics - Top Predictive Physical Signals", fontweight='bold', fontsize=13)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "04_feature_importance.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"[+] Saved Figure 4: {fig4_path}")

    # --- FIGURE 5: Regime Classifier Confusion Matrix ---
    cm = confusion_matrix(y_test_reg, y_pred_reg, labels=rf_regime.classes_)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(range(len(rf_regime.classes_)))
    ax.set_yticks(range(len(rf_regime.classes_)))
    ax.set_xticklabels(rf_regime.classes_, rotation=30, ha='left', fontweight='bold', fontsize=9)
    ax.set_yticklabels(rf_regime.classes_, fontweight='bold', fontsize=9)

    for i in range(len(rf_regime.classes_)):
        for j in range(len(rf_regime.classes_)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight='bold', fontsize=11)

    ax.set_xlabel("Predicted Regime", fontweight='bold', labelpad=10)
    ax.set_ylabel("True Ground Truth Regime", fontweight='bold')
    plt.title(f"Aerodynamic Regime Confusion Matrix (Acc = {reg_acc*100:.2f}%)", fontweight='bold', pad=20)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "05_descent_regime_confusion_matrix.png")
    plt.savefig(fig5_path)
    plt.close()
    print(f"[+] Saved Figure 5: {fig5_path}")

    # -------------------------------------------------------------------------
    # 4. Model Serialization & Export
    # -------------------------------------------------------------------------
    model_artifact = {
        'scaler': scaler,
        'model_median': gbr_median,
        'model_q10': gbr_q10,
        'model_q90': gbr_q90,
        'classifier_regime': rf_regime,
        'feature_names': FEATURE_COLS,
        'regime_classes': list(rf_regime.classes_),
        'metrics': {
            'ttd_r2': float(r2),
            'ttd_rmse_seconds': float(rmse),
            'ttd_mae_seconds': float(mae),
            'baseline_rmse_seconds': float(baseline_rmse),
            'picp_coverage': float(coverage),
            'regime_accuracy': float(reg_acc)
        }
    }

    model_path = os.path.join(MODELS_DIR, "touchdown_prognostics.joblib")
    joblib.dump(model_artifact, model_path)
    print(f"\n[+] Successfully serialized model artifact to: {model_path}")

    # Update model_metrics.json
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                master_metrics = json.load(f)
        except Exception:
            master_metrics = {}
    else:
        master_metrics = {}

    master_metrics['touchdown_prognostics'] = {
        'algorithm': 'GradientBoostingRegressor (Quantile Loss) + RandomForestClassifier',
        'dataset_descent_samples': len(dataset),
        'r2_score': float(round(r2, 4)),
        'rmse_seconds': float(round(rmse, 2)),
        'mae_seconds': float(round(mae, 2)),
        'naive_physics_baseline_rmse': float(round(baseline_rmse, 2)),
        'error_reduction_pct': float(round((1.0 - rmse / baseline_rmse) * 100, 1)),
        'uncertainty_80pct_coverage_picp': float(round(coverage * 100, 2)),
        'regime_classification_accuracy': float(round(reg_acc * 100, 2)),
        'features_count': len(FEATURE_COLS),
        'figures_generated': [
            'analysis/figures/01_correlation_matrix.png',
            'analysis/figures/02_aerodynamic_drag_profile.png',
            'analysis/figures/03_model_benchmark_ttd.png',
            'analysis/figures/04_feature_importance.png',
            'analysis/figures/05_descent_regime_confusion_matrix.png'
        ]
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(master_metrics, f, indent=2)
    print(f"[+] Updated master model metrics file: {METRICS_PATH}")

    return model_artifact


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("COGNITIVE CANSAT - TOUCHDOWN PROGNOSTICS & DESCENT ANALYTICS PIPELINE")
    print("=" * 70)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV flight test cases found in {DATA_DIR}")

    all_descent_dfs = []
    print(f"Parsing flight profiles from: {DATA_DIR}")
    for f in csv_files:
        processed = process_flight_profile(f)
        if processed is not None:
            all_descent_dfs.append(processed)
            print(f"  [+] Ingested: {os.path.basename(f)} ({len(processed)} descent frames)")

    combined_dataset = pd.concat(all_descent_dfs, ignore_index=True)
    print(f"\nTotal compiled descent dataset: {len(combined_dataset)} telemetry frames across {len(all_descent_dfs)} flight regimes.")

    # 1. Run Data Analytics
    run_descent_analytics(combined_dataset)

    # 2. Train and Serialize Models
    train_touchdown_prognostics(combined_dataset)

    print("\n" + "=" * 70)
    print("TOUCHDOWN PROGNOSTICS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
