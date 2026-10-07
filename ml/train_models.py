"""
Machine Learning Training Pipeline for Cognitive CanSat Telemetry
Trains:
1. Flight Phase Classifier: Random Forest Classifier (Multi-Class)
2. Anomaly & Outlier Detector: PyOD Isolation Forest (Unsupervised)
3. Apogee Regressor: Gradient Boosting Regressor
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score
from pyod.models.iforest import IForest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from backend.core.features import FEATURE_COLUMNS, add_derived_columns  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "test_cases")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

PHASE_LABELS = ['PAD_IDLE', 'BALLOON_ASCENT', 'APOGEE_BURST', 'PARACHUTE_DESCENT', 'TOUCHDOWN_RECOVERY']

def label_flight_profile(df, filename):
    """
    Labels each telemetry frame into one of the 5 flight stages
    based on ground truth flight dynamics and physics boundaries.
    """
    df = df.copy()
    n = len(df)
    
    # Calculate numerical vertical speed vz
    if 'altitude' in df.columns:
        dt = 0.8
        vz = np.gradient(df['altitude'].values, dt)
        df['vSpd'] = vz
    else:
        df['vSpd'] = 0.0

    # Magnitudes and pad-relative deltas (shared with live inference)
    pad_alt = df['altitude'].iloc[0] if n > 0 else 0.0
    pad_temp = df['temp'].iloc[0] if n > 0 else 25.0
    pad_press = df['pressure'].iloc[0] if n > 0 else 1013.25
    add_derived_columns(df, pad_alt, pad_temp, pad_press)
    
    # Static pad test profile
    if "10_ground_pad_static_test" in filename:
        df['phase'] = 'PAD_IDLE'
        df['is_nominal'] = 1
        df['target_apogee'] = pad_alt
        return df

    # Find apogee index
    max_alt = df['altitude'].max()
    apogee_idx = df['altitude'].idxmax()
    df['target_apogee'] = max_alt

    phases = ['PAD_IDLE'] * n
    
    for i in range(n):
        alt = df.loc[i, 'altitude']
        v = df.loc[i, 'vSpd']
        acc = df.loc[i, 'accelMag']
        
        if i < 25 and alt < pad_alt + 5.0 and abs(v) < 1.0:
            phases[i] = 'PAD_IDLE'
        elif i <= apogee_idx - 2:
            phases[i] = 'BALLOON_ASCENT'
        elif abs(i - apogee_idx) <= 3 or (i > apogee_idx and i <= apogee_idx + 4 and acc > 1.8):
            phases[i] = 'APOGEE_BURST'
        else:
            # Check for landed condition near end
            if i > apogee_idx + 10 and (alt <= pad_alt + 4.0 or alt <= 5.0) and abs(v) < 1.2:
                phases[i] = 'TOUCHDOWN_RECOVERY'
            else:
                phases[i] = 'PARACHUTE_DESCENT'

    df['phase'] = phases
    
    # Nominal vs Anomaly labeling for unsupervised baseline
    is_anomaly_file = any(k in filename for k in ['03_severe_wind_shear', '04_apogee_ejection', '05_parachute_pendulum', '08_sensor_glitch', '09_low_battery'])
    df['is_nominal'] = 0 if is_anomaly_file else 1

    return df

def extract_features(df):
    return df[FEATURE_COLUMNS]

def main():
    print("=" * 70)
    print("COGNITIVE CANSAT ML MODEL TRAINING PIPELINE")
    print("=" * 70)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV test cases found in {DATA_DIR}")

    print(f"Loading {len(csv_files)} flight profiles...")
    all_dfs = []
    for f in csv_files:
        raw_df = pd.read_csv(f)
        labeled_df = label_flight_profile(raw_df, os.path.basename(f))
        all_dfs.append(labeled_df)
        print(f"  Loaded: {os.path.basename(f)} ({len(labeled_df)} rows)")

    full_dataset = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal Dataset Samples: {len(full_dataset)} records across {len(PHASE_LABELS)} phases.")
    print("Phase Class Distribution:")
    print(full_dataset['phase'].value_counts())

    X = extract_features(full_dataset)
    y = full_dataset['phase']
    feature_names = list(X.columns)

    # -------------------------------------------------------------
    # 1. TRAIN FLIGHT PHASE CLASSIFIER (Random Forest)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. TRAINING FLIGHT PHASE CLASSIFIER (RandomForestClassifier)")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    classifier_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=3,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    classifier_pipeline.fit(X_train, y_train)
    y_pred = classifier_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    clf_path = os.path.join(MODELS_DIR, "flight_phase_classifier.joblib")
    joblib.dump({
        'pipeline': classifier_pipeline,
        'feature_names': feature_names,
        'classes': list(classifier_pipeline.classes_)
    }, clf_path)
    print(f"Saved Flight Phase Classifier to: {clf_path}")

    # -------------------------------------------------------------
    # 2. TRAIN UNSUPERVISED OUTLIER / ANOMALY DETECTOR (PyOD IForest)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. TRAINING UNSUPERVISED ANOMALY DETECTOR (PyOD IForest)")
    print("-" * 70)

    nominal_data = full_dataset[full_dataset['is_nominal'] == 1]
    X_nominal = extract_features(nominal_data)

    scaler_anomaly = StandardScaler()
    X_nominal_scaled = scaler_anomaly.fit_transform(X_nominal)

    anomaly_model = IForest(
        n_estimators=100,
        contamination=0.06,
        random_state=42,
        behaviour='new'
    )
    anomaly_model.fit(X_nominal_scaled)

    # Score full dataset
    X_full_scaled = scaler_anomaly.transform(X)
    anomaly_scores = anomaly_model.decision_function(X_full_scaled)
    min_s, max_s = anomaly_scores.min(), anomaly_scores.max()
    normalized_scores = (anomaly_scores - min_s) / (max_s - min_s + 1e-9)

    print(f"Trained PyOD IForest on {len(X_nominal)} nominal flight telemetry frames.")
    print(f"Baseline Anomaly Score Range: [{min_s:.4f}, {max_s:.4f}]")

    anomaly_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
    joblib.dump({
        'scaler': scaler_anomaly,
        'model': anomaly_model,
        'score_min': float(min_s),
        'score_max': float(max_s),
        'feature_names': feature_names
    }, anomaly_path)
    print(f"Saved PyOD Anomaly Detector to: {anomaly_path}")

    # -------------------------------------------------------------
    # 3. TRAIN APOGEE & TRAJECTORY REGRESSOR (Gradient Boosting)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. TRAINING APOGEE & TRAJECTORY REGRESSOR (GradientBoostingRegressor)")
    print("-" * 70)

    ascent_data = full_dataset[full_dataset['phase'].isin(['BALLOON_ASCENT', 'PAD_IDLE'])]
    X_ascent = extract_features(ascent_data)
    y_ascent = ascent_data['target_apogee']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_ascent, y_ascent, test_size=0.25, random_state=42)

    regressor_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ))
    ])

    regressor_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = regressor_pipeline.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

    print(f"Apogee Regressor R^2 Score: {r2:.4f}")
    print(f"Apogee Regressor RMSE: {rmse:.2f} meters")

    reg_path = os.path.join(MODELS_DIR, "apogee_regressor.joblib")
    joblib.dump({
        'pipeline': regressor_pipeline,
        'feature_names': feature_names
    }, reg_path)
    print(f"Saved Apogee Regressor to: {reg_path}")

    # -------------------------------------------------------------
    # 4. SAVE MODEL METRICS & FEATURE IMPORTANCES
    # -------------------------------------------------------------
    rf_feature_importances = dict(zip(feature_names, [float(x) for x in classifier_pipeline.named_steps['classifier'].feature_importances_]))
    
    metrics = {
        "dataset_total_samples": len(full_dataset),
        "flight_phase_classifier": {
            "algorithm": "RandomForestClassifier",
            "test_accuracy": float(acc),
            "classification_report": clf_report,
            "classes": list(classifier_pipeline.classes_),
            "feature_importances": rf_feature_importances
        },
        "anomaly_detector": {
            "algorithm": "PyOD.IForest",
            "nominal_samples_trained": len(X_nominal),
            "contamination_threshold": 0.06,
            "raw_score_min": float(min_s),
            "raw_score_max": float(max_s)
        },
        "apogee_regressor": {
            "algorithm": "GradientBoostingRegressor",
            "r2_score": float(r2),
            "rmse_meters": float(rmse)
        }
    }

    metrics_path = os.path.join(BASE_DIR, "ml", "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved Model Evaluation Metrics JSON to: {metrics_path}")

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED & SERIALIZED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
