"""
Cognitive CanSat - Shared ML Feature Engineering
------------------------------------------------
Single source of truth for the 17-feature vector consumed by the flight-phase
classifier, anomaly detector and apogee regressor. Used by both the training
pipeline (ml/train_models.py) and live inference (backend/app.py) so the two
cannot drift apart.

The models were trained on the synthetic test_cases/ profiles, which all start
from a sea-level pad (altitude ~0 m, pressure ~1013 hPa) on a 1S LiPo. Live
packets are therefore mapped into that frame before inference:
  - altitude becomes height above the pad,
  - pressure is replaced by the ISA pressure at that height (removes site
    elevation and weather offsets the models never saw),
  - battery voltage is clamped to the 1S range.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'altitude', 'vSpd', 'pressure', 'temp', 'humidity',
    'ax', 'ay', 'az', 'accelMag',
    'gx', 'gy', 'gz', 'gyroMag',
    'batteryVoltage', 'dAlt_pad', 'dTemp_pad', 'dPress_pad',
]

# International Standard Atmosphere (troposphere)
ISA_SEA_LEVEL_HPA = 1013.25
ISA_SEA_LEVEL_K = 288.15
ISA_LAPSE_RATE_K_PER_M = 0.0065
ISA_EXPONENT = 5.255

# Height above pad beyond which a packet is assumed to already carry relative
# altitude (e.g. pad baseline not set), so the raw altitude is used as-is.
MAX_RELATIVE_ALTITUDE_M = 2500.0

# 1S LiPo envelope seen in training. Readings above the plausible 1S maximum
# (e.g. a 2S pack) are replaced by a nominal mid-charge value.
LIPO_1S_MIN_V = 3.30
LIPO_1S_MAX_V = 4.20
LIPO_1S_PLAUSIBLE_MAX_V = 4.30
LIPO_1S_NOMINAL_V = 3.85

DEFAULT_PAD_TEMP_C = 25.0


def isa_pressure_hpa(altitude_m: float) -> float:
    """ISA barometric pressure (hPa) at a geometric altitude above sea level."""
    ratio = max(0.1, 1.0 - (ISA_LAPSE_RATE_K_PER_M * altitude_m) / ISA_SEA_LEVEL_K)
    return ISA_SEA_LEVEL_HPA * math.pow(ratio, ISA_EXPONENT)


def add_derived_columns(df: pd.DataFrame, pad_alt: float, pad_temp: float, pad_press: float) -> pd.DataFrame:
    """Add accelMag, gyroMag and pad-relative delta columns in place and return df."""
    df['accelMag'] = np.sqrt(df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2)
    df['gyroMag'] = np.sqrt(df['gx'] ** 2 + df['gy'] ** 2 + df['gz'] ** 2)
    df['dAlt_pad'] = df['altitude'] - pad_alt
    df['dTemp_pad'] = df['temp'] - pad_temp
    df['dPress_pad'] = df['pressure'] - pad_press
    return df


def normalize_battery(voltage: float) -> float:
    if voltage > LIPO_1S_PLAUSIBLE_MAX_V:
        return LIPO_1S_NOMINAL_V
    return min(LIPO_1S_MAX_V, max(LIPO_1S_MIN_V, voltage))


def resolve_pad_baseline(altitude: float, temp: float, pad_alt: Optional[float],
                         pad_temp: Optional[float], pad_press: Optional[float]) -> Tuple[float, float, float]:
    """Fill in a missing pad baseline from the current packet (or ISA sea level for pressure)."""
    return (
        pad_alt if (pad_alt is not None and pad_alt > 0) else altitude,
        pad_temp if pad_temp is not None else temp,
        pad_press if pad_press is not None else ISA_SEA_LEVEL_HPA,
    )


def build_inference_features(
    *, temp: float, pressure: float, altitude: float,
    ax: float, ay: float, az: float, gx: float, gy: float, gz: float,
    humidity: float, battery_voltage: float, v_spd: float,
    pad_alt: Optional[float] = None, pad_temp: Optional[float] = None,
    pad_press: Optional[float] = None,
) -> Dict[str, float]:
    """Map one live telemetry packet into the training frame (see module docstring)."""
    pad_alt, pad_temp, pad_press = resolve_pad_baseline(altitude, temp, pad_alt, pad_temp, pad_press)

    d_alt = altitude - pad_alt
    ml_alt = max(0.0, d_alt) if abs(d_alt) < MAX_RELATIVE_ALTITUDE_M else altitude

    row = {
        'altitude': ml_alt, 'vSpd': v_spd, 'pressure': isa_pressure_hpa(ml_alt),
        'temp': temp, 'humidity': humidity,
        'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz,
        'batteryVoltage': normalize_battery(battery_voltage),
    }
    row['accelMag'] = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    row['gyroMag'] = math.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    # Pad deltas use the raw (un-normalized) packet, matching training.
    row['dAlt_pad'] = d_alt
    row['dTemp_pad'] = temp - pad_temp
    row['dPress_pad'] = pressure - pad_press
    return row


def to_frame(row: Dict[str, float], feature_names=FEATURE_COLUMNS) -> pd.DataFrame:
    """One-row DataFrame in the column order the fitted pipelines expect."""
    return pd.DataFrame({k: [row[k]] for k in feature_names})
