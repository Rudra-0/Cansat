"""Unit tests for backend/core/features.py (shared training/inference feature engineering)."""

import unittest

import pandas as pd

from backend.core import features


class TestFeatures(unittest.TestCase):

    def _build(self, **overrides):
        kw = dict(temp=20.0, pressure=1000.0, altitude=300.0, ax=0.0, ay=0.0, az=1.0,
                  gx=3.0, gy=4.0, gz=0.0, humidity=50.0, battery_voltage=4.0, v_spd=2.0)
        kw.update(overrides)
        return features.build_inference_features(**kw)

    def test_isa_pressure(self):
        self.assertAlmostEqual(features.isa_pressure_hpa(0.0), 1013.25, places=2)
        self.assertAlmostEqual(features.isa_pressure_hpa(1000.0), 898.7, delta=0.5)

    def test_relative_altitude_and_isa_pressure(self):
        row = self._build(altitude=300.0, pad_alt=100.0)
        self.assertEqual(row['altitude'], 200.0)
        self.assertEqual(row['dAlt_pad'], 200.0)
        self.assertAlmostEqual(row['pressure'], features.isa_pressure_hpa(200.0))

    def test_missing_pad_uses_current_packet(self):
        row = self._build(altitude=300.0, pad_alt=0.0)
        self.assertEqual(row['altitude'], 0.0)
        self.assertEqual(row['dTemp_pad'], 0.0)
        self.assertEqual(row['dPress_pad'], 1000.0 - features.ISA_SEA_LEVEL_HPA)

    def test_large_offset_keeps_raw_altitude(self):
        row = self._build(altitude=3000.0, pad_alt=100.0)
        self.assertEqual(row['altitude'], 3000.0)

    def test_battery_normalization(self):
        self.assertEqual(features.normalize_battery(3.0), features.LIPO_1S_MIN_V)
        self.assertEqual(features.normalize_battery(4.25), features.LIPO_1S_MAX_V)
        self.assertEqual(features.normalize_battery(7.8), features.LIPO_1S_NOMINAL_V)
        self.assertEqual(features.normalize_battery(3.9), 3.9)

    def test_magnitudes_and_frame_order(self):
        row = self._build()
        self.assertEqual(row['gyroMag'], 5.0)
        self.assertEqual(row['accelMag'], 1.0)
        frame = features.to_frame(row)
        self.assertEqual(list(frame.columns), features.FEATURE_COLUMNS)

    def test_add_derived_columns_matches_inference(self):
        df = pd.DataFrame({'altitude': [100.0, 300.0], 'temp': [20.0, 18.0], 'pressure': [1001.0, 978.0],
                           'ax': [0.0, 0.0], 'ay': [0.0, 0.0], 'az': [1.0, 1.0],
                           'gx': [3.0, 3.0], 'gy': [4.0, 4.0], 'gz': [0.0, 0.0]})
        features.add_derived_columns(df, 100.0, 20.0, 1001.0)
        row = self._build(altitude=300.0, temp=18.0, pressure=978.0, pad_alt=100.0, pad_temp=20.0, pad_press=1001.0)
        for col in ['accelMag', 'gyroMag', 'dAlt_pad', 'dTemp_pad', 'dPress_pad']:
            self.assertAlmostEqual(df[col].iloc[1], row[col])


if __name__ == "__main__":
    unittest.main()
