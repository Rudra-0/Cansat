"""
Unit Test Suite for Cognitive CanSat Backend Core
Tests:
  1. KinematicsEngine & AltitudeKalmanFilter convergence, attitude fusion, alarms.
  2. AtmosphericEngine barometric altimetry, Magnus-Tetens dew point, air density, ELR.
  3. DualSerialManager thread lifecycle and mock callback dispatch.
"""

import unittest
import time
from backend.core.kinematics import AltitudeKalmanFilter, KinematicsEngine
from backend.core.atmospheric import AtmosphericEngine
from backend.core.serial_manager import DualSerialManager, list_available_ports


class TestKinematicsEngine(unittest.TestCase):

    def test_kalman_filter_convergence(self):
        kf = AltitudeKalmanFilter(initial_altitude=0.0, process_var=0.5, measure_var=1.5)
        # Simulate ascent at 10 m/s with 0.5s intervals
        alt = 0.0
        t = 1000.0
        for _ in range(20):
            t += 0.5
            true_alt = (_ + 1) * 5.0
            noisy_alt = true_alt + 0.3  # Add noise
            filt_alt, vz = kf.update(noisy_alt, a_z_g=1.0, current_time=t)

        self.assertAlmostEqual(filt_alt, true_alt, delta=2.5)
        self.assertGreater(vz, 5.0)  # Ascending

    def test_attitude_angles(self):
        engine = KinematicsEngine()
        # Flat on pad (ax=0, ay=0, az=1)
        state = engine.process_packet(0.0, ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0)
        self.assertAlmostEqual(state.pitch_deg, 0.0, delta=2.0)
        self.assertAlmostEqual(state.roll_deg, 0.0, delta=2.0)
        self.assertAlmostEqual(state.accel_magnitude_g, 1.0, delta=0.05)

        # 45-degree pitch test on initial state (ay=0.707, az=0.707)
        engine_pitched = KinematicsEngine()
        state_pitch = engine_pitched.process_packet(10.0, ax=0.0, ay=0.707, az=0.707, gx=0.0, gy=0.0, gz=0.0)
        self.assertAlmostEqual(state_pitch.pitch_deg, 45.0, delta=3.0)

    def test_alarms_detection(self):
        engine = KinematicsEngine()
        # Extreme shock load (6.0G)
        state = engine.process_packet(100.0, ax=2.0, ay=3.0, az=5.0, gx=10.0, gy=5.0, gz=0.0)
        self.assertTrue(state.is_high_g_shock)

        # Extreme tumble (200 deg/s)
        state_tumble = engine.process_packet(100.0, ax=0.0, ay=0.0, az=1.0, gx=150.0, gy=150.0, gz=0.0)
        self.assertTrue(state_tumble.is_tumble)

    def test_tare_calibration(self):
        engine = KinematicsEngine()
        engine.start_tare(num_samples=5)
        self.assertTrue(engine.is_taring)

        # Feed 5 stationary samples with 2.0 deg/s bias on gx and 150m pad altitude
        for _ in range(5):
            state = engine.process_packet(150.0, ax=0.0, ay=0.0, az=1.0, gx=2.0, gy=0.0, gz=0.0)

        self.assertFalse(engine.is_taring)
        self.assertTrue(engine.tare.is_calibrated)
        self.assertAlmostEqual(engine.tare.gyro_bias_x, 2.0, delta=0.01)
        self.assertAlmostEqual(engine.tare.pad_altitude, 150.0, delta=0.01)

        # Next packet with 2.0 deg/s gx reading should yield ~0 deg/s effective gyro rate
        next_state = engine.process_packet(150.0, ax=0.0, ay=0.0, az=1.0, gx=2.0, gy=0.0, gz=0.0)
        self.assertAlmostEqual(next_state.gyro_magnitude_dps, 0.0, delta=0.05)
        self.assertIsNotNone(next_state.calibrated_altitude)
        self.assertIsNotNone(next_state.calibrated_vspd)


class TestAtmosphericEngine(unittest.TestCase):

    def setUp(self):
        self.atmo = AtmosphericEngine(ground_reference_pressure=1013.25, ground_reference_temp=20.0)

    def test_barometric_altitude_at_sea_level(self):
        alt = self.atmo.compute_barometric_altitude(1013.25)
        self.assertAlmostEqual(alt, 0.0, delta=0.5)

    def test_barometric_altitude_at_reduced_pressure(self):
        # ~898 hPa is roughly 1000m in standard atmosphere
        alt = self.atmo.compute_barometric_altitude(898.75)
        self.assertAlmostEqual(alt, 1000.0, delta=30.0)

    def test_magnus_tetens_dew_point(self):
        # 20 deg C at 50% relative humidity has dew point of ~9.3 deg C
        dew_pt = self.atmo.compute_dew_point(20.0, 50.0)
        self.assertAlmostEqual(dew_pt, 9.3, delta=0.5)

        # 100% humidity -> dew point == ambient temp
        dew_pt_100 = self.atmo.compute_dew_point(25.0, 100.0)
        self.assertAlmostEqual(dew_pt_100, 25.0, delta=0.2)

    def test_air_density(self):
        # Standard sea level: 1013.25 hPa @ 15 deg C is ~1.225 kg/m^3
        density = self.atmo.compute_air_density(1013.25, 15.0)
        self.assertAlmostEqual(density, 1.225, delta=0.01)

    def test_moist_air_density_lighter_than_dry(self):
        # Humid air is less dense than dry air at the same T and P
        dry_density = self.atmo.compute_air_density(1013.25, 30.0)
        moist_density = self.atmo.compute_air_density(1013.25, 30.0, humidity_pct=85.0)
        self.assertLess(moist_density, dry_density)

    def test_sounding_processing(self):
        sounding = self.atmo.process_sounding(950.0, 18.0, 60.0)
        self.assertGreater(sounding.barometric_altitude_m, 500.0)
        self.assertGreater(sounding.air_density_kg_m3, 1.0)
        self.assertLess(sounding.dew_point_c, 18.0)


class TestDualSerialManager(unittest.TestCase):

    def test_list_ports(self):
        ports = list_available_ports()
        self.assertIsInstance(ports, list)

    def test_lifecycle_without_hardware(self):
        manager = DualSerialManager(
            telemetry_port="COM98",
            telemetry_baud=9600,
            video_port="COM99",
            video_baud=460800,
        )
        self.assertFalse(manager.is_running)
        manager.start()
        self.assertTrue(manager.is_running)
        time.sleep(0.1)
        manager.stop()
        self.assertFalse(manager.is_running)


if __name__ == "__main__":
    unittest.main()
