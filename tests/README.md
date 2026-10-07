# Verification & Test Suites

This directory contains automated unit tests, verification suites, and synthetic flight data generators.

## Test Scripts

* **`selftest.ps1`**: Automated PowerShell test suite verifying mission elapsed time formatting, flight state machine phase progression across all 10 profiles, and mathematical bounds (17 assertions).
* **`selftest.js`**: Node.js offline unit tests verifying packet parsing, kinematics derivation (pitch, roll, vertical speed), and RFC 4180 CSV serialization (27 assertions).
* **`test_backend_core.py`**: Python unit test suite verifying 1D Kalman state estimation, 6-DOF attitude fusion, atmospheric sounding physics, tare calibration, and ML inference (12 assertions).
* **`test_api.py`**: FastAPI smoke tests over every REST/WebSocket endpoint, operator-key protection, CORS policy and the hardware-disabled mode (isolated temporary database).
* **`test_features.py`**: Shared ML feature engineering used by both training and live inference.
* **`test_frontend_static.py`**: Checks that every inline `on*=` handler in the dashboard is exported on `window`.
* **`test_firmware_protocol.py`**: Python binary struct packing and ESP-NOW chunk slicing / reassembly verification (4 assertions).
* **`test_sm.ps1`**: Focused state machine transition verifier for rapid flight phase debugging (10 profiles).
* **`generate_test_cases.ps1` / `generate_test_cases.js`**: Generators that synthesize the 10 distinct flight profiles stored in `test_cases/`.

## Running the Tests

CI (`.github/workflows/ci.yml`) runs these on every push and pull request. To run the same checks locally:
```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest -q tests
node tests/selftest.js
```

### 1. PowerShell Mission Verification Suite (17 assertions)
```powershell
powershell -ExecutionPolicy Bypass -File tests/selftest.ps1
```
Verifies Mission Elapsed Time (MET) clock formatting, CSV flight profiles across the 5-phase flight sequence, and UI component integrity.

### 2. Node.js Telemetry & Parsing Tests (27 assertions)
```bash
node tests/selftest.js
```
Validates 13-field CSV parsing, invalid packet rejection, kinematic derivations, attitude math, battery clamping, RFC 4180 export compliance, Web Serial compatibility, and UI styling tokens.

### 3. Python Backend Core Unit Tests (12 assertions)
```bash
.\.venv\Scripts\python.exe -m unittest tests/test_backend_core.py
```
Validates 1D Kalman state estimation convergence ($z, v_z$), complementary 6-DOF IMU attitude angles, high-G shock and gyro tumble alarms, barometric altimetry, moist air density, stationary tare calibration, and ML inference pipelines.

### 4. Firmware Protocol & ESP-NOW Chunking Tests (4 assertions)
```bash
.\.venv\Scripts\python.exe tests/test_firmware_protocol.py
```
Validates ESP-NOW 250-byte MTU constraints, 200-byte frame chunking, bit-for-bit SHA-256 JPEG payload reassembly, packet loss detection, and Base64 serial framing.

### 5. 5-Phase Flight State Machine Verification (10 profiles)
```powershell
powershell -ExecutionPolicy Bypass -File tests/test_sm.ps1
```
Validates end-to-end HMM and ML state transitions across all 10 mission profiles (`PAD_IDLE` -> `BALLOON_ASCENT` -> `APOGEE_BURST` -> `PARACHUTE_DESCENT` -> `TOUCHDOWN_RECOVERY`).

