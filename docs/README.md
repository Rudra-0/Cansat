# Cognitive CanSat Documentation & Mission Journals

This directory contains the engineering manuals, technical architecture formulations, and persistent activity logs.

## Documentation Index

* **[`architecture_and_ml.txt`](architecture_and_ml.txt)**: Comprehensive technical breakdown (written for 4th-year CSE / aerospace engineers) covering sensor kinematics, the full 6-model machine learning architecture (TinyLandingNet Edge Vision CNN, Random Forest Phase Classifier, GBDT Touchdown Prognostics, ExtraTrees Sensor Calibrator, GradientBoosting Apogee Regressor, and PyOD Isolation Forest), INT8 quantization footprints, latency benchmarks, confusion matrices, and precision/recall evaluations.
* **[`python-ml_focused.text`](python-ml_focused.text)**: The master architectural roadmap and tracking document detailing the transition to a Python-dominated backend, FastAPI routing, and TinyML edge vision pipeline.
* **[`GPU_TRAINING_INSTRUCTIONS.md`](GPU_TRAINING_INSTRUCTIONS.md)**: Standalone instructions and automated setup guide for the Antigravity agent running on the secondary NVIDIA GPU workstation.
* **[`project_log.txt`](project_log.txt)**: Chronological mission and development journal recording all implemented milestones, 3D PLA kit modeling, nadir camera orientation, side GPS bracket, blinking beacon highlight, `models.html` creation, `analysis.html` post-flight telemetry and sensor ablation suite, bugs resolved, and verification results.
* **[`tasks.txt`](../tasks.txt)**: Master task tracker with assigned deliverables, status badges, and milestone checklists (includes completed `TASK DA-06`).
* **[`QA_REPORT.md`](QA_REPORT.md)**: Comprehensive Quality Assurance and verification report auditing code, metrics, and documentation consistency.
* **[`audit_report.md`](audit_report.md)**: Systematic repository audit covering logic, data inconsistencies, priority classifications, and paper readiness.
* **[`research_paper_draft.md`](research_paper_draft.md)**: IEEE format research paper draft ("Autonomous Landing Site Selection for Sounding Pico-Satellites Using Quantized Edge TinyML") synthesizing the project's architecture, ML models, and experimental methodology.

## Repository Organization

The repository is structured into focused modular directories:

* **`frontend/`**: Complete web client suite containing all 6 presentation and mission control surfaces (`index.html`, `tinyml.html`, `dashboard.html`, `models.html`, `analysis.html`, `results.html`), stylesheets (`css/resend-theme.css`), and interactive visualization engines (`js/cansat-3d.js`, `js/mission-slider.js`, `js/results-charts.js`, `js/tinyml-demo.js`).
* **`backend/`**: FastAPI high-throughput async machine learning telemetry engine (`app.py`), standalone HTTP server (`standalone_server.ps1`), core analytical modules (`core/`), and persistent SQLite WAL flight telemetry database (`core/database.py`).
* **`ml/`**: Machine learning pipelines, INT8 quantized TinyLandingNet ONNX model, 6 scikit-learn models, flight analyzer script, and sensor ablation benchmarks.
* **`firmware/`**: Dual-band airborne ESP32 telemetry transmitter (LoRa 433 MHz + 2.4 GHz ESP-NOW / Wi-Fi) and ground station receiver firmware.
* **`reports/`**: Post-flight sounding reports, including all 10 mission PDF evaluations (`reports/pdf/`), 60+ 4-panel publication figures (`reports/figures/`), and summary metrics.
* **`analysis/`**: Post-flight telemetry analysis, sensor ablation evaluation notebook, and figure assets.
* **`test_cases/`**: 10 comprehensive CSV mission flight profiles covering nominal and extreme edge-case flight dynamics.
* **`tests/`**: Automated verification and batch test suites (`selftest.js`, `selftest.ps1`, `test_backend_core.py`, `test_database.py`, `test_firmware_protocol.py`, `test_sm.ps1`, `batch_test_ml.py`, `batch_test_ws.py`).
* **`scripts/`**: GPU workstation automated setup and dependency installation scripts.
* **`docs/`**: Technical documentation, engineering manuals, mission journals, and team task trackers.


