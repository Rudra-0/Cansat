"""Filesystem locations shared by the server modules."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "model_metrics.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
