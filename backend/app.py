"""
Cognitive CanSat FastAPI Real-Time Machine Learning Backend Server
Serves real-time inference using trained scikit-learn and PyOD models.

Entry point: uvicorn backend.app:app. Endpoints live in backend/routers/.
"""
import os
import logging
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.core import database

# Suppress minor version warnings for clean telemetry logs
warnings.filterwarnings("ignore", category=UserWarning)

# No-op when uvicorn (or another host) has already configured the root logger.
logging.basicConfig(
    level=os.environ.get("CANSAT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cansat.server")

# Initialize persistent SQLite telemetry & mission database
try:
    database.init_db()
    logger.info("SQLite telemetry & mission database initialized in WAL mode.")
except Exception as db_err:
    logger.warning("Could not initialize database: %s", db_err)

from backend import ml_service  # noqa: E402,F401  (loads the models on import)
from backend.routers import analysis, hardware, missions, ml  # noqa: E402
from backend.settings import FRONTEND_DIR, REPORTS_DIR, TEST_CASES_DIR  # noqa: E402

app = FastAPI(
    title="Cognitive CanSat ML Telemetry Server",
    description="Real-Time Machine Learning Inference API for Sounding Pico-Satellite Telemetry",
    version="2.0.0"
)

# Pages served by this server are same-origin and need no CORS. Cross-origin
# access is allowed for local development (file:// pages send Origin "null",
# dev servers on other localhost ports) plus any origins in CANSAT_CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CANSAT_CORS_ORIGINS", "").split(",") if o.strip()],
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (analysis, missions, ml, hardware):
    app.include_router(module.router)


# Short page aliases kept for old bookmarks.
PAGE_ALIASES = {"/home.html": "/index.html", "/models": "/models.html", "/analysis": "/analysis.html"}


def _redirect_to(target: str):
    return lambda: RedirectResponse(target, status_code=308)


for alias, target in PAGE_ALIASES.items():
    app.add_api_route(alias, _redirect_to(target), methods=["GET"], include_in_schema=False)


for url_path, directory in (("/test_cases", TEST_CASES_DIR), ("/reports", REPORTS_DIR)):
    if os.path.isdir(directory):
        app.mount(url_path, StaticFiles(directory=directory), name=url_path.strip("/"))

# Web UI (index.html at /, plus every page, css/ and js/). Mounted last so the
# API and WebSocket routes above take precedence.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
