from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from .api import register_routes
from .config import load_settings_from_env, validate_startup_environment
from .inference_engine import InferenceEngine
from .state import SharedState
from .stream import CameraStreamProcessor


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"
SNAPSHOT_DIR = STATIC_DIR / "snapshots"


def create_app() -> Flask:
    load_dotenv()
    settings = load_settings_from_env()

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    startup_warning = validate_startup_environment(settings)
    if startup_warning:
        logger.warning(startup_warning)

    app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR), static_url_path="/static")

    state = SharedState(
        camera_source=settings.camera_source,
        confidence_threshold=settings.confidence_threshold,
        frame_skip=settings.frame_skip,
    )

    inference_engine = InferenceEngine(
        model_id=settings.model_id,
        api_key=settings.roboflow_api_key,
        confidence_threshold=settings.confidence_threshold,
        inference_width=settings.inference_width,
    )
    inference_engine.initialize()

    stream_processor = CameraStreamProcessor(
        state=state,
        inference_engine=inference_engine,
        snapshot_dir=SNAPSHOT_DIR,
        snapshot_retention_max=settings.snapshot_retention_max,
        snapshot_cooldown_sec=settings.snapshot_cooldown_sec,
        detection_cooldown_sec=settings.detection_cooldown_sec,
        reconnect_delay_sec=settings.reconnect_delay_sec,
        reconnect_backoff_max_sec=settings.reconnect_backoff_max_sec,
        stream_error_frame_delay_sec=settings.stream_error_frame_delay_sec,
    )

    register_routes(app, state=state, stream_processor=stream_processor, settings=settings)

    app.config["runtime_settings"] = settings
    app.config["shared_state"] = state
    app.config["stream_processor"] = stream_processor

    return app
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv():
        return False
