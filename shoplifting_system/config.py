import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class PerformanceTargets:
    latency_ms_p95: int = 180
    min_effective_fps: int = 12
    max_cpu_percent: int = 75
    max_false_alert_rate_per_hour: int = 3
    min_stream_uptime_percent: float = 99.0
    max_processing_lag_ms: int = 250


@dataclass(frozen=True)
class SensitivityProfile:
    name: str
    confirmation_frames: int
    event_hold_seconds: float
    active_event_interval_seconds: float


SENSITIVITY_PROFILES: Dict[str, SensitivityProfile] = {
    "strict": SensitivityProfile(
        name="strict",
        confirmation_frames=3,
        event_hold_seconds=2.0,
        active_event_interval_seconds=1.5,
    ),
    "balanced": SensitivityProfile(
        name="balanced",
        confirmation_frames=2,
        event_hold_seconds=1.2,
        active_event_interval_seconds=1.0,
    ),
    "sensitive": SensitivityProfile(
        name="sensitive",
        confirmation_frames=1,
        event_hold_seconds=0.8,
        active_event_interval_seconds=0.8,
    ),
}


@dataclass(frozen=True)
class Settings:
    roboflow_api_key: str
    model_id: str
    camera_source: int | str
    fallback_camera_source: Optional[int | str]
    confidence_threshold: float
    frame_skip: int
    inference_width: int
    sensitivity_profile: str
    snapshot_cooldown_sec: float
    detection_cooldown_sec: float
    snapshot_retention_max: int
    reconnect_delay_sec: float
    reconnect_backoff_max_sec: float
    stream_error_frame_delay_sec: float
    host: str
    port: int
    log_level: str
    targets: PerformanceTargets


def parse_camera_source(raw_source: Any) -> int | str:
    if raw_source is None:
        return 0
    source_str = str(raw_source).strip()
    if source_str == "":
        return 0
    if source_str.isdigit():
        return int(source_str)
    return source_str


def parse_optional_camera_source(raw_source: Any) -> Optional[int | str]:
    if raw_source is None:
        return None
    source_str = str(raw_source).strip()
    if source_str == "":
        return None
    return parse_camera_source(source_str)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_profile(raw: Any) -> str:
    candidate = str(raw or "balanced").strip().lower() or "balanced"
    return candidate if candidate in SENSITIVITY_PROFILES else "balanced"


def load_settings_from_env() -> Settings:
    frame_skip = int(_clamp(float(os.environ.get("PROCESSING_FRAME_SKIP", "3")), 1, 30))
    confidence = _clamp(float(os.environ.get("CONFIDENCE_THRESHOLD", "0.40")), 0.05, 1.0)
    inference_width = int(_clamp(float(os.environ.get("INFERENCE_WIDTH", "960")), 320, 1920))
    snapshot_cooldown = _clamp(float(os.environ.get("SNAPSHOT_COOLDOWN_SECONDS", "3")), 1.0, 60.0)
    detection_cooldown = _clamp(float(os.environ.get("DETECTION_COOLDOWN_SECONDS", "3")), 0.0, 60.0)
    retention_max = int(_clamp(float(os.environ.get("SNAPSHOT_RETENTION_MAX", "200")), 20, 2000))

    return Settings(
        roboflow_api_key=os.environ.get("ROBOFLOW_API_KEY", "").strip(),
        model_id=os.environ.get("MODEL_ID", "shoplifting-detection/1").strip() or "shoplifting-detection/1",
        camera_source=parse_camera_source(os.environ.get("CAMERA_SOURCE", "0")),
        fallback_camera_source=parse_optional_camera_source(os.environ.get("FALLBACK_CAMERA_SOURCE", "")),
        confidence_threshold=confidence,
        frame_skip=frame_skip,
        inference_width=inference_width,
        sensitivity_profile=_normalize_profile(os.environ.get("SENSITIVITY_PROFILE", "balanced")),
        snapshot_cooldown_sec=snapshot_cooldown,
        detection_cooldown_sec=detection_cooldown,
        snapshot_retention_max=retention_max,
        reconnect_delay_sec=_clamp(float(os.environ.get("CAMERA_RECONNECT_DELAY_SECONDS", "1.0")), 0.2, 15.0),
        reconnect_backoff_max_sec=_clamp(float(os.environ.get("CAMERA_RECONNECT_MAX_BACKOFF_SECONDS", "10.0")), 1.0, 60.0),
        stream_error_frame_delay_sec=_clamp(float(os.environ.get("STREAM_ERROR_FRAME_DELAY_SECONDS", "0.4")), 0.1, 5.0),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(_clamp(float(os.environ.get("PORT", "5000")), 1, 65535)),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        targets=PerformanceTargets(),
    )


def validate_config_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    clean: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    if "camera_source" in data:
        source = str(data.get("camera_source", "")).strip()
        if source == "":
            errors["camera_source"] = "camera_source cannot be empty"
        else:
            clean["camera_source"] = parse_camera_source(source)

    if "fallback_camera_source" in data:
        clean["fallback_camera_source"] = parse_optional_camera_source(data.get("fallback_camera_source"))

    if "sensitivity_profile" in data:
        candidate = str(data.get("sensitivity_profile", "")).strip().lower()
        if candidate not in SENSITIVITY_PROFILES:
            errors["sensitivity_profile"] = "sensitivity_profile must be one of: strict, balanced, sensitive"
        else:
            clean["sensitivity_profile"] = candidate

    if "confidence_threshold" in data:
        try:
            value = float(data.get("confidence_threshold"))
            if not (0.05 <= value <= 1.0):
                raise ValueError
            clean["confidence_threshold"] = value
        except (TypeError, ValueError):
            errors["confidence_threshold"] = "confidence_threshold must be a float in [0.05, 1.0]"

    if "frame_skip" in data:
        try:
            value = int(data.get("frame_skip"))
            if not (1 <= value <= 30):
                raise ValueError
            clean["frame_skip"] = value
        except (TypeError, ValueError):
            errors["frame_skip"] = "frame_skip must be an integer in [1, 30]"

    if "inference_width" in data:
        try:
            value = int(data.get("inference_width"))
            if not (320 <= value <= 1920):
                raise ValueError
            clean["inference_width"] = value
        except (TypeError, ValueError):
            errors["inference_width"] = "inference_width must be an integer in [320, 1920]"

    if "snapshot_cooldown_sec" in data:
        try:
            value = float(data.get("snapshot_cooldown_sec"))
            if not (1.0 <= value <= 60.0):
                raise ValueError
            clean["snapshot_cooldown_sec"] = value
        except (TypeError, ValueError):
            errors["snapshot_cooldown_sec"] = "snapshot_cooldown_sec must be a float in [1.0, 60.0]"

    if "detection_cooldown_sec" in data:
        try:
            value = float(data.get("detection_cooldown_sec"))
            if not (0.0 <= value <= 60.0):
                raise ValueError
            clean["detection_cooldown_sec"] = value
        except (TypeError, ValueError):
            errors["detection_cooldown_sec"] = "detection_cooldown_sec must be a float in [0.0, 60.0]"

    if "snapshot_retention_max" in data:
        try:
            value = int(data.get("snapshot_retention_max"))
            if not (20 <= value <= 2000):
                raise ValueError
            clean["snapshot_retention_max"] = value
        except (TypeError, ValueError):
            errors["snapshot_retention_max"] = "snapshot_retention_max must be an integer in [20, 2000]"

    return clean, errors


def validate_startup_environment(settings: Settings) -> Optional[str]:
    if not settings.roboflow_api_key:
        return "ROBOFLOW_API_KEY is missing; inference will run in degraded mode without detections."
    return None
