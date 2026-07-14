from __future__ import annotations

from flask import Response, jsonify, render_template, request

from .config import SENSITIVITY_PROFILES, validate_config_payload


def register_routes(app, state, stream_processor, settings):
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(stream_processor.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def get_status():
        payload = state.snapshot_status()
        payload["status"] = "active" if payload["connection_status"] in {"healthy", "recovering"} else "degraded"
        return jsonify(payload)

    @app.route("/api/metrics")
    def get_metrics():
        payload = state.snapshot_status()["metrics"]
        payload["targets"] = {
            "latency_ms_p95": settings.targets.latency_ms_p95,
            "min_effective_fps": settings.targets.min_effective_fps,
            "max_cpu_percent": settings.targets.max_cpu_percent,
            "max_false_alert_rate_per_hour": settings.targets.max_false_alert_rate_per_hour,
            "min_stream_uptime_percent": settings.targets.min_stream_uptime_percent,
            "max_processing_lag_ms": settings.targets.max_processing_lag_ms,
        }
        return jsonify(payload)

    @app.route("/api/targets")
    def get_targets():
        return jsonify(
            {
                "latency_ms_p95": settings.targets.latency_ms_p95,
                "min_effective_fps": settings.targets.min_effective_fps,
                "max_cpu_percent": settings.targets.max_cpu_percent,
                "max_false_alert_rate_per_hour": settings.targets.max_false_alert_rate_per_hour,
                "min_stream_uptime_percent": settings.targets.min_stream_uptime_percent,
                "max_processing_lag_ms": settings.targets.max_processing_lag_ms,
            }
        )

    @app.route("/api/config", methods=["POST"])
    def update_config():
        data = request.get_json(silent=True) or {}
        clean, errors = validate_config_payload(data)
        if errors:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid configuration payload",
                        "errors": errors,
                    }
                ),
                400,
            )

        if clean:
            state.update_config(clean)
            stream_processor.apply_runtime_config(clean)
            stream_processor.request_restart()

        return jsonify(
            {
                "success": True,
                "message": "Settings updated",
                "applied": {
                    "camera_source": state.camera_source,
                    "fallback_camera_source": state.fallback_camera_source,
                    "confidence_threshold": state.confidence_threshold,
                    "frame_skip": state.base_frame_skip,
                    "sensitivity_profile": state.sensitivity_profile,
                    "allowed_sensitivity_profiles": sorted(SENSITIVITY_PROFILES.keys()),
                },
            }
        )

    @app.route("/api/feedback", methods=["POST"])
    def add_feedback():
        data = request.get_json(silent=True) or {}
        feedback_type = str(data.get("type", "")).strip().lower()

        if feedback_type not in {"false_alert", "confirmed_incident"}:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid feedback type",
                        "errors": {"type": "type must be one of: false_alert, confirmed_incident"},
                    }
                ),
                400,
            )

        state.record_feedback(feedback_type)
        snapshot = state.snapshot_status()["feedback"]
        return jsonify(
            {
                "success": True,
                "message": "Feedback recorded",
                "feedback": snapshot,
            }
        )
