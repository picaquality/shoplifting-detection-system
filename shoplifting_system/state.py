from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class RuntimeMetrics:
    started_at: float = field(default_factory=time.time)
    frames_read: int = 0
    frames_processed: int = 0
    dropped_frames: int = 0
    reconnect_count: int = 0
    backpressure_events: int = 0
    inference_ms_ema: float = 0.0
    effective_fps_ema: float = 0.0


class SharedState:
    def __init__(
        self,
        camera_source: int | str,
        confidence_threshold: float,
        frame_skip: int,
        sensitivity_profile: str,
        fallback_camera_source: Optional[int | str] = None,
    ):
        self._lock = threading.RLock()
        self.camera_source = camera_source
        self.fallback_camera_source = fallback_camera_source
        self.confidence_threshold = confidence_threshold
        self.base_frame_skip = frame_skip
        self.adaptive_frame_skip = frame_skip
        self.sensitivity_profile = sensitivity_profile

        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_snapshot: Optional[str] = None
        self.last_detection_fingerprint: str = ""
        self.last_detection_time: float = 0.0
        self.last_snapshot_time: float = 0.0

        self.alert_event: Optional[Dict[str, Any]] = None
        self.recent_alert_events: Deque[Dict[str, Any]] = deque(maxlen=25)
        self.alert_timestamps: Deque[float] = deque(maxlen=1000)

        self.feedback_false_alert_count: int = 0
        self.feedback_confirmed_incident_count: int = 0

        self.connection_status: str = "initializing"
        self.last_error: Optional[str] = None
        self.pending_restart: bool = False

        self.metrics = RuntimeMetrics()

    def lock(self):
        return self._lock

    def _prune_alert_timestamps(self, now: float, window_seconds: float = 3600.0) -> None:
        while self.alert_timestamps and (now - self.alert_timestamps[0]) > window_seconds:
            self.alert_timestamps.popleft()

    def alert_rate_per_hour(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        self._prune_alert_timestamps(now)
        return float(len(self.alert_timestamps))

    def false_alert_ratio(self) -> float:
        total_feedback = self.feedback_false_alert_count + self.feedback_confirmed_incident_count
        if total_feedback == 0:
            return 0.0
        return self.feedback_false_alert_count / float(total_feedback)

    def snapshot_status(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            uptime_seconds = max(0, now - self.metrics.started_at)
            alert_rate = self.alert_rate_per_hour(now)
            false_alert_ratio = self.false_alert_ratio()
            return {
                "detections": [{"label": d["label"], "confidence": d["confidence"]} for d in self.latest_detections],
                "current_source": self.camera_source,
                "fallback_source": self.fallback_camera_source,
                "snapshot_url": self.latest_snapshot,
                "confidence_threshold": self.confidence_threshold,
                "frame_skip": self.base_frame_skip,
                "adaptive_frame_skip": self.adaptive_frame_skip,
                "sensitivity_profile": self.sensitivity_profile,
                "connection_status": self.connection_status,
                "last_error": self.last_error,
                "alert_event": self.alert_event,
                "recent_alert_events": list(self.recent_alert_events),
                "feedback": {
                    "false_alert_count": self.feedback_false_alert_count,
                    "confirmed_incident_count": self.feedback_confirmed_incident_count,
                    "false_alert_ratio": round(false_alert_ratio, 4),
                },
                "metrics": {
                    "uptime_seconds": round(uptime_seconds, 2),
                    "frames_read": self.metrics.frames_read,
                    "frames_processed": self.metrics.frames_processed,
                    "dropped_frames": self.metrics.dropped_frames,
                    "reconnect_count": self.metrics.reconnect_count,
                    "backpressure_events": self.metrics.backpressure_events,
                    "inference_ms_ema": round(self.metrics.inference_ms_ema, 2),
                    "effective_fps_ema": round(self.metrics.effective_fps_ema, 2),
                    "alert_rate_per_hour": round(alert_rate, 2),
                    "false_alert_ratio": round(false_alert_ratio, 4),
                },
            }

    def update_config(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            if "camera_source" in updates:
                self.camera_source = updates["camera_source"]
            if "fallback_camera_source" in updates:
                self.fallback_camera_source = updates["fallback_camera_source"]
            if "confidence_threshold" in updates:
                self.confidence_threshold = float(updates["confidence_threshold"])
            if "frame_skip" in updates:
                self.base_frame_skip = int(updates["frame_skip"])
                self.adaptive_frame_skip = int(updates["frame_skip"])
            if "sensitivity_profile" in updates:
                self.sensitivity_profile = str(updates["sensitivity_profile"])

    def set_connection_state(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self.connection_status = status
            self.last_error = error

    def record_alert_event(
        self,
        *,
        event_type: str,
        event_id: int,
        severity: str,
        confidence_band: str,
        confidence: float,
        now: float,
    ) -> None:
        with self._lock:
            payload = {
                "event_type": event_type,
                "event_id": event_id,
                "severity": severity,
                "confidence_band": confidence_band,
                "confidence": round(confidence, 4),
                "timestamp": round(now, 3),
            }
            self.alert_event = payload
            self.recent_alert_events.append(payload)
            if event_type == "start":
                self.alert_timestamps.append(now)
                self._prune_alert_timestamps(now)

    def record_feedback(self, feedback_type: str) -> None:
        with self._lock:
            if feedback_type == "false_alert":
                self.feedback_false_alert_count += 1
            elif feedback_type == "confirmed_incident":
                self.feedback_confirmed_incident_count += 1

    def mark_restart_requested(self) -> None:
        with self._lock:
            self.pending_restart = True

    def consume_restart_request(self) -> bool:
        with self._lock:
            requested = self.pending_restart
            self.pending_restart = False
            return requested
