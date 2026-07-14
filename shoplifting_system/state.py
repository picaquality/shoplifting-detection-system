from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeMetrics:
    started_at: float = field(default_factory=time.time)
    frames_read: int = 0
    frames_processed: int = 0
    dropped_frames: int = 0
    reconnect_count: int = 0
    inference_ms_ema: float = 0.0
    effective_fps_ema: float = 0.0


class SharedState:
    def __init__(self, camera_source: int | str, confidence_threshold: float, frame_skip: int):
        self._lock = threading.RLock()
        self.camera_source = camera_source
        self.confidence_threshold = confidence_threshold
        self.base_frame_skip = frame_skip
        self.adaptive_frame_skip = frame_skip

        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_snapshot: Optional[str] = None
        self.last_detection_fingerprint: str = ""
        self.last_detection_time: float = 0.0
        self.last_snapshot_time: float = 0.0

        self.connection_status: str = "initializing"
        self.last_error: Optional[str] = None
        self.pending_restart: bool = False

        self.metrics = RuntimeMetrics()

    def lock(self):
        return self._lock

    def snapshot_status(self) -> Dict[str, Any]:
        with self._lock:
            uptime_seconds = max(0, time.time() - self.metrics.started_at)
            return {
                "detections": [{"label": d["label"], "confidence": d["confidence"]} for d in self.latest_detections],
                "current_source": self.camera_source,
                "snapshot_url": self.latest_snapshot,
                "confidence_threshold": self.confidence_threshold,
                "frame_skip": self.base_frame_skip,
                "adaptive_frame_skip": self.adaptive_frame_skip,
                "connection_status": self.connection_status,
                "last_error": self.last_error,
                "metrics": {
                    "uptime_seconds": round(uptime_seconds, 2),
                    "frames_read": self.metrics.frames_read,
                    "frames_processed": self.metrics.frames_processed,
                    "dropped_frames": self.metrics.dropped_frames,
                    "reconnect_count": self.metrics.reconnect_count,
                    "inference_ms_ema": round(self.metrics.inference_ms_ema, 2),
                    "effective_fps_ema": round(self.metrics.effective_fps_ema, 2),
                },
            }

    def update_config(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            if "camera_source" in updates:
                self.camera_source = updates["camera_source"]
            if "confidence_threshold" in updates:
                self.confidence_threshold = float(updates["confidence_threshold"])
            if "frame_skip" in updates:
                self.base_frame_skip = int(updates["frame_skip"])
                self.adaptive_frame_skip = int(updates["frame_skip"])

    def set_connection_state(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self.connection_status = status
            self.last_error = error

    def mark_restart_requested(self) -> None:
        with self._lock:
            self.pending_restart = True

    def consume_restart_request(self) -> bool:
        with self._lock:
            requested = self.pending_restart
            self.pending_restart = False
            return requested
