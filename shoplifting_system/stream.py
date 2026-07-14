from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from .detection import should_emit_detection
from .state import SharedState

try:
    import cv2
except ImportError:  # pragma: no cover - validated by runtime environment
    cv2 = None


class CameraStreamProcessor:
    def __init__(
        self,
        state: SharedState,
        inference_engine,
        snapshot_dir: Path,
        snapshot_retention_max: int,
        snapshot_cooldown_sec: float,
        detection_cooldown_sec: float,
        reconnect_delay_sec: float,
        reconnect_backoff_max_sec: float,
        stream_error_frame_delay_sec: float,
    ):
        self.state = state
        self.inference_engine = inference_engine
        self.snapshot_dir = snapshot_dir
        self.snapshot_retention_max = snapshot_retention_max
        self.snapshot_cooldown_sec = snapshot_cooldown_sec
        self.detection_cooldown_sec = detection_cooldown_sec
        self.reconnect_delay_sec = reconnect_delay_sec
        self.reconnect_backoff_max_sec = reconnect_backoff_max_sec
        self.stream_error_frame_delay_sec = stream_error_frame_delay_sec
        self.logger = logging.getLogger(__name__)

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def request_restart(self) -> None:
        self.state.mark_restart_requested()

    def _open_capture(self, source):
        if cv2 is None:
            return None
        return cv2.VideoCapture(source)

    def _cleanup_old_snapshots(self) -> None:
        files = sorted(self.snapshot_dir.glob("snapshot_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_file in files[self.snapshot_retention_max :]:
            try:
                old_file.unlink(missing_ok=True)
            except OSError:
                self.logger.warning("Failed to remove old snapshot %s", old_file)

    def _save_snapshot_if_needed(self, frame, now: float, should_snapshot: bool) -> Optional[str]:
        if not should_snapshot or cv2 is None:
            return None

        if now - self.state.last_snapshot_time < self.snapshot_cooldown_sec:
            return None

        filename = f"snapshot_{int(now)}.jpg"
        filepath = self.snapshot_dir / filename
        cv2.imwrite(str(filepath), frame)
        self._cleanup_old_snapshots()
        with self.state.lock():
            self.state.last_snapshot_time = now
            self.state.latest_snapshot = f"/static/snapshots/{filename}"
        return self.state.latest_snapshot

    def _draw_detections(self, frame):
        if cv2 is None:
            return
        with self.state.lock():
            detections = list(self.state.latest_detections)

        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            label = det["label"]
            confidence = det["confidence"]
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (int(x - w / 2), int(y - h / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    def _update_adaptive_skip(self, inference_ms: float) -> None:
        with self.state.lock():
            old_ema = self.state.metrics.inference_ms_ema
            new_ema = inference_ms if old_ema <= 0 else (0.85 * old_ema + 0.15 * inference_ms)
            self.state.metrics.inference_ms_ema = new_ema

            base_skip = self.state.base_frame_skip
            if new_ema > 240:
                self.state.adaptive_frame_skip = min(30, base_skip + 2)
            elif new_ema > 180:
                self.state.adaptive_frame_skip = min(30, base_skip + 1)
            elif new_ema < 120:
                self.state.adaptive_frame_skip = max(1, base_skip - 1)
            else:
                self.state.adaptive_frame_skip = base_skip

    def _update_effective_fps(self, frame_ts: float, previous_ts: float) -> float:
        elapsed = max(0.00001, frame_ts - previous_ts)
        fps = 1.0 / elapsed
        with self.state.lock():
            old_ema = self.state.metrics.effective_fps_ema
            self.state.metrics.effective_fps_ema = fps if old_ema <= 0 else (0.9 * old_ema + 0.1 * fps)
        return frame_ts

    def generate_frames(self):
        if cv2 is None:
            self.state.set_connection_state("error", "opencv-python-headless is not installed")
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
                time.sleep(self.stream_error_frame_delay_sec)

        current_source = None
        cap = None
        frame_count = 0
        last_frame_ts = time.time()
        reconnect_delay = self.reconnect_delay_sec

        while True:
            with self.state.lock():
                desired_source = self.state.camera_source

            if cap is None or current_source != desired_source or self.state.consume_restart_request():
                if cap is not None and cap.isOpened():
                    cap.release()
                current_source = desired_source
                cap = self._open_capture(current_source)

                if not cap or not cap.isOpened():
                    with self.state.lock():
                        self.state.metrics.reconnect_count += 1
                    self.state.set_connection_state("degraded", f"Could not open camera source: {current_source}")
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(self.reconnect_backoff_max_sec, reconnect_delay * 1.5)
                    continue

                reconnect_delay = self.reconnect_delay_sec
                self.state.set_connection_state("active", None)

            success, frame = cap.read()
            frame_now = time.time()
            last_frame_ts = self._update_effective_fps(frame_now, last_frame_ts)

            if not success:
                with self.state.lock():
                    self.state.metrics.dropped_frames += 1
                self.state.set_connection_state("degraded", "Camera frame read failed")
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
                time.sleep(self.stream_error_frame_delay_sec)
                continue

            with self.state.lock():
                self.state.metrics.frames_read += 1
                adaptive_skip = self.state.adaptive_frame_skip

            frame_count += 1
            should_process = frame_count % max(1, adaptive_skip) == 0

            if should_process and self.inference_engine.ready:
                result = self.inference_engine.infer(frame, time.time)
                self._update_adaptive_skip(result.inference_ms)

                with self.state.lock():
                    now = time.time()
                    emit, new_fingerprint = should_emit_detection(
                        result.detections,
                        self.state.last_detection_fingerprint,
                        self.state.last_detection_time,
                        now,
                        self.detection_cooldown_sec,
                    )

                    if emit:
                        self.state.latest_detections = result.detections
                        self.state.last_detection_fingerprint = new_fingerprint
                        self.state.last_detection_time = now

                    self.state.metrics.frames_processed += 1
                    detections_exist = bool(self.state.latest_detections)

                self._save_snapshot_if_needed(frame, time.time(), should_snapshot=emit and detections_exist)

            self._draw_detections(frame)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

        if cap is not None and cap.isOpened():
            cap.release()
