from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .config import PerformanceTargets, SENSITIVITY_PROFILES
from .detection import DetectionEventTracker
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
        targets: PerformanceTargets,
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
        self.targets = targets
        self.logger = logging.getLogger(__name__)

        self.backpressure_skip_remaining = 0
        self.recovering_success_frames = 0
        self.recovering_required_frames = 12

        self.current_profile_name = state.sensitivity_profile
        profile = SENSITIVITY_PROFILES[self.current_profile_name]
        self.event_tracker = DetectionEventTracker(
            required_confirmations=profile.confirmation_frames,
            event_hold_seconds=profile.event_hold_seconds,
            active_event_interval_seconds=profile.active_event_interval_seconds,
            detection_cooldown_sec=self.detection_cooldown_sec,
        )

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def request_restart(self) -> None:
        self.state.mark_restart_requested()

    def apply_runtime_config(self, updates: dict) -> None:
        if "snapshot_retention_max" in updates:
            self.snapshot_retention_max = int(updates["snapshot_retention_max"])
        if "snapshot_cooldown_sec" in updates:
            self.snapshot_cooldown_sec = float(updates["snapshot_cooldown_sec"])
        if "detection_cooldown_sec" in updates:
            self.detection_cooldown_sec = float(updates["detection_cooldown_sec"])
            self.event_tracker.detection_cooldown_sec = self.detection_cooldown_sec
        if "confidence_threshold" in updates:
            self.inference_engine.confidence_threshold = float(updates["confidence_threshold"])
        if "inference_width" in updates:
            self.inference_engine.inference_width = int(updates["inference_width"])

    def _sync_sensitivity_profile(self) -> None:
        with self.state.lock():
            profile_name = self.state.sensitivity_profile

        if profile_name == self.current_profile_name:
            return

        profile = SENSITIVITY_PROFILES.get(profile_name, SENSITIVITY_PROFILES["balanced"])
        self.event_tracker.required_confirmations = profile.confirmation_frames
        self.event_tracker.event_hold_seconds = profile.event_hold_seconds
        self.event_tracker.active_event_interval_seconds = profile.active_event_interval_seconds
        self.current_profile_name = profile.name

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
            fps_ema = self.state.metrics.effective_fps_ema

            latency_breach = new_ema > (self.targets.latency_ms_p95 * 1.2)
            fps_breach = fps_ema > 0 and fps_ema < (self.targets.min_effective_fps * 0.8)

            if latency_breach or fps_breach:
                self.state.adaptive_frame_skip = min(30, max(base_skip, self.state.adaptive_frame_skip + 1))
            elif new_ema < (self.targets.latency_ms_p95 * 0.75) and fps_ema >= self.targets.min_effective_fps:
                self.state.adaptive_frame_skip = max(1, self.state.adaptive_frame_skip - 1)
            else:
                self.state.adaptive_frame_skip = max(1, base_skip)

    def _update_effective_fps(self, frame_ts: float, previous_ts: float) -> float:
        elapsed = max(0.00001, frame_ts - previous_ts)
        fps = 1.0 / elapsed
        with self.state.lock():
            old_ema = self.state.metrics.effective_fps_ema
            self.state.metrics.effective_fps_ema = fps if old_ema <= 0 else (0.9 * old_ema + 0.1 * fps)
        return frame_ts

    def _handle_backpressure(self, elapsed_ms: float) -> None:
        if elapsed_ms <= self.targets.max_processing_lag_ms:
            return
        with self.state.lock():
            self.state.metrics.backpressure_events += 1
        self.backpressure_skip_remaining = min(10, self.backpressure_skip_remaining + 2)

    def _try_failover(self, current_source) -> bool:
        with self.state.lock():
            fallback_source = self.state.fallback_camera_source

        if fallback_source is None or fallback_source == current_source:
            return False

        with self.state.lock():
            self.state.camera_source = fallback_source
        self.state.set_connection_state("recovering", f"Switching to fallback source: {fallback_source}")
        self.logger.warning("Switching to fallback camera source from %s to %s", current_source, fallback_source)
        return True

    def _mark_success_frame_health(self) -> None:
        with self.state.lock():
            current_status = self.state.connection_status

        if current_status == "recovering":
            self.recovering_success_frames += 1
            if self.recovering_success_frames >= self.recovering_required_frames:
                self.state.set_connection_state("healthy", None)
        elif current_status == "degraded":
            self.recovering_success_frames = 1
            self.state.set_connection_state("recovering", None)
        else:
            self.recovering_success_frames = min(self.recovering_required_frames, self.recovering_success_frames + 1)

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
        source_open_failures = 0

        while True:
            self._sync_sensitivity_profile()
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
                    source_open_failures += 1
                    if source_open_failures >= 3 and self._try_failover(current_source):
                        source_open_failures = 0
                        cap = None
                        continue

                    self.state.set_connection_state("degraded", f"Could not open camera source: {current_source}")
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(self.reconnect_backoff_max_sec, reconnect_delay * 1.5)
                    continue

                source_open_failures = 0
                self.recovering_success_frames = 0
                reconnect_delay = self.reconnect_delay_sec
                self.state.set_connection_state("recovering", None)

            success, frame = cap.read()
            frame_now = time.time()
            last_frame_ts = self._update_effective_fps(frame_now, last_frame_ts)

            if not success:
                with self.state.lock():
                    self.state.metrics.dropped_frames += 1
                self.recovering_success_frames = 0
                self.state.set_connection_state("degraded", "Camera frame read failed")
                if self._try_failover(current_source):
                    cap = None
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
                time.sleep(self.stream_error_frame_delay_sec)
                continue

            self._mark_success_frame_health()

            with self.state.lock():
                self.state.metrics.frames_read += 1
                adaptive_skip = self.state.adaptive_frame_skip

            frame_count += 1
            should_process = frame_count % max(1, adaptive_skip) == 0
            if self.backpressure_skip_remaining > 0:
                should_process = False
                self.backpressure_skip_remaining -= 1

            event = None
            detections_exist = False
            loop_start = time.time()

            if should_process and self.inference_engine.ready:
                result = self.inference_engine.infer(frame, time.time)
                self._update_adaptive_skip(result.inference_ms)

                now = time.time()
                event = self.event_tracker.update(result.detections, now)
                with self.state.lock():
                    self.state.metrics.frames_processed += 1
                    self.state.latest_detections = result.detections
                    detections_exist = bool(result.detections)

                if event:
                    self.state.record_alert_event(
                        event_type=event.event_type,
                        event_id=event.event_id,
                        severity=event.severity,
                        confidence_band=event.confidence_band,
                        confidence=event.confidence,
                        now=now,
                    )

            self._save_snapshot_if_needed(
                frame,
                time.time(),
                should_snapshot=bool(event and event.event_type == "start" and detections_exist),
            )
            self._draw_detections(frame)
            self._handle_backpressure((time.time() - loop_start) * 1000.0)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

        if cap is not None and cap.isOpened():
            cap.release()
