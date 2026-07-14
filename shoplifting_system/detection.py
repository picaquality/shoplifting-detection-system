from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def detection_fingerprint(detections: Iterable[Dict[str, float | str]]) -> str:
    parts: List[str] = []
    for det in detections:
        label = str(det.get("label", "unknown"))
        confidence = float(det.get("confidence", 0.0))
        x = round(float(det.get("x", 0.0)), 1)
        y = round(float(det.get("y", 0.0)), 1)
        parts.append(f"{label}:{confidence:.2f}@{x},{y}")
    return "|".join(sorted(parts))


def detection_signature(detections: Iterable[Dict[str, float | str]]) -> str:
    labels = sorted(str(det.get("label", "unknown")) for det in detections)
    return "|".join(labels)


def confidence_band(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.9:
        return "critical"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


@dataclass
class DetectionEvent:
    event_type: str
    event_id: int
    confidence: float
    confidence_band: str
    severity: str


class DetectionEventTracker:
    def __init__(
        self,
        required_confirmations: int,
        event_hold_seconds: float,
        active_event_interval_seconds: float,
        detection_cooldown_sec: float,
    ):
        self.required_confirmations = max(1, int(required_confirmations))
        self.event_hold_seconds = max(0.2, float(event_hold_seconds))
        self.active_event_interval_seconds = max(0.2, float(active_event_interval_seconds))
        self.detection_cooldown_sec = max(0.0, float(detection_cooldown_sec))

        self.pending_confirmation_count = 0
        self.pending_signature = ""
        self.pending_confidence = 0.0

        self.active = False
        self.current_event_id = 0
        self.active_until = 0.0
        self.last_active_emit = 0.0
        self.last_event_closed_at = 0.0

    def _start_event(self, now: float, confidence: float) -> DetectionEvent:
        self.current_event_id += 1
        self.active = True
        self.active_until = now + self.event_hold_seconds
        self.last_active_emit = now
        return DetectionEvent(
            event_type="start",
            event_id=self.current_event_id,
            confidence=confidence,
            confidence_band=confidence_band(confidence),
            severity=severity_from_confidence(confidence),
        )

    def _active_event(self, now: float, confidence: float) -> DetectionEvent:
        self.last_active_emit = now
        return DetectionEvent(
            event_type="active",
            event_id=self.current_event_id,
            confidence=confidence,
            confidence_band=confidence_band(confidence),
            severity=severity_from_confidence(confidence),
        )

    def _end_event(self) -> DetectionEvent:
        return DetectionEvent(
            event_type="end",
            event_id=self.current_event_id,
            confidence=0.0,
            confidence_band="low",
            severity="low",
        )

    def update(self, detections: List[Dict[str, float | str]], now: float) -> Optional[DetectionEvent]:
        if detections:
            max_confidence = max(float(det.get("confidence", 0.0)) for det in detections)
            signature = detection_signature(detections)

            if not self.active:
                if signature == self.pending_signature:
                    self.pending_confirmation_count += 1
                    self.pending_confidence = max(self.pending_confidence, max_confidence)
                else:
                    self.pending_signature = signature
                    self.pending_confirmation_count = 1
                    self.pending_confidence = max_confidence

                cooldown_ok = (now - self.last_event_closed_at) >= self.detection_cooldown_sec
                if self.pending_confirmation_count >= self.required_confirmations and cooldown_ok:
                    event = self._start_event(now, self.pending_confidence)
                    self.pending_confirmation_count = 0
                    self.pending_signature = ""
                    self.pending_confidence = 0.0
                    return event
                return None

            self.active_until = now + self.event_hold_seconds
            if (now - self.last_active_emit) >= self.active_event_interval_seconds:
                return self._active_event(now, max_confidence)
            return None

        self.pending_confirmation_count = 0
        self.pending_signature = ""
        self.pending_confidence = 0.0

        if self.active and now >= self.active_until:
            self.active = False
            self.last_event_closed_at = now
            return self._end_event()

        return None


def should_emit_detection(
    detections: List[Dict[str, float | str]],
    last_fingerprint: str,
    last_emitted_at: float,
    now: float,
    cooldown_seconds: float,
) -> Tuple[bool, str]:
    if not detections:
        return False, ""

    new_fingerprint = detection_fingerprint(detections)
    changed = new_fingerprint != last_fingerprint
    cooldown_elapsed = (now - last_emitted_at) >= cooldown_seconds

    return changed or cooldown_elapsed, new_fingerprint
