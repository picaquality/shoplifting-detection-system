from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def detection_fingerprint(detections: Iterable[Dict[str, float | str]]) -> str:
    parts: List[str] = []
    for det in detections:
        label = str(det.get("label", "unknown"))
        confidence = float(det.get("confidence", 0.0))
        x = round(float(det.get("x", 0.0)), 1)
        y = round(float(det.get("y", 0.0)), 1)
        parts.append(f"{label}:{confidence:.2f}@{x},{y}")
    return "|".join(sorted(parts))


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
