import unittest

from shoplifting_system.detection import (
    DetectionEventTracker,
    confidence_band,
    detection_fingerprint,
    should_emit_detection,
)


class DetectionLogicTests(unittest.TestCase):
    def test_detection_fingerprint_stable(self):
        detections = [
            {"label": "shoplifting", "confidence": 0.91, "x": 10.04, "y": 11.06},
            {"label": "shoplifting", "confidence": 0.82, "x": 15.01, "y": 8.99},
        ]
        fp_1 = detection_fingerprint(detections)
        fp_2 = detection_fingerprint(list(reversed(detections)))
        self.assertEqual(fp_1, fp_2)

    def test_emit_when_changed(self):
        emit, fp = should_emit_detection(
            detections=[{"label": "shoplifting", "confidence": 0.9, "x": 1, "y": 1}],
            last_fingerprint="old",
            last_emitted_at=100.0,
            now=101.0,
            cooldown_seconds=5.0,
        )
        self.assertTrue(emit)
        self.assertTrue(fp)

    def test_do_not_emit_without_change_before_cooldown(self):
        detections = [{"label": "shoplifting", "confidence": 0.9, "x": 1, "y": 1}]
        same_fp = detection_fingerprint(detections)
        emit, _ = should_emit_detection(
            detections=detections,
            last_fingerprint=same_fp,
            last_emitted_at=100.0,
            now=101.0,
            cooldown_seconds=5.0,
        )
        self.assertFalse(emit)

    def test_multi_frame_confirmation_requires_threshold(self):
        tracker = DetectionEventTracker(
            required_confirmations=2,
            event_hold_seconds=1.0,
            active_event_interval_seconds=1.0,
            detection_cooldown_sec=0.0,
        )
        first = tracker.update([{"label": "shoplifting", "confidence": 0.8}], now=1.0)
        second = tracker.update([{"label": "shoplifting", "confidence": 0.8}], now=1.1)
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.event_type, "start")

    def test_event_window_emits_end_after_hold(self):
        tracker = DetectionEventTracker(
            required_confirmations=1,
            event_hold_seconds=0.5,
            active_event_interval_seconds=1.0,
            detection_cooldown_sec=0.0,
        )
        start = tracker.update([{"label": "shoplifting", "confidence": 0.95}], now=1.0)
        no_end = tracker.update([], now=1.2)
        end = tracker.update([], now=1.6)
        self.assertEqual(start.event_type, "start")
        self.assertIsNone(no_end)
        self.assertEqual(end.event_type, "end")

    def test_confidence_band(self):
        self.assertEqual(confidence_band(0.9), "high")
        self.assertEqual(confidence_band(0.7), "medium")
        self.assertEqual(confidence_band(0.4), "low")


if __name__ == "__main__":
    unittest.main()
