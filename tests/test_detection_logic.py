import unittest

from shoplifting_system.detection import detection_fingerprint, should_emit_detection


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


if __name__ == "__main__":
    unittest.main()
