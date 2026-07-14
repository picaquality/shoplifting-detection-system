import unittest

from shoplifting_system.config import parse_camera_source, parse_optional_camera_source, validate_config_payload


class ConfigValidationTests(unittest.TestCase):
    def test_parse_camera_source_numeric(self):
        self.assertEqual(parse_camera_source("2"), 2)

    def test_parse_camera_source_rtsp(self):
        source = "rtsp://10.0.0.1/live"
        self.assertEqual(parse_camera_source(source), source)

    def test_parse_optional_camera_source_empty(self):
        self.assertIsNone(parse_optional_camera_source(""))

    def test_validate_payload_success(self):
        clean, errors = validate_config_payload(
            {
                "camera_source": "1",
                "fallback_camera_source": "2",
                "confidence_threshold": 0.65,
                "frame_skip": 4,
                "sensitivity_profile": "strict",
                "inference_width": 1280,
                "snapshot_cooldown_sec": 4,
                "detection_cooldown_sec": 2,
                "snapshot_retention_max": 120,
            }
        )
        self.assertFalse(errors)
        self.assertEqual(clean["camera_source"], 1)
        self.assertEqual(clean["fallback_camera_source"], 2)
        self.assertEqual(clean["frame_skip"], 4)
        self.assertEqual(clean["sensitivity_profile"], "strict")

    def test_validate_payload_rejects_bad_values(self):
        _, errors = validate_config_payload(
            {
                "camera_source": "",
                "confidence_threshold": 2.0,
                "frame_skip": 0,
                "sensitivity_profile": "aggressive",
                "inference_width": 100,
                "snapshot_cooldown_sec": 0,
                "detection_cooldown_sec": 100,
                "snapshot_retention_max": 5,
            }
        )
        self.assertIn("camera_source", errors)
        self.assertIn("confidence_threshold", errors)
        self.assertIn("frame_skip", errors)
        self.assertIn("sensitivity_profile", errors)
        self.assertIn("inference_width", errors)
        self.assertIn("snapshot_cooldown_sec", errors)
        self.assertIn("detection_cooldown_sec", errors)
        self.assertIn("snapshot_retention_max", errors)


if __name__ == "__main__":
    unittest.main()
