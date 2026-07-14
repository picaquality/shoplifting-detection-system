import unittest

from shoplifting_system.config import parse_camera_source, validate_config_payload


class ConfigValidationTests(unittest.TestCase):
    def test_parse_camera_source_numeric(self):
        self.assertEqual(parse_camera_source("2"), 2)

    def test_parse_camera_source_rtsp(self):
        source = "rtsp://10.0.0.1/live"
        self.assertEqual(parse_camera_source(source), source)

    def test_validate_payload_success(self):
        clean, errors = validate_config_payload(
            {
                "camera_source": "1",
                "confidence_threshold": 0.65,
                "frame_skip": 4,
            }
        )
        self.assertFalse(errors)
        self.assertEqual(clean["camera_source"], 1)
        self.assertEqual(clean["frame_skip"], 4)

    def test_validate_payload_rejects_bad_values(self):
        _, errors = validate_config_payload(
            {
                "camera_source": "",
                "confidence_threshold": 2.0,
                "frame_skip": 0,
            }
        )
        self.assertIn("camera_source", errors)
        self.assertIn("confidence_threshold", errors)
        self.assertIn("frame_skip", errors)


if __name__ == "__main__":
    unittest.main()
