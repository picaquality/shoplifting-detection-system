import unittest

from shoplifting_system.app_factory import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_status_endpoint(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("detections", payload)
        self.assertIn("metrics", payload)

    def test_config_endpoint_rejects_invalid_payload(self):
        response = self.client.post(
            "/api/config",
            json={"camera_source": "", "confidence_threshold": 3.0, "frame_skip": 0},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("errors", payload)

    def test_config_endpoint_updates(self):
        response = self.client.post(
            "/api/config",
            json={"camera_source": "1", "confidence_threshold": 0.7, "frame_skip": 2},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["applied"]["camera_source"], 1)


if __name__ == "__main__":
    unittest.main()
