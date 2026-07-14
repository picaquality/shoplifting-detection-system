from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from inference.models.utils import get_roboflow_model

    HAS_INFERENCE_SDK = True
except ImportError:
    HAS_INFERENCE_SDK = False


@dataclass
class InferenceResult:
    detections: List[Dict[str, float | str]]
    inference_ms: float


class InferenceEngine:
    def __init__(self, model_id: str, api_key: str, confidence_threshold: float, inference_width: int):
        self.model_id = model_id
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.inference_width = inference_width
        self.model = None
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        if not HAS_INFERENCE_SDK:
            self.logger.warning("Inference SDK unavailable; running without detections.")
            return

        if not self.api_key:
            self.logger.warning("ROBOFLOW_API_KEY missing; running without detections.")
            return

        try:
            self.model = get_roboflow_model(model_id=self.model_id, api_key=self.api_key)
            self.logger.info("Loaded inference model: %s", self.model_id)
        except Exception as exc:
            self.logger.exception("Failed to initialize model: %s", exc)
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _resize_for_inference(self, frame):
        try:
            import cv2
        except ImportError:
            return frame, 1.0, 1.0

        height, width = frame.shape[:2]
        if width <= self.inference_width:
            return frame, 1.0, 1.0

        ratio = self.inference_width / float(width)
        new_size = (self.inference_width, int(height * ratio))
        resized = cv2.resize(frame, new_size)
        scale_x = width / float(new_size[0])
        scale_y = height / float(new_size[1])
        return resized, scale_x, scale_y

    def infer(self, frame_bgr, now_fn) -> InferenceResult:
        if not self.model:
            return InferenceResult(detections=[], inference_ms=0.0)

        try:
            import cv2
        except ImportError:
            self.logger.error("OpenCV unavailable; cannot run inference.")
            return InferenceResult(detections=[], inference_ms=0.0)

        resized, scale_x, scale_y = self._resize_for_inference(frame_bgr)
        start = now_fn()

        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self.model.infer(frame_rgb)
        elapsed_ms = (now_fn() - start) * 1000.0

        detections_raw = []
        if isinstance(results, list) and results:
            detections_raw = getattr(results[0], "predictions", [])
        elif isinstance(results, dict):
            detections_raw = results.get("predictions", [])
        else:
            detections_raw = getattr(results, "predictions", [])

        detections: List[Dict[str, float | str]] = []
        for det in detections_raw:
            x, y, w, h, label, confidence = self._extract_prediction(det)
            if confidence < self.confidence_threshold:
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "x": x * scale_x,
                    "y": y * scale_y,
                    "w": w * scale_x,
                    "h": h * scale_y,
                }
            )

        return InferenceResult(detections=detections, inference_ms=elapsed_ms)

    @staticmethod
    def _extract_prediction(det) -> Tuple[float, float, float, float, str, float]:
        x = float(det.x if hasattr(det, "x") else det.get("x", 0.0))
        y = float(det.y if hasattr(det, "y") else det.get("y", 0.0))
        w = float(det.width if hasattr(det, "width") else det.get("width", 0.0))
        h = float(det.height if hasattr(det, "height") else det.get("height", 0.0))
        label = str(det.class_name if hasattr(det, "class_name") else det.get("class", "unknown"))
        confidence = float(det.confidence if hasattr(det, "confidence") else det.get("confidence", 0.0))
        return x, y, w, h, label, confidence
