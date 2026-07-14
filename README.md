# Shoplifting Detection System

A Flask-based real-time monitoring app for camera streams with Roboflow local inference.

## Performance Targets

The app now exposes measurable operational targets via `/api/targets`:

- **Latency (p95 inference):** <= 180ms
- **Effective FPS:** >= 12
- **CPU usage:** <= 75%
- **False alert rate:** <= 3 per hour
- **Stream uptime:** >= 99%

## Baseline Run

Capture a short baseline before tuning:

```bash
python /home/runner/work/shoplifting-detection-system/shoplifting-detection-system/scripts/baseline_run.py --source 0 --seconds 15
```

This records quick baseline FPS, read failures, and estimated CPU usage.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `ROBOFLOW_API_KEY` | empty | Roboflow API key for loading local model |
| `MODEL_ID` | `shoplifting-detection/1` | Roboflow model id |
| `CAMERA_SOURCE` | `0` | Camera device id or RTSP URL |
| `CONFIDENCE_THRESHOLD` | `0.40` | Detection confidence threshold (0.05-1.0) |
| `PROCESSING_FRAME_SKIP` | `3` | Base frame skip for inference (1-30) |
| `INFERENCE_WIDTH` | `960` | Resize width for inference acceleration |
| `SNAPSHOT_COOLDOWN_SECONDS` | `3` | Minimum seconds between snapshots |
| `DETECTION_COOLDOWN_SECONDS` | `3` | Minimum seconds between duplicate alert events |
| `SNAPSHOT_RETENTION_MAX` | `200` | Maximum stored snapshots before pruning |
| `CAMERA_RECONNECT_DELAY_SECONDS` | `1.0` | Initial reconnect delay |
| `CAMERA_RECONNECT_MAX_BACKOFF_SECONDS` | `10.0` | Maximum reconnect delay |
| `STREAM_ERROR_FRAME_DELAY_SECONDS` | `0.4` | Delay between empty frame retries |
| `LOG_LEVEL` | `INFO` | Logging level |
| `HOST` | `0.0.0.0` | Flask host |
| `PORT` | `5000` | Flask port |

## Running

```bash
python /home/runner/work/shoplifting-detection-system/shoplifting-detection-system/app.py
```

## API Summary

- `GET /api/status` - stream health, latest detections, runtime metrics
- `POST /api/config` - validated config updates with clear errors
- `GET /api/metrics` - metrics plus performance targets
- `GET /api/targets` - target values only

## Troubleshooting

- If detections are empty, verify `ROBOFLOW_API_KEY` and `MODEL_ID`.
- If stream is degraded, verify `CAMERA_SOURCE` reachability.
- If snapshot folder grows too quickly, decrease sensitivity or increase cooldown.
