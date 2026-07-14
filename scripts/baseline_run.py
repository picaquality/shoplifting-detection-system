#!/usr/bin/env python3
"""Capture a short baseline of stream performance metrics."""

from __future__ import annotations

import argparse
import json
import time


def parse_source(raw: str):
    return int(raw) if raw.isdigit() else raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short camera baseline test.")
    parser.add_argument("--source", default="0", help="Camera source ID or RTSP URL")
    parser.add_argument("--seconds", type=int, default=15, help="Duration of baseline run")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv-python-headless is required to run baseline checks")
        return 2

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(json.dumps({"success": False, "error": f"Could not open source: {source}"}, indent=2))
        return 1

    start_wall = time.time()
    start_cpu = time.process_time()
    frames = 0
    reads_failed = 0

    while time.time() - start_wall < args.seconds:
        ok, _ = cap.read()
        if ok:
            frames += 1
        else:
            reads_failed += 1
        time.sleep(0.001)

    cap.release()

    elapsed = max(0.001, time.time() - start_wall)
    fps = frames / elapsed
    cpu_percent_est = max(0.0, min(100.0, (time.process_time() - start_cpu) / elapsed * 100.0))

    output = {
        "success": True,
        "source": str(source),
        "duration_seconds": round(elapsed, 2),
        "frames_read": frames,
        "read_failures": reads_failed,
        "fps": round(fps, 2),
        "cpu_percent_estimate": round(cpu_percent_est, 2),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
