"""Compare a video-only center prediction with the original recorded overlay."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import statistics
import subprocess

import cv2
import numpy as np


class ScreenPrediction:
    def __init__(self):
        self.observations = deque(maxlen=4)

    def update(self, track, camera, playback_time):
        if not track or track["center"] is None:
            return
        stamp = track["observed_at"]
        if self.observations and stamp <= self.observations[-1][0]:
            return
        age = (camera["frame_age_ms"] + camera["inference_ms"]) / 1000
        if self.observations and stamp - self.observations[-1][0] > 0.22:
            self.observations.clear()
        self.observations.append((stamp, playback_time - age, np.array(track["center"])))

    def point(self, playback_time):
        if not self.observations:
            return None
        stamp, image_time, center = self.observations[-1]
        horizon = max(0.0, playback_time - image_time)
        if horizon > 0.22:
            return None
        if len(self.observations) == 1:
            return center
        times = np.array([item[0] - stamp for item in self.observations])
        times -= times.mean()
        points = np.stack([item[2] for item in self.observations])
        velocity = times @ points / (times @ times)
        return center + velocity * horizon


def orange_outline(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (2, 65, 95), (30, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea, default=None)
    if contour is None or cv2.contourArea(contour) < 100:
        return None, None
    moments = cv2.moments(contour)
    reference = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    return contour, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--frames", nargs=2, type=int, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--crop", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    first, last = args.frames
    left, top, right, bottom = args.crop
    width, height = right - left, bottom - top
    rows = [json.loads(line) for line in (args.recording / "frames.jsonl").read_text().splitlines()]
    predictor = ScreenPrediction()
    results = {"RAW": [], "PRED": []}
    colors = {"RAW": (30, 130, 255), "PRED": (255, 255, 0)}
    reference_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate", "15",
               "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(args.output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        with encoder.stdin as video:
            for index in range(first, last + 1):
                row = rows[index]
                frame = cv2.imread(str(args.recording / f"{index:04}.jpg"))[top:bottom, left:right].copy()
                contour, reference = orange_outline(frame)
                track = next((t for t in row["tracks"] if t["id"] == args.track), None)
                predictor.update(track, row["camera"], row["elapsed"])
                if contour is not None:
                    reference_count += 1
                    cv2.drawContours(frame, [contour], -1, (105, 235, 105), 1)
                    cv2.drawMarker(frame, tuple(round(v) for v in reference), (105, 235, 105),
                                   cv2.MARKER_CROSS, 8, 1)
                for label, point in (("RAW", track["center"] if track else None),
                                     ("PRED", predictor.point(row["elapsed"]))):
                    if point is None:
                        continue
                    xy = (float(point[0] - left), float(point[1] - top))
                    if contour is not None:
                        inside = cv2.pointPolygonTest(contour, xy, False) >= 0
                        results[label].append((inside, math.dist(xy, reference)))
                    cv2.circle(frame, tuple(round(v) for v in xy), 5, colors[label], 2)
                for line, (label, color) in enumerate(colors.items()):
                    cv2.putText(frame, label, (7, height - 34 + 16 * line),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)
                elapsed = row["elapsed"] - rows[first]["elapsed"]
                cv2.putText(frame, f"{elapsed:.2f} s", (width - 72, height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
                video.write(frame.tobytes())
        if encoder.wait():
            raise RuntimeError("Could not encode the review video")
    print(f"Reference outlines: {reference_count}/{last - first + 1}")
    for label, observations in results.items():
        errors = [error for _, error in observations]
        if errors:
            print(f"{label}: {sum(inside for inside, _ in observations)} inside, "
                  f"{len(observations)} visible, median {statistics.median(errors):.2f} px, "
                  f"maximum {max(errors):.2f} px")


if __name__ == "__main__":
    main()
