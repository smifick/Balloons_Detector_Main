"""Exclude fixed camera hardware from detections using calibrated image regions."""
import json
from pathlib import Path


def load_regions(config_path, camera_index, frame_shape):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera exclusion config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    camera = config.get(str(camera_index))
    if camera is None:
        return []
    width, height = camera["reference_size"]
    actual_h, actual_w = frame_shape[:2]
    if abs((actual_w / actual_h) / (width / height) - 1) > 0.01:
        raise ValueError("Camera aspect ratio changed; recalibrate camera_exclusions.json before using this source.")
    regions = camera["regions"]
    for region in regions:
        x1, y1, x2, y2 = region["rect"]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Invalid exclusion rectangle: {region['name']}")
        if not 0 < region["min_box_overlap"] <= 1:
            raise ValueError(f"Invalid exclusion overlap threshold: {region['name']}")
    return regions


def is_excluded(xyxy, frame_shape, regions):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    area = max(0, x2-x1) * max(0, y2-y1)
    if area <= 0:
        return False
    for region in regions:
        a, b, c, d = region["rect"]
        intersection = max(0, min(x2, c*width)-max(x1, a*width)) * max(0, min(y2, d*height)-max(y1, b*height))
        if intersection / area >= region["min_box_overlap"]:
            return True
    return False


def filter_result(result, regions):
    """Update an Ultralytics result so excluded boxes never enter downstream output."""
    if not regions or result.boxes is None or len(result.boxes) == 0:
        return 0
    keep = [not is_excluded(box.xyxy[0].tolist(), result.orig_shape, regions)
            for box in result.boxes]
    removed = len(keep) - sum(keep)
    result.update(boxes=result.boxes.data[keep])
    return removed
