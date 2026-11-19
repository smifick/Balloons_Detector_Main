"""Conservative object association and image-error to angular-speed control."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def aim_point(dimensions, settings):
    return (dimensions[0] * (0.5 + settings.aim_x_percent / 100),
            dimensions[1] * (0.5 + settings.aim_y_percent / 100))


def area(box):
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / (area(a) + area(b) - intersection + 1e-9)


@dataclass
class Track:
    id: int
    box: list[float]
    confidence: float
    seen_at: float
    streak: int = 1
    visible: bool = True
    vx: float = 0
    vy: float = 0

    def public(self):
        return {"id": self.id, "box": self.box[:], "confidence": self.confidence,
                "center": list(center(self.box)), "streak": self.streak,
                "visible": self.visible, "observed_at": self.seen_at}


class Tracker:
    def __init__(self, confirmations=3, max_age=0.6):
        self.confirmations = confirmations
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def clear(self):
        self.tracks.clear()

    def update(self, detections, now):
        self.tracks = {key: track for key, track in self.tracks.items() if now - track.seen_at <= self.max_age}
        candidates = []
        for key, track in self.tracks.items():
            dt = min(now - track.seen_at, 0.2)
            predicted = [track.box[0] + track.vx * dt, track.box[1] + track.vy * dt,
                         track.box[2] + track.vx * dt, track.box[3] + track.vy * dt]
            px, py = center(predicted)
            diagonal = math.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1])
            for i, detection in enumerate(detections):
                box = detection["box"]
                ratio = area(box) / area(track.box)
                if not 0.4 <= ratio <= 2.5:
                    continue
                x, y = center(box)
                distance = math.hypot(x - px, y - py)
                gate = max(20, diagonal * 1.1)
                iou = overlap(predicted, box)
                if iou >= 0.1 or distance <= gate:
                    candidates.append((iou + max(0, 1 - distance / gate) * 0.5, key, i))
        used_tracks, used_detections = set(), set()
        for score, key, i in sorted(candidates, reverse=True):
            if key in used_tracks or i in used_detections:
                continue
            track = self.tracks[key]
            box = detections[i]["box"]
            old_x, old_y = center(track.box)
            x, y = center(box)
            dt = max(0.001, now - track.seen_at)
            track.vx = 0.5 * track.vx + 0.5 * (x - old_x) / dt
            track.vy = 0.5 * track.vy + 0.5 * (y - old_y) / dt
            track.streak = track.streak + 1 if track.visible else 1
            track.box, track.confidence = list(box), detections[i]["confidence"]
            track.seen_at, track.visible = now, True
            used_tracks.add(key)
            used_detections.add(i)
        for key, track in self.tracks.items():
            if key not in used_tracks:
                track.visible, track.streak = False, 0
        for i, detection in enumerate(detections):
            if i not in used_detections:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = Track(key, list(detection["box"]), detection["confidence"], now)
        return self.confirmed(now)

    def confirmed(self, now):
        return [track.public() for track in self.tracks.values()
                if track.visible and track.streak >= self.confirmations and now - track.seen_at < 0.2]


@dataclass
class MotionSettings:
    confidence: float = 0.35
    max_speed: float = 6.0
    gain_yaw: float = 8.0
    gain_pitch: float = 8.0
    deadband_px: float = 7.0
    acceleration: float = 25.0
    yaw_limit: float = 45.0
    pitch_limit: float = 30.0
    invert_yaw: bool = False
    invert_pitch: bool = False
    aim_x_percent: float = 0.0
    aim_y_percent: float = 0.0
    braking_lookahead: float = 0.28

    def public(self):
        return asdict(self)


class MotionControl:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pitch_speed = self.yaw_speed = 0.0
        self.limited_axes = []
        self.braking_axes = []
        self.last_observation = None
        self.image_velocity = [0.0, 0.0]

    def compute(self, target, dimensions, offsets, settings, dt):
        self.limited_axes = []
        width, height = dimensions
        x, y = center(target["box"])
        ax, ay = aim_point(dimensions, settings)
        ex, ey = x - ax, y - ay
        stamp = target.get("observed_at")
        identity = (target.get("id"), tuple(dimensions))
        if stamp is not None:
            previous = self.last_observation
            if previous and previous[0] == identity and 0.005 <= stamp - previous[1] <= 0.2:
                sample_dt = stamp - previous[1]
                alpha = 1 - math.exp(-sample_dt / 0.1)
                for i, error in enumerate((ex, ey)):
                    measured = (error - previous[2][i]) / sample_dt
                    measured = max(-3*max(dimensions), min(3*max(dimensions), measured))
                    self.image_velocity[i] += alpha * (measured - self.image_velocity[i])
            elif not previous or previous[0] != identity or stamp - previous[1] > 0.2:
                self.image_velocity = [0.0, 0.0]
            if not previous or stamp > previous[1] or previous[0] != identity:
                self.last_observation = (identity, stamp, (ex, ey))
        self.braking_axes = []

        def approach_speed(error, half_frame, gain, velocity, axis):
            # Predict closing motion only. Prediction can brake to zero, never
            # reverse before the measured target has actually crossed the aim.
            distance_to_aim = abs(error)
            if error * velocity < 0:
                distance_to_aim = max(0.0, distance_to_aim - abs(velocity)*settings.braking_lookahead)
                if distance_to_aim < abs(error) and abs(error) > settings.deadband_px:
                    self.braking_axes.append(axis)
            # Reach the selected speed away from the target; taper only near
            # the deadband. Gain controls the width of this braking region.
            distance = max(0.0, distance_to_aim - settings.deadband_px)
            return math.copysign(settings.max_speed * math.tanh(gain * distance / half_frame), error)

        yaw = approach_speed(ex, width / 2, settings.gain_yaw, self.image_velocity[0], "Yaw")
        pitch = -approach_speed(ey, height / 2, settings.gain_pitch, self.image_velocity[1], "Pitch")
        yaw *= -1 if settings.invert_yaw else 1
        pitch *= -1 if settings.invert_pitch else 1
        maximum_delta = settings.acceleration * min(max(dt, 0), 0.05)
        def limited(desired, previous, offset, bound, axis):
            desired = max(-settings.max_speed, min(settings.max_speed, desired))
            # Acceleration limits speed-up only. Ramping a stop/reversal with
            # the same low limit keeps turning after the target was crossed.
            # Stop first on reversal; accelerate in the new direction next tick.
            if desired * previous < 0:
                value = 0.0
            elif abs(desired) < abs(previous):
                value = desired
            else:
                value = previous + max(-maximum_delta, min(maximum_delta, desired - previous))
            before_limit = value
            # Look ahead beyond the telemetry interval so braking begins inside the limit.
            if value > 0:
                value = min(value, max(0, (bound - offset) / 0.3))
            elif value < 0:
                value = max(value, min(0, (-bound - offset) / 0.3))
            if abs(value - before_limit) > 1e-6:
                self.limited_axes.append(axis)
            return value
        self.pitch_speed = limited(pitch, self.pitch_speed, offsets[0], settings.pitch_limit, "Pitch")
        self.yaw_speed = limited(yaw, self.yaw_speed, offsets[1], settings.yaw_limit, "Yaw")
        return self.pitch_speed, self.yaw_speed, ex, ey
