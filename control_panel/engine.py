"""Camera processing and a single serialized owner of gimbal commands."""
from __future__ import annotations

import base64
import copy
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from zoneinfo import ZoneInfo

from .sbgc import SimpleBGC, SimulatedBGC
from .tracking import MotionControl, MotionSettings, Tracker, aim_point

ROOT = Path(__file__).resolve().parents[1]


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.settings = MotionSettings()
        settings_path = ROOT / "panel_settings.json"
        if settings_path.is_file():
            # Saved settings are validated by the same schema as incoming API data.
            try:
                from .schemas import SettingsInput
                values = SettingsInput.model_validate_json(settings_path.read_text(encoding="utf-8"))
                self.settings = MotionSettings(**values.model_dump())
            except Exception:
                pass
        self.tracker = Tracker()
        self.motion = MotionControl()
        self.last_target_motion = None
        self.board = None
        self.board_info = None
        self.board_error = None
        self.angles = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.home = {"pitch": 0.0, "yaw": 0.0}
        self.yaw_unwrapped = 0.0
        self.yaw_previous = None
        self.telemetry_at = 0.0
        self.last_poll = 0.0
        self.last_motor_poll = 0.0
        self.last_config = 0.0
        self.selected_id = None
        self.active = False
        self.owner = None
        self.heartbeat_at = 0.0
        self.jog_until = 0.0
        self.jog_rates = (0.0, 0.0)
        self.command = {"pitch": 0.0, "yaw": 0.0}
        self.error_px = {"x": 0.0, "y": 0.0}
        self.reason = "Выберите шарик на видео"
        self.camera_status = "off"
        self.camera_error = None
        self.camera_source = 0
        self.camera_generation = 0
        self.camera_thread = None
        self.frame_at = 0.0
        self.frame_number = 0
        self.dimensions = (640, 480)
        self.frame_b64 = None
        self.fps = 0.0
        self.inference_ms = 0.0
        self.device = "—"
        self.events = deque(maxlen=40)
        self.log("Панель готова. Управление остановлено.")
        self.control_thread = threading.Thread(target=self._control_loop, name="gimbal-control", daemon=True)

    def log(self, message, level="info"):
        with self.lock:
            self.events.appendleft({"time": datetime.now(ZoneInfo("Europe/Moscow")).strftime("%H:%M:%S"),
                                    "message": message, "level": level})

    def start(self):
        self.control_thread.start()

    def _read_telemetry_locked(self):
        angles = self.board.read_angles()
        if self.yaw_previous is None:
            self.yaw_unwrapped = angles.yaw
        else:
            delta = (angles.yaw - self.yaw_previous + 180) % 360 - 180
            self.yaw_unwrapped += delta
        self.yaw_previous = angles.yaw
        self.angles = {"roll": angles.roll, "pitch": angles.pitch, "yaw": self.yaw_unwrapped}
        self.telemetry_at = angles.received_at
        self.last_poll = time.monotonic()

    def connect(self, port="", baud=115200, simulated=False, protocol=2):
        with self.lock:
            if self.board:
                raise ValueError("Сначала отключите текущий контроллер")
            self._stop_locked("Подключение контроллера")
            try:
                candidate = SimulatedBGC() if simulated else SimpleBGC(port, baud, protocol)
                self.board = candidate
                self.yaw_previous = None
                self._read_telemetry_locked()
                self.board.read_motor_state()
                self.last_motor_poll = time.monotonic()
                self.home = {"pitch": self.angles["pitch"], "yaw": self.angles["yaw"]}
                self.board_info = candidate.info.copy()
                self.board_error = None
                self.last_config = 0
                self.reason = "Контроллер подключён. Выберите цель."
                self.log(f"{'Симулятор' if simulated else port}: {candidate.info['firmware']}")
            except Exception:
                if self.board:
                    self.board.close()
                self.board = None
                raise

    def disconnect(self):
        with self.lock:
            self._stop_locked("Контроллер отключён")
            if self.board:
                try:
                    self.board.close()
                except Exception as error:
                    self.board_error = str(error)
                    self.log(f"Отключение контроллера: {error}", "warning")
            self.board = None
            self.board_info = None
            self.telemetry_at = 0
            self.log("Контроллер отключён")

    def _stop_locked(self, reason):
        self.active = False
        self.jog_until = 0
        self.owner = None
        self.motion.reset()
        self.last_target_motion = None
        self.command = {"pitch": 0.0, "yaw": 0.0}
        self.reason = reason
        if self.board:
            try:
                self.board.stop()
            except Exception as error:
                self.board_error = str(error)

    def stop(self, reason="Наведение остановлено"):
        with self.lock:
            self._stop_locked(reason)
            self.log(reason)

    def select(self, track_id, client=None):
        with self.lock:
            track = self.tracker.tracks.get(track_id)
            now = time.monotonic()
            if track_id is not None and (not track or not track.visible or
                    track.streak < self.tracker.confirmations or now - track.seen_at > 0.2):
                raise ValueError("Цель уже не видна. Выберите шарик на текущем кадре.")
            if self.active and client is not None and self.owner != client:
                raise ValueError("Управление уже запущено в другой вкладке")
            if self.active and client is not None and track_id is not None:
                if track.visible_center is None:
                    raise ValueError("Центр шарика пока не определён")
                if self.selected_id != track_id:
                    self.selected_id = track_id
                    self.last_target_motion = None
                    self.heartbeat_at = now
                    self.reason = "Переход к выбранному шарику"
                    self.log(f"Выбрана цель #{track_id:02d}")
                return
            self._stop_locked("Цель выбрана. Нажмите «Начать наведение».")
            self.selected_id = track_id
            if track_id is None:
                self.reason = "Выберите шарик на видео"
            self.log(f"Выбрана цель #{track_id:02d}" if track_id is not None else "Выбор цели сброшен")

    def _configure_locked(self):
        if not self.board:
            raise ValueError("Подключите SimpleBGC")
        if not self.board.read_motor_state():
            raise ValueError("Сначала включите моторы в блоке «КОНТРОЛЛЕР»")
        self._read_telemetry_locked()
        self.board.configure()
        self.last_config = time.monotonic()

    def set_motors(self, enabled):
        with self.lock:
            if not self.board:
                raise ValueError("Подключите SimpleBGC")
            was_enabled = self.board.read_motor_state()
            self._stop_locked("Переключение питания моторов — наведение остановлено")
            self.board.set_motors(enabled)
            if enabled and not was_enabled:
                # A reference sampled with unpowered motors can differ from the
                # stabilized position after startup. Rebase only on a real OFF→ON.
                self._read_telemetry_locked()
                self.home = {"pitch": self.angles["pitch"], "yaw": self.angles["yaw"]}
                self.log("Ноль программных лимитов обновлён после включения моторов")
            self.last_motor_poll = time.monotonic()
            self.board_error = None
            self.reason = "Моторы включены. Наведение остановлено." if enabled else "Моторы выключены. Наведение остановлено."
            self.log(self.reason)

    def arm(self, client):
        with self.lock:
            now = time.monotonic()
            if self.active and self.owner != client:
                raise ValueError("Управление уже запущено в другой вкладке")
            if self.camera_status != "live" or now - self.frame_at > 0.2:
                raise ValueError("Нет свежего изображения камеры")
            track = self.tracker.tracks.get(self.selected_id)
            if not track or not track.visible or track.streak < self.tracker.confirmations or now - track.seen_at > 0.2:
                raise ValueError("Выберите подтверждённую цель на видео")
            if track.visible_center is None:
                raise ValueError("Центр видимой части шарика пока не определён")
            self._configure_locked()
            self.owner = client
            self.heartbeat_at = time.monotonic()
            self.motion.reset()
            self.last_target_motion = None
            self.active = True
            self.reason = "Наведение на центр выбранного шарика"
            self.log(f"Наведение запущено: цель #{self.selected_id:02d}")

    def heartbeat(self, client):
        with self.lock:
            if self.owner == client:
                self.heartbeat_at = time.monotonic()

    def jog(self, pitch, yaw, client):
        with self.lock:
            if self.owner != client or self.jog_until <= time.monotonic():
                self._stop_locked("Ручное управление")
                self._configure_locked()
            speed = min(3, self.settings.max_speed)
            self.jog_rates = (pitch * speed * (-1 if self.settings.invert_pitch else 1),
                              yaw * speed * (-1 if self.settings.invert_yaw else 1))
            self.jog_until = time.monotonic() + 0.2
            self.owner = client
            self.heartbeat_at = time.monotonic()

    def stop_jog(self, client):
        with self.lock:
            if self.jog_until and self.owner == client:
                self._stop_locked("Ручное управление остановлено")

    def set_home(self):
        with self.lock:
            self._stop_locked("Нулевая точка лимитов обновлена")
            if not self.board:
                raise ValueError("Подключите контроллер")
            self._read_telemetry_locked()
            self.home = {"pitch": self.angles["pitch"], "yaw": self.angles["yaw"]}
            self.log("Текущее положение принято за ноль программных лимитов")

    def calibrate_aim(self, aim_x_percent, aim_y_percent):
        with self.lock:
            values = self.settings.public()
            values.update(aim_x_percent=aim_x_percent, aim_y_percent=aim_y_percent)
            self.apply_settings(values)
            self.reason = "Поправка прицела сохранена. Наведение остановлено."
            self.log(self.reason)

    def apply_settings(self, values):
        with self.lock:
            self._stop_locked("Настройки применены. Наведение остановлено.")
            self.settings = MotionSettings(**values)
            path = ROOT / "panel_settings.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
            temporary.replace(path)
            self.log("Настройки наведения сохранены")

    def control_tick(self, now=None, dt=0.02):
        with self.lock:
            now = time.monotonic() if now is None else now
            if not self.board:
                return
            try:
                if now - self.last_poll > 0.1:
                    self._read_telemetry_locked()
                if now - self.last_motor_poll > 0.5:
                    self.board.read_motor_state()
                    self.last_motor_poll = time.monotonic()
                if self.board.motors_on is False and (self.active or self.jog_until):
                    self._stop_locked("Моторы выключены — наведение остановлено")
                if now - self.telemetry_at > 0.3:
                    self._stop_locked("Нет свежих углов от контроллера")
                    return
                if self.owner and now - self.heartbeat_at > 1.2:
                    self._stop_locked("Связь с окном управления потеряна")
                    self.log(self.reason, "warning")
                offsets = (self.angles["pitch"] - self.home["pitch"], self.angles["yaw"] - self.home["yaw"])
                pitch = yaw = 0.0
                if self.jog_until > now:
                    pitch, yaw = self.jog_rates
                    if (pitch > 0 and offsets[0] >= self.settings.pitch_limit - 1) or (pitch < 0 and offsets[0] <= -self.settings.pitch_limit + 1):
                        pitch = 0.0
                    if (yaw > 0 and offsets[1] >= self.settings.yaw_limit - 1) or (yaw < 0 and offsets[1] <= -self.settings.yaw_limit + 1):
                        yaw = 0.0
                elif self.jog_until:
                    self._stop_locked("Проверка направления завершена")
                elif self.active:
                    track = self.tracker.tracks.get(self.selected_id)
                    if self.camera_status != "live" or now - self.frame_at > 0.2:
                        self._stop_locked("Камера не отдаёт свежие кадры")
                    elif not track:
                        self._stop_locked("Выбранная цель потеряна. Выберите шарик снова.")
                        self.log(self.reason, "warning")
                    elif (not track.visible or track.streak < self.tracker.confirmations or
                          now - track.seen_at >= self.tracker.reacquire_window or track.visible_center is None):
                        last = self.last_target_motion
                        if last is not None and now - last[0] < self.tracker.reacquire_window:
                            factor = max(0.0, 1 - max(0, now - last[0]) / self.tracker.reacquire_window)
                            pitch, yaw = self.motion.coast(last[1], offsets, self.settings, factor)
                            self.reason = "Краткий пропуск распознавания — плавное торможение"
                        else:
                            self.motion.reset()
                            self.last_target_motion = None
                            self.reason = "Цель временно не видна — движение остановлено"
                    else:
                        pitch, yaw, ex, ey = self.motion.compute(track.public(), self.dimensions,
                                                                offsets, self.settings, dt)
                        self.last_target_motion = (track.seen_at, (pitch, yaw))
                        self.error_px = {"x": ex, "y": ey}
                        if abs(ex) <= self.settings.deadband_px and abs(ey) <= self.settings.deadband_px:
                            self.reason = ("Цель в точке прицеливания" if
                                           self.settings.aim_x_percent or self.settings.aim_y_percent
                                           else "Цель в центре")
                        elif self.motion.limited_axes:
                            parts = []
                            for axis in self.motion.limited_axes:
                                key = axis.lower()
                                limit = getattr(self.settings, key + "_limit")
                                offset = self.angles[key] - self.home[key]
                                parts.append(f"{axis}: {offset:+.1f}° / ±{limit:g}°")
                            self.reason = "Ограничение " + "; ".join(parts) + ". Цель ещё не в центре."
                        elif self.motion.braking_axes and abs(pitch) < 0.01 and abs(yaw) < 0.01:
                            self.reason = "Торможение перед точкой прицеливания"
                        else:
                            self.reason = "Наведение на центр выбранного шарика"
                # Re-arm firmware watchdog while connected and in serial control.
                if self.board.configured:
                    self.board.speed(pitch, yaw)
                self.command = {"pitch": pitch, "yaw": yaw}
            except Exception as error:
                self._stop_locked("Ошибка связи с контроллером")
                self.board_error = str(error)
                self.log(f"Связь с контроллером: {error}", "error")
                try:
                    self.board.close()
                except Exception:
                    pass
                self.board = None
                self.board_info = None

    def _control_loop(self):
        previous = time.monotonic()
        while not self.shutdown_event.is_set():
            started = time.monotonic()
            self.control_tick(started, started - previous)
            previous = started
            self.shutdown_event.wait(max(0, 0.02 - (time.monotonic() - started)))

    def start_camera(self, source=0):
        with self.lock:
            if self.camera_thread and self.camera_thread.is_alive():
                raise ValueError("Сначала остановите текущую камеру")
            self._stop_locked("Камера запускается")
            self.selected_id = None
            self.tracker.clear()
            self.camera_generation += 1
            generation = self.camera_generation
            self.camera_source = source
            self.camera_status = "starting"
            self.camera_error = None
            self.frame_b64 = None
            self.frame_at = self.fps = self.inference_ms = 0.0
            self.camera_thread = threading.Thread(target=self._camera_loop, args=(source, generation),
                                                  name="balloon-camera", daemon=True)
            self.camera_thread.start()
            self.log(f"Запуск USB-камеры {source}")

    def stop_camera(self):
        with self.lock:
            self._stop_locked("Камера остановлена")
            self.camera_generation += 1
            self.camera_status = "off"
            self.selected_id = None
            self.tracker.clear()
            self.frame_b64 = None
            thread = self.camera_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)

    def _camera_loop(self, source, generation):
        cap = None
        capture = None
        stopped = threading.Event()
        available = threading.Condition()
        latest = deque(maxlen=1)
        try:
            import cv2
            import torch
            os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
            from ultralytics import YOLO
            from detection_filter import filter_result, load_regions
            from .balloon_geometry import visible_center
            weights = ROOT / "models/balloon.pt"
            if not weights.is_file():
                weights = ROOT / "runs/balloon_yolov8n/weights/best.pt"
            if not weights.is_file():
                raise FileNotFoundError("Не найдены обученные веса детектора")
            device = 0 if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                torch.set_num_threads(min(2, torch.get_num_threads()))
                if torch.get_num_interop_threads() > 1:
                    torch.set_num_interop_threads(1)
            config_path = ROOT / "camera_config.json"
            camera_config = json.loads(config_path.read_text(encoding="utf-8")).get(str(source), {}) if config_path.is_file() else {}
            model_path = weights
            if device == "cpu" and (cpu_model := camera_config.get("cpu_openvino_model")):
                candidate = ROOT / cpu_model
                if candidate.is_dir() and find_spec("openvino") is not None:
                    model_path = candidate
            model = YOLO(str(model_path), task="detect")
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(f"Камера {source} недоступна. Закройте другие программы с камерой.")
            if camera_config:
                if fourcc := camera_config.get("capture_fourcc"):
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["capture_width"]))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["capture_height"]))
                if capture_fps := camera_config.get("capture_fps"):
                    cap.set(cv2.CAP_PROP_FPS, capture_fps)
            inference_size = int(camera_config.get("inference_size", 640))
            if device == "cpu":
                inference_size = int(camera_config.get("cpu_inference_size", inference_size))
            crop_to_reference = camera_config.get("center_crop_to_reference", False)
            mjpeg = cv2.VideoWriter_fourcc(*"MJPG")
            native_jpeg = (
                not crop_to_reference and cap.getBackendName() == "V4L2"
                and int(cap.get(cv2.CAP_PROP_FOURCC)) == mjpeg
                and cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            )
            capture_dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            from .camera_geometry import center_crop_reference

            def read_frames():
                previous = None
                try:
                    while not stopped.is_set() and not self.shutdown_event.is_set():
                        ok, frame = cap.read()
                        captured_at = time.monotonic()
                        if not ok or frame is None:
                            raise RuntimeError("Камера перестала отдавать кадры")
                        if crop_to_reference:
                            frame = center_crop_reference(frame)
                        with available:
                            latest.append(frame)
                            available.notify()
                        if native_jpeg:
                            width, height = capture_dimensions
                            encoded = frame
                        else:
                            height, width = frame.shape[:2]
                            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if not ok:
                                raise RuntimeError("Не удалось подготовить изображение камеры")
                        frame_b64 = base64.b64encode(encoded).decode("ascii")
                        with self.lock:
                            if generation != self.camera_generation or stopped.is_set():
                                return
                            self.dimensions = (width, height)
                            self.frame_number += 1
                            self.frame_b64 = frame_b64
                            if previous is not None:
                                rate = 1 / max(captured_at - previous, 0.001)
                                self.fps = 0.9 * self.fps + 0.1 * rate if self.fps else rate
                        previous = captured_at
                except Exception as error:
                    self._camera_failed(generation, error)
                finally:
                    cap.release()
                    stopped.set()
                    with available:
                        available.notify()

            capture = threading.Thread(target=read_frames, name="balloon-video", daemon=True)
            capture.start()
            regions = None
            while not stopped.is_set() and not self.shutdown_event.is_set():
                with self.lock:
                    if generation != self.camera_generation:
                        return
                    threshold = self.settings.confidence
                with available:
                    available.wait_for(lambda: latest or stopped.is_set(), timeout=0.1)
                    if not latest or stopped.is_set():
                        continue
                    frame = latest.pop()
                if native_jpeg:
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                height, width = frame.shape[:2]
                if regions is None:
                    regions = load_regions(ROOT / "camera_exclusions.json", source, frame.shape)
                infer_start = time.monotonic()
                result = model.predict(frame, conf=threshold, iou=0.4, imgsz=inference_size, device=device, verbose=False)[0]
                result.update(boxes=result.boxes.data[result.boxes.conf >= threshold])
                filter_result(result, regions)
                detections = [{"box": box.xyxy[0].tolist(), "confidence": float(box.conf[0])} for box in result.boxes]
                for detection in detections:
                    detection["center"] = visible_center(frame, detection["box"])
                inference_ms = (time.monotonic() - infer_start) * 1000
                with self.lock:
                    if generation != self.camera_generation or stopped.is_set() or self.camera_status == "error":
                        return
                    self.frame_at = time.monotonic()
                    self.tracker.update(detections, self.frame_at)
                    self.inference_ms = inference_ms
                    self.device = "RTX 3060 / CUDA" if device == 0 else "CPU"
                    if self.camera_status != "live":
                        self.log(f"Камера {source}: {width}×{height}, детектор готов")
                    self.camera_status = "live"
        except Exception as error:
            self._camera_failed(generation, error)
        finally:
            stopped.set()
            if capture is not None and capture.ident is not None:
                capture.join()
            elif cap:
                cap.release()

    def _camera_failed(self, generation, error):
        with self.lock:
            if generation == self.camera_generation:
                self.camera_status = "error"
                self.camera_error = str(error)
                self._stop_locked("Ошибка камеры — наведение остановлено")
                self.log(str(error), "error")

    def snapshot(self, with_frame=False):
        with self.lock:
            now = time.monotonic()
            tracks = self.tracker.confirmed(now) if self.camera_status == "live" and now - self.frame_at < 0.2 else []
            target = next((track for track in tracks if track["id"] == self.selected_id), None)
            if target and target["center"] is not None:
                cx, cy = target["center"]
                ax, ay = aim_point(self.dimensions, self.settings)
                error = {"x": cx - ax, "y": cy - ay}
            else:
                error = {"x": None, "y": None}
            state = {"camera": {"status": self.camera_status, "error": self.camera_error,
                                "source": self.camera_source, "width": self.dimensions[0], "height": self.dimensions[1],
                                "fps": round(self.fps, 1), "inference_ms": round(self.inference_ms, 1), "device": self.device,
                                "frame_age_ms": round((now - self.frame_at) * 1000) if self.frame_at else None},
                     "controller": {"connected": self.board is not None, "info": self.board_info,
                                    "motors_enabled": self.board.motors_on if self.board and now - self.board.motors_at < 1.5 else None,
                                    "error": self.board_error, "angles": self.angles.copy(),
                                    "offsets": {key: self.angles[key] - self.home[key] for key in ("pitch", "yaw")},
                                    "watchdog_configured": bool(self.board and self.board.configured)},
                     "tracking": {"active": self.active, "selected_id": self.selected_id, "target": target,
                                  "error_px": error, "command": self.command.copy(), "reason": self.reason,
                                  "limited_axes": self.motion.limited_axes[:],
                                  "confirmations": self.tracker.confirmations},
                     "tracks": copy.deepcopy(tracks), "settings": self.settings.public(),
                     "events": list(self.events), "frame_number": self.frame_number}
            if with_frame:
                state["frame"] = self.frame_b64
            return state

    def close(self):
        self.shutdown_event.set()
        self.stop_camera()
        self.disconnect()
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=2)
