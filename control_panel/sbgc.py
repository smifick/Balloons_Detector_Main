"""Small SimpleBGC Serial API transport, based on the official protocol.

Reference: https://www.basecamelectronics.com/serialapi/
Uses commands available in 2.70b0; no profile/EEPROM writes.
"""
from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass

BOARD_INFO = 86
GET_ANGLES = 73
CONTROL = 67
CONTROL_CONFIG = 90
MOTORS_ON = 77
MOTORS_OFF = 109
REALTIME_DATA_3 = 23
ERROR = 255
SPEED_UNIT = 0.1220740379
ANGLE_UNIT = 360 / 16384


def crc16(data: bytes) -> int:
    value = 0
    for byte in data:
        for bit in range(8):
            different = ((byte >> bit) & 1) != ((value >> 15) & 1)
            value = (value << 1) & 0xFFFF
            if different:
                value ^= 0x8005
    return value


def packet(command: int, payload: bytes = b"", version: int = 2) -> bytes:
    if not 0 <= command <= 255 or len(payload) > 255:
        raise ValueError("Invalid Serial API packet")
    header = bytes((command, len(payload), (command + len(payload)) & 255))
    if version == 2:
        return b"$" + header + payload + struct.pack("<H", crc16(header + payload))
    if version == 1:
        return b">" + header + payload + bytes((sum(payload) & 255,))
    raise ValueError("Serial API version must be 1 or 2")


class PacketParser:
    def __init__(self):
        self.buffer = bytearray()
        self.errors = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buffer.extend(data)
        found = []
        while len(self.buffer) >= 4:
            start, command, size, checksum = self.buffer[:4]
            if start not in (0x24, 0x3E) or checksum != ((command + size) & 255):
                del self.buffer[0]
                self.errors += 1
                continue
            total = 4 + size + (2 if start == 0x24 else 1)
            if len(self.buffer) < total:
                break
            payload = bytes(self.buffer[4:4 + size])
            expected = struct.unpack("<H", self.buffer[4 + size:total])[0] if start == 0x24 else self.buffer[4 + size]
            actual = crc16(bytes(self.buffer[1:4 + size])) if start == 0x24 else sum(payload) & 255
            if actual != expected:
                del self.buffer[0]
                self.errors += 1
                continue
            del self.buffer[:total]
            found.append((command, payload))
        return found


def firmware_name(number: int) -> str:
    return f"{number // 1000}.{number % 1000 // 10:02d}b{number % 10}"


def speed_payload(pitch: float, yaw: float) -> bytes:
    if not all(math.isfinite(value) and abs(value) <= 60 for value in (pitch, yaw)):
        raise ValueError("Axis speed must be finite and within ±60 deg/s")
    # Three mode bytes, followed by (speed, angle) for Roll, Pitch, Yaw.
    return struct.pack("<BBBhhhhhh", 0, 1, 1, 0, 0,
                       round(pitch / SPEED_UNIT), 0, round(yaw / SPEED_UNIT), 0)


def control_config_payload(timeout_ms: int = 500) -> bytes:
    if not 100 <= timeout_ms <= 2000:
        raise ValueError("Invalid command timeout")
    data = bytearray(41)
    struct.pack_into("<H", data, 0, timeout_ms)
    # Keep priorities and GUI acceleration settings. LPF smooths 50 Hz updates.
    for axis in range(3):
        data[7 + axis * 7] = 3
        data[8 + axis * 7] = 3
    # FLAG_NO_CONFIRM applies to CONTROL; CONFIG still returns confirmation.
    struct.pack_into("<H", data, 29, 1)
    return bytes(data)


@dataclass
class Angles:
    roll: float
    pitch: float
    yaw: float
    received_at: float


class SimpleBGC:
    simulated = False

    def __init__(self, port: str, baud: int = 115200, version: int = 2):
        import serial
        self.port = port
        self.version = version
        self.serial = serial.Serial(port=port, baudrate=baud, timeout=0.015, write_timeout=0.12)
        self.parser = PacketParser()
        self.lock = threading.RLock()
        self.info = {}
        self.configured = False
        self.closed = False
        self.motors_on = None
        self.motors_at = 0.0
        try:
            data = self.request(BOARD_INFO, timeout=0.8)
            if len(data) < 4:
                raise RuntimeError("Неполный ответ BOARD_INFO")
            firmware = struct.unpack_from("<H", data, 1)[0]
            if not 2610 <= firmware < 4000:
                raise RuntimeError(f"Прошивка {firmware_name(firmware)} не поддерживает нужный тайм-аут управления")
            self.info = {"board_version": f"{data[0] / 10:.1f}",
                         "firmware": firmware_name(firmware), "firmware_number": firmware,
                         "port": port, "baud": baud, "protocol": version, "simulated": False}
            self.angles = self.read_angles()
        except Exception:
            self.serial.close()
            self.closed = True
            raise

    def _write(self, command: int, payload: bytes = b""):
        if self.closed:
            raise ConnectionError("COM-порт закрыт")
        data = packet(command, payload, self.version)
        if self.serial.write(data) != len(data):
            raise ConnectionError("Неполная запись в COM-порт")

    def request(self, command: int, payload: bytes = b"", timeout: float = 0.12,
                confirmation: bool = False) -> bytes:
        with self.lock:
            self._write(command, payload)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                chunk = self.serial.read(min(max(self.serial.in_waiting, 1), 512))
                for response, data in self.parser.feed(chunk):
                    if response == ERROR:
                        raise RuntimeError(f"SimpleBGC отклонил команду {command}: {data.hex()}")
                    if confirmation:
                        if response == CONTROL and data and data[0] == command:
                            return data
                    elif response == command:
                        return data
            raise TimeoutError(f"Нет ответа SimpleBGC на команду {command}")

    def read_angles(self) -> Angles:
        data = self.request(GET_ANGLES)
        if len(data) != 18:
            raise RuntimeError("Неверный размер ответа GET_ANGLES")
        values = struct.unpack("<9h", data)
        self.angles = Angles(*(values[i] * ANGLE_UNIT for i in (0, 3, 6)), time.monotonic())
        return self.angles

    def configure(self):
        self.request(CONTROL_CONFIG, control_config_payload(), timeout=0.5, confirmation=True)
        self.configured = True

    def read_motor_state(self):
        data = self.request(REALTIME_DATA_3)
        if len(data) != 63:
            self.motors_on = None
            raise RuntimeError("Неверный размер ответа REALTIME_DATA_3")
        self.motors_on = bool(data[57] & 1)
        self.motors_at = time.monotonic()
        return self.motors_on

    def set_motors(self, enabled: bool):
        with self.lock:
            if self.read_motor_state() == enabled:
                return
            self.configured = False
            command = MOTORS_ON if enabled else MOTORS_OFF
            # Safe-stop mode is supported since 2.68b7; earlier boards use normal off.
            payload = b"" if enabled else (b"\x02" if self.info["firmware_number"] >= 2687 else b"")
            self.request(command, payload, timeout=2, confirmation=True)
            deadline = time.monotonic() + 3
            while True:
                if self.read_motor_state() == enabled:
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError("Команда принята, но состояние моторов не изменилось")
                time.sleep(0.05)

    def speed(self, pitch: float, yaw: float):
        if not self.configured:
            raise RuntimeError("Тайм-аут управления не подтверждён платой")
        with self.lock:
            self._write(CONTROL, speed_payload(pitch, yaw))

    def stop(self):
        if self.configured and not self.closed:
            self.speed(0, 0)

    def close(self):
        with self.lock:
            try:
                self.stop()
            finally:
                self.serial.close()
                self.closed = True


class SimulatedBGC:
    """Explicit UI/test mode. It never opens a serial port."""
    simulated = True

    def __init__(self):
        self.info = {"board_version": "3.0", "firmware": "2.70b0 · симуляция",
                     "firmware_number": 2700, "port": "SIMULATION", "baud": 115200,
                     "protocol": 2, "simulated": True}
        self.configured = False
        self.closed = False
        self.pitch = self.yaw = 0.0
        self.pitch_speed = self.yaw_speed = 0.0
        self.updated = time.monotonic()
        self.commands = []
        self.motors_on = True
        self.motors_at = time.monotonic()
        self.motor_commands = []

    def _advance(self):
        now = time.monotonic()
        dt = min(now - self.updated, 0.5)
        self.pitch += self.pitch_speed * dt
        self.yaw += self.yaw_speed * dt
        self.updated = now

    def configure(self):
        self.configured = True

    def read_motor_state(self):
        self.motors_at = time.monotonic()
        return self.motors_on

    def set_motors(self, enabled):
        self.stop()
        self.motors_on = enabled
        self.motors_at = time.monotonic()
        self.configured = False
        self.motor_commands.append(enabled)

    def read_angles(self):
        self._advance()
        return Angles(0, self.pitch, self.yaw, time.monotonic())

    def speed(self, pitch, yaw):
        if not self.configured or self.closed:
            raise RuntimeError("Симулятор не готов")
        if not self.motors_on and (pitch or yaw):
            raise RuntimeError("Моторы выключены")
        self._advance()
        self.pitch_speed, self.yaw_speed = pitch, yaw
        self.commands.append((pitch, yaw))

    def stop(self):
        if self.configured and not self.closed:
            self.speed(0, 0)

    def close(self):
        self.stop()
        self.closed = True
