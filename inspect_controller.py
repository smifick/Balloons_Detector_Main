"""Read-only COM9 configuration backup and idle telemetry; never sends control/write commands."""
from __future__ import annotations

import ctypes as ct
import json
import re
import struct
import time
from datetime import datetime
from pathlib import Path

from control_panel.sbgc import SimpleBGC

ROOT = Path(__file__).resolve().parent
PROFILE_COMMANDS = {
    21: ("main", "sbgcMainParams3_t", 134),
    33: ("extended", "sbgcMainParamsExt_t", 104),
    62: ("extended2", "sbgcMainParamsExt2_t", 151),
    104: ("extended3", "sbgcMainParamsExt3_t", 221),
}


class ReadOnlyBGC(SimpleBGC):
    def _write(self, command, payload=b""):
        if command not in {86, 73, 23, 25, *PROFILE_COMMANDS}:
            raise RuntimeError(f"Read-only inspection rejects command {command}")
        return super()._write(command, payload)


def official_structures():
    """Build packed layouts from the downloaded, Apache-licensed official C header."""
    source = (ROOT / "research/basecam/profiles.h").read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    definitions = dict((name, body) for body, name in re.findall(
        r"typedef\s+struct\s+PACKED__\s*\{([^{}]*)\}\s*(\w+)\s*;", source, re.S))
    types = {"ui8": ct.c_uint8, "i8": ct.c_int8, "ui16": ct.c_uint16,
             "i16": ct.c_int16, "ui32": ct.c_uint32, "i32": ct.c_int32}

    def build(name):
        if name in types:
            return types[name]
        fields = []
        for declaration in definitions[name].split(";"):
            if not declaration.strip():
                continue
            match = re.fullmatch(r"\s*(\w+)\s+(.+?)\s*", declaration, re.S)
            if not match:
                raise ValueError(f"Unsupported declaration: {declaration}")
            base_type = build(match[1])
            for variable in match[2].split(","):
                match_var = re.fullmatch(r"\s*(\w+)\s*((?:\[\d+\]\s*)*)", variable)
                if not match_var:
                    raise ValueError(f"Unsupported field: {variable}")
                field_type = base_type
                for size in reversed(re.findall(r"\[(\d+)\]", match_var[2])):
                    field_type *= int(size)
                fields.append((match_var[1], field_type))
        types[name] = type(name, (ct.LittleEndianStructure,), {"_pack_": 1, "_fields_": fields})
        return types[name]

    result = {}
    for _, name, size in PROFILE_COMMANDS.values():
        result[name] = build(name)
        if ct.sizeof(result[name]) != size:
            raise ValueError(f"Official layout size mismatch for {name}")
    return result


def public(value):
    if isinstance(value, ct.Array):
        return [public(item) for item in value]
    if isinstance(value, ct.Structure):
        return {name: public(getattr(value, name)) for name, _ in value._fields_
                if not name.startswith("reserved")}
    return value


def decode_realtime(data):
    if len(data) != 63:
        raise ValueError(f"Unexpected telemetry length {len(data)}")
    return {
        "at": time.monotonic(),
        "gyro_dps": [struct.unpack_from("<h", data, offset)[0]*.06103701895 for offset in (2, 6, 10)],
        "imu_degrees": [n*360/16384 for n in struct.unpack_from("<3h", data, 32)],
        "frame_imu_degrees": [n*360/16384 for n in struct.unpack_from("<3h", data, 38)],
        "target_degrees": [n*360/16384 for n in struct.unpack_from("<3h", data, 44)],
        "rc_input": list(struct.unpack_from("<3h", data, 20)),
        "serial_errors": struct.unpack_from("<H", data, 12)[0],
        "system_error": struct.unpack_from("<H", data, 14)[0],
        "i2c_errors": struct.unpack_from("<H", data, 52)[0],
        "battery_volts": struct.unpack_from("<H", data, 55)[0]/100,
        "motors_on": bool(data[57] & 1), "profile_id": data[59],
        "motor_power": list(data[60:63]),
    }


def main():
    layouts = official_structures()
    folder = ROOT / "diagnostics" / ("controller_readonly_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    report = {"read_only": True, "source": "official BaseCam profiles.h; firmware 2.70 legacy scales",
              "profiles": {}, "telemetry": []}
    board = ReadOnlyBGC("COM9")
    try:
        report["board"] = board.info
        for profile in (255, 0, 1, 2, 3, 4):
            values = {}
            for command, (label, name, length) in PROFILE_COMMANDS.items():
                data = board.request(command, bytes([profile]), timeout=.8)
                (folder / f"profile_{profile}_{command}.bin").write_bytes(data)
                if len(data) != length:
                    raise ValueError(f"Command {command}: expected {length}, received {len(data)}")
                values[label] = public(layouts[name].from_buffer_copy(data))
                time.sleep(.035)
            report["profiles"][str(profile)] = values
            (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for _ in range(200):
            report["telemetry"].append(decode_realtime(board.request(23, timeout=.3)))
            time.sleep(.03)
    finally:
        board.close()  # configured=False: closes the serial port without a CONTROL packet.
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", folder / "report.json")
    print(json.dumps(report["profiles"]["255"], indent=2))
    for i, axis in enumerate(("Roll", "Pitch", "Yaw")):
        angles = [sample["imu_degrees"][i] for sample in report["telemetry"]]
        gyros = [sample["gyro_dps"][i] for sample in report["telemetry"]]
        print(axis, "angle min/max:", min(angles), max(angles), "gyro min/max:", min(gyros), max(gyros))
    print("Telemetry first/last:", json.dumps([report["telemetry"][0], report["telemetry"][-1]]))


if __name__ == "__main__":
    main()
