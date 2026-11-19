"""Apply the approved Frame IMU disable only, with byte-level backup and readback."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from control_panel.sbgc import SimpleBGC
from inspect_controller import PROFILE_COMMANDS, ReadOnlyBGC, decode_realtime, official_structures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    original = root/'diagnostics/controller_readonly_20261119_154042'
    folder = root/'diagnostics'/('frame_imu_disable_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    layouts = official_structures()
    offset = layouts['sbgcMainParams3_t'].frameIMU_Pos.offset
    assert offset == 74
    status = {'write_sent':False, 'approved_change':{'field':'frameIMU_Pos','before':3,'after':0},
              'before':{},'after':{},'telemetry':[]}

    class ApprovedSingleWriteBGC(ReadOnlyBGC):
        expected_payload = None
        sent = False

        def _write(self, command, payload=b''):
            if command == 22:
                if self.sent or payload != self.expected_payload or payload is None:
                    raise RuntimeError('Rejecting unexpected or repeated parameter write')
                self.sent = True
                status['write_sent'] = True
                (folder/'result.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
                return SimpleBGC._write(self,command,payload)
            return super()._write(command,payload)

    board = ApprovedSingleWriteBGC('COM9')
    try:
        if board.info['firmware_number'] != 2700:
            raise RuntimeError('Firmware differs from reviewed 2.70b0')
        # Read active + all stored profiles without switching profiles.
        for profile in (255,0,1,2,3,4):
            for command,(_,_,length) in PROFILE_COMMANDS.items():
                data = board.request(command,bytes([profile]),timeout=.8)
                if len(data) != length:
                    raise RuntimeError('Unexpected configuration length')
                status['before'][f'{profile}:{command}'] = data.hex()
                (folder/f'before_{profile}_{command}.bin').write_bytes(data)
                time.sleep(.035)
        before = bytes.fromhex(status['before']['255:21'])
        if before != (original/'profile_255_21.bin').read_bytes():
            raise RuntimeError('Active configuration changed since review; write aborted')
        if before[0] != 0 or before[offset] != 3:
            raise RuntimeError('Profile or old Frame IMU setting differs from reviewed state')
        payload = bytearray(before)
        payload[offset] = 0
        payload = bytes(payload)
        assert [i for i,(a,b) in enumerate(zip(before,payload)) if a != b] == [offset]
        board.expected_payload = payload
        (folder/'proposed_21.bin').write_bytes(payload)
        if not args.apply:
            print('Prepared only; use --apply for approved change')
            return
        # Send exactly once. Any timeout is an unknown outcome: do not retry writing.
        try:
            ack = board.request(22,payload,timeout=2,confirmation=True)
            status['ack_hex'] = ack.hex()
        except TimeoutError as error:
            status['ack_error'] = str(error)
        time.sleep(2)
        # Allow the controller to reinitialize, using read requests only.
        deadline = time.monotonic()+10
        while True:
            try:
                board.request(86,timeout=.8)
                break
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.3)
        changes = {}
        for profile in (255,0,1,2,3,4):
            for command,(_,_,length) in PROFILE_COMMANDS.items():
                data = board.request(command,bytes([profile]),timeout=.8)
                if len(data) != length:
                    raise RuntimeError('Unexpected readback length')
                key = f'{profile}:{command}'
                status['after'][key] = data.hex()
                (folder/f'after_{profile}_{command}.bin').write_bytes(data)
                old = bytes.fromhex(status['before'][key])
                diff = [{'offset':i,'before':a,'after':b} for i,(a,b) in enumerate(zip(old,data)) if a!=b]
                if diff:
                    changes[key] = diff
                # Frame IMU position is a hardware setting, potentially shared across profiles.
                allowed = [{'offset':offset,'before':3,'after':0}] if command == 21 else []
                if diff and diff != allowed:
                    raise RuntimeError(f'Unexpected readback changes in {key}: {diff}')
                time.sleep(.035)
        status['verified_changes'] = changes
        readback = bytes.fromhex(status['after']['255:21'])
        if readback[offset] != 0:
            raise RuntimeError('Frame IMU disable not confirmed')
        for _ in range(200):
            status['telemetry'].append(decode_realtime(board.request(23,timeout=.4)))
            time.sleep(.03)
        status['verified'] = True
        print('Verified changes:',json.dumps(changes))
        for i,axis in enumerate(('Roll','Pitch','Yaw')):
            values=[row['imu_degrees'][i] for row in status['telemetry']]
            print(axis,'angle range:',round(max(values)-min(values),5))
        print('System errors:',sorted({s['system_error'] for s in status['telemetry']}))
        print('I2C errors min/max:',min(s['i2c_errors'] for s in status['telemetry']),max(s['i2c_errors'] for s in status['telemetry']))
        print('Report:',folder/'result.json')
    except Exception as error:
        status['error'] = repr(error)
        raise
    finally:
        board.close()
        (folder/'result.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
