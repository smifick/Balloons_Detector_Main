import math
import struct
import time

import pytest

from control_panel.engine import Engine
from control_panel.sbgc import (Angles, CONTROL, CONTROL_CONFIG, GET_ANGLES,
                                MOTORS_ON, MOTORS_OFF, REALTIME_DATA_3,
                                PacketParser, SimpleBGC, control_config_payload,
                                firmware_name, packet, speed_payload)
from control_panel.tracking import MotionControl, MotionSettings, Tracker


def detection(box=(440, 200, 470, 230), confidence=0.8):
    return {"box": list(box), "confidence": confidence}


def test_official_crc_reference_vector():
    assert packet(86, b"\0\0").hex() == "245602580000e613"
    assert packet(82, b"\x01", version=1).hex() == "3e5201530101"


def test_parser_recovers_from_noise_corruption_and_partial_reads():
    parser = PacketParser()
    broken = bytearray(packet(73, b"bad"))
    broken[-1] ^= 1
    good = packet(73, b"actual angles")
    assert parser.feed(b"noise" + broken + good[:5]) == []
    assert parser.feed(good[5:]) == [(73, b"actual angles")]
    assert parser.errors > 0
    assert parser.feed(packet(86, b"v1", 1)) == [(86, b"v1")]


def test_control_axis_order_and_stop_packet():
    modes_and_axes = struct.unpack("<BBBhhhhhh", speed_payload(-3, 6))
    assert modes_and_axes[:3] == (0, 1, 1)  # Roll released, Pitch/Yaw speed.
    assert modes_and_axes[3:5] == (0, 0)
    assert modes_and_axes[5] < 0 and modes_and_axes[7] > 0
    assert struct.unpack("<BBBhhhhhh", speed_payload(0, 0))[3:] == (0,) * 6
    for bad in (float("nan"), float("inf"), 61):
        with pytest.raises(ValueError):
            speed_payload(bad, 0)


def test_watchdog_config_uses_270b0_fields_only():
    payload = control_config_payload()
    assert len(payload) == 41
    assert struct.unpack_from("<H", payload, 0)[0] == 500
    assert payload[2:7] == bytes(5)  # Keep channel priorities.
    for offset in (7, 14, 21):
        assert payload[offset:offset+7] == bytes((3, 3, 0, 0, 0, 0, 0))
    assert struct.unpack_from("<H", payload, 29)[0] == 1
    assert payload[31:] == bytes(10)  # No 2.72+ Euler options.
    assert firmware_name(2700) == "2.70b0"


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.incoming = bytearray()
        self.parser = PacketParser()
        self.commands = []
        self.closed = False
        self.confirm_config = True
        self.motors_enabled = True
        self.apply_motor_commands = True

    @property
    def in_waiting(self):
        return len(self.incoming)

    def write(self, data):
        for command, payload in self.parser.feed(data):
            self.commands.append((command, payload))
            if command == 86:
                self.incoming.extend(packet(86, struct.pack("<BHB", 30, 2700, 0)))
            elif command == GET_ANGLES:
                self.incoming.extend(packet(GET_ANGLES, struct.pack("<9h", 10, 11, 12, -100, 0, 0, -1000, 0, 0)))
            elif command == CONTROL_CONFIG and self.confirm_config:
                self.incoming.extend(packet(CONTROL, bytes((CONTROL_CONFIG,))))
            elif command == REALTIME_DATA_3:
                state = bytearray(63)
                state[57] = 1 if self.motors_enabled else 0
                self.incoming.extend(packet(REALTIME_DATA_3, state))
            elif command in (MOTORS_ON, MOTORS_OFF):
                if self.apply_motor_commands:
                    self.motors_enabled = command == MOTORS_ON
                self.incoming.extend(packet(CONTROL, bytes((command,))))
        return len(data)

    def read(self, length):
        data = bytes(self.incoming[:length])
        del self.incoming[:length]
        return data

    def close(self):
        self.closed = True


def test_real_transport_connect_is_read_only_and_arm_requires_config_ack(monkeypatch):
    import serial
    fake = FakeSerial()
    monkeypatch.setattr(serial, "Serial", lambda **kwargs: fake)
    board = SimpleBGC("COM_TEST")
    assert [command for command, _ in fake.commands] == [86, GET_ANGLES]
    assert board.info["firmware"] == "2.70b0"
    assert board.angles.pitch < 0 and board.angles.yaw < 0
    with pytest.raises(RuntimeError):
        board.speed(1, 1)
    board.configure()
    board.speed(2, -2)
    board.close()
    assert fake.closed
    assert fake.commands[-1] == (CONTROL, speed_payload(0, 0))


def test_failed_watchdog_confirmation_never_allows_movement(monkeypatch):
    import serial
    fake = FakeSerial()
    fake.confirm_config = False
    monkeypatch.setattr(serial, "Serial", lambda **kwargs: fake)
    board = SimpleBGC("COM_TEST")
    with pytest.raises(TimeoutError):
        board.configure()
    with pytest.raises(RuntimeError):
        board.speed(3, 4)
    board.close()
    assert all(command != CONTROL for command, payload in fake.commands)


def test_single_frame_false_positive_is_never_confirmed():
    tracker = Tracker()
    assert tracker.update([detection()], 1) == []
    assert tracker.update([], 1.04) == []
    assert tracker.update([detection()], 1.08) == []
    assert tracker.update([detection()], 1.12) == []
    confirmed = tracker.update([detection()], 1.16)
    assert len(confirmed) == 1
    assert confirmed[0]["streak"] == 3


def test_track_is_not_reassigned_to_distant_balloon_after_loss():
    tracker = Tracker()
    for t in (1, 1.03, 1.06):
        tracker.update([detection()], t)
    selected = tracker.confirmed(1.06)[0]["id"]
    tracker.update([detection((50, 50, 80, 80))], 1.09)
    assert not tracker.tracks[selected].visible
    assert tracker.tracks[selected].streak == 0
    assert any(key != selected for key in tracker.tracks)


def test_motion_direction_deadband_speed_and_acceleration():
    settings = MotionSettings()
    motion = MotionControl()
    target = {"box": [450, 100, 470, 120]}
    pitch, yaw, ex, ey = motion.compute(target, (640, 480), (0, 0), settings, 0.02)
    assert pitch > 0 and yaw > 0
    assert pitch <= 0.5 and yaw <= 0.5
    motion.reset()
    pitch, yaw, _, _ = motion.compute({"box": [315, 235, 325, 245]}, (640, 480), (0, 0), settings, 0.02)
    assert pitch == yaw == 0
    for _ in range(200):
        pitch, yaw, _, _ = motion.compute(target, (640, 480), (0, 0), settings, 0.02)
    assert max(abs(pitch), abs(yaw)) <= settings.max_speed


def test_follow_speed_reaches_selected_maximum_and_brakes_near_center():
    settings = MotionSettings(max_speed=30, gain_yaw=16, gain_pitch=16, acceleration=60)

    def steady_speed(error, axis):
        motion = MotionControl()
        x, y = (320 + error, 240) if axis == 'yaw' else (320, 240 - error)
        target = {'box': [x - 5, y - 5, x + 5, y + 5]}
        for _ in range(100):
            pitch, yaw, _, _ = motion.compute(target, (640, 480), (0, 0), settings, 0.02)
        return yaw if axis == 'yaw' else pitch

    for axis in ('yaw', 'pitch'):
        assert steady_speed(100, axis) > 29
        assert steady_speed(20, axis) > 15
        assert 0 < steady_speed(8, axis) < 3
        assert steady_speed(7, axis) == 0
        assert steady_speed(-100, axis) < -29
        assert steady_speed(400, axis) <= 30


def test_crossing_target_stops_old_direction_even_with_low_acceleration():
    settings = MotionSettings(max_speed=30, gain_yaw=4, gain_pitch=4, deadband_px=2, acceleration=10)
    motion = MotionControl()
    motion.yaw_speed = motion.pitch_speed = 30
    # A target crossing must never leave a nonzero command in the old direction.
    pitch, yaw, _, _ = motion.compute({'box':[290,260,300,270]}, (640,480), (0,0), settings, .02)
    assert pitch == yaw == 0
    pitch, yaw, _, _ = motion.compute({'box':[290,260,300,270]}, (640,480), (0,0), settings, .02)
    assert -.2 <= pitch < 0
    assert -.2 <= yaw < 0
    # Entering the deadband brakes immediately instead of coasting for seconds.
    motion.yaw_speed = motion.pitch_speed = 30
    pitch, yaw, _, _ = motion.compute({'box':[315,235,325,245]}, (640,480), (0,0), settings, .02)
    assert pitch == yaw == 0


@pytest.mark.parametrize('initial_error', [100, -100])
@pytest.mark.parametrize('acceleration', [10,60])
def test_delayed_gimbal_model_settles_without_repeated_target_crossings(initial_error, acceleration):
    # Delayed camera feedback and a lagging motor, rather than an instantaneous
    # simulator: 160 ms frame delay, 120 ms motor lag, 60 deg/s² motor braking.
    settings = MotionSettings(max_speed=30, gain_yaw=4, gain_pitch=4, deadband_px=2, acceleration=acceleration)
    motion = MotionControl()
    dt, pixels_per_degree = .02, 640/65
    error, velocity = initial_error, 0.0
    observations, errors = [error]*8, []
    crossings = 0
    for _ in range(600):
        observed = observations.pop(0)
        target = {'box':[315+observed,235,325+observed,245]}
        _, command, _, _ = motion.compute(target,(640,480),(0,0),settings,dt)
        velocity += max(-60*dt,min(60*dt,(command-velocity)*dt/.12))
        previous_error = error
        error -= velocity*pixels_per_degree*dt
        if previous_error*error < 0:
            crossings += 1
        observations.append(error)
        errors.append(error)
    assert max(abs(value) for value in errors[-50:]) < 3
    assert crossings <= 2
    assert max(abs(value) for value in errors) <= abs(initial_error)+1


def test_limits_block_outward_motion_but_allow_return_and_inversion():
    settings = MotionSettings()
    motion = MotionControl()
    target = {"box": [450, 100, 470, 120]}
    pitch, yaw, _, _ = motion.compute(target, (640,480), (30,45), settings, 0.02)
    assert pitch == yaw == 0
    settings.invert_pitch = settings.invert_yaw = True
    pitch, yaw, _, _ = motion.compute(target, (640,480), (30,45), settings, 0.02)
    assert pitch < 0 and yaw < 0


def test_predictive_braking_uses_new_frames_and_never_reverses_before_crossing():
    settings = MotionSettings(max_speed=30,gain_yaw=4,deadband_px=1,acceleration=40)
    motion = MotionControl()
    def target(error, stamp, identity=1):
        return {'id':identity,'observed_at':stamp,'box':[315+error,235,325+error,245]}
    motion.compute(target(100,1),(640,480),(0,0),settings,.02)
    motion.yaw_speed = 30
    _, yaw, ex, _ = motion.compute(target(40,1.1),(640,480),(0,0),settings,.02)
    assert ex == 40 and yaw == 0  # Brake while measured target is still to the right.
    velocity = motion.image_velocity[:]
    motion.compute(target(40,1.1),(640,480),(0,0),settings,.02)
    assert motion.image_velocity == velocity  # Repeated control ticks aren't new video frames.
    _,yaw,_,_ = motion.compute(target(40,1.13,identity=2),(640,480),(0,0),settings,.02)
    assert yaw > 0 and motion.image_velocity == [0,0]  # New target has no inherited derivative.


@pytest.mark.parametrize('delay', [.08,.16,.24])
@pytest.mark.parametrize('initial_error', [-180,180])
def test_predictive_approach_settles_with_motor_inertia_and_delayed_frames(delay,initial_error):
    settings=MotionSettings(max_speed=30,gain_yaw=4,deadband_px=1,acceleration=40,braking_lookahead=.28)
    motion=MotionControl()
    dt=.02
    error=float(initial_error)
    velocity=0.0
    observations=[error]*round(delay/dt)
    errors=[]
    for step in range(900):
        observed=observations.pop(0)
        target={'id':1,'observed_at':1+step*dt,'box':[315+observed,235,325+observed,245]}
        _,command,_,_=motion.compute(target,(640,480),(0,0),settings,dt)
        velocity+=max(-60*dt,min(60*dt,(command-velocity)*dt/.16))
        error-=velocity*(640/65)*dt
        observations.append(error)
        errors.append(error)
    assert max(abs(x) for x in errors[-100:])<1.5
    assert max((-1 if initial_error>0 else 1)*x for x in errors)<3


@pytest.fixture
def engine(tmp_path, monkeypatch):
    import control_panel.engine as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    instance = Engine()
    yield instance
    instance.close()


def ready_engine(engine):
    engine.connect(simulated=True)
    now = time.monotonic()
    engine.camera_status = "live"
    engine.frame_at = now
    for instant in (now-0.06, now-0.03, now):
        engine.tracker.update([detection()], instant)
    selected = engine.tracker.confirmed(now)[0]["id"]
    engine.select(selected)
    engine.arm("test-client-123")
    engine.control_tick(dt=0.02)
    return selected


def test_loss_stops_motion_without_switching_to_another_target(engine):
    selected = ready_engine(engine)
    previous_speed = engine.command["yaw"]
    now = engine.frame_at + 0.06
    engine.tracker.update([detection((20,20,40,40))], now)
    engine.frame_at = now
    engine.control_tick(now, dt=0.02)
    assert 0 < engine.command["yaw"] < previous_speed
    now += engine.tracker.reacquire_window
    engine.tracker.update([detection((20,20,40,40))], now)
    engine.frame_at = now
    engine.control_tick(now, dt=0.02)
    assert engine.command == {"pitch": 0, "yaw": 0}
    assert engine.selected_id == selected
    assert engine.active  # May re-acquire this same ID within 0.6 s.


def test_one_missed_frame_does_not_require_reconfirming_the_selected_balloon(engine):
    selected = ready_engine(engine)
    missed_at = engine.frame_at + 0.04
    engine.tracker.update([], missed_at)
    engine.frame_at = missed_at
    engine.control_tick(missed_at)
    coasting = engine.command["yaw"]
    returned_at = missed_at + 0.04
    engine.tracker.update([detection()], returned_at)
    engine.frame_at = returned_at
    engine.control_tick(returned_at)
    assert engine.selected_id == selected
    assert engine.tracker.confirmed(returned_at)[0]["id"] == selected
    assert engine.command["yaw"] > coasting > 0


def test_active_owner_can_switch_target_without_rearming(engine):
    selected = ready_engine(engine)
    now = time.monotonic()
    for observed_at in (now + 0.01, now + 0.02, now + 0.03):
        engine.tracker.update([detection(), detection((365, 155, 395, 185))], observed_at)
    other = next(track["id"] for track in engine.tracker.confirmed(now + 0.03) if track["id"] != selected)
    with pytest.raises(ValueError):
        engine.select(other, "other-client")
    engine.select(other, "test-client-123")
    engine.control_tick()
    assert engine.active and engine.selected_id == other
    assert engine.command["pitch"] > 0 and engine.command["yaw"] > 0
    engine.select(None, "test-client-123")
    assert not engine.active


def test_displayed_error_is_relative_to_calibrated_aim(engine):
    ready_engine(engine)
    target=engine.snapshot()['tracking']['target']
    x,y=target['center']
    engine.settings.aim_x_percent=(x/engine.dimensions[0]-.5)*100
    engine.settings.aim_y_percent=(y/engine.dimensions[1]-.5)*100
    error=engine.snapshot()['tracking']['error_px']
    assert error['x']==pytest.approx(0)
    assert error['y']==pytest.approx(0)


@pytest.mark.parametrize("failure", ["heartbeat", "camera"])
def test_heartbeat_and_camera_freshness_disarm(engine, failure):
    ready_engine(engine)
    if failure == "heartbeat":
        engine.heartbeat_at = time.monotonic() - 2
    else:
        engine.frame_at = time.monotonic() - 1
    engine.control_tick(dt=0.02)
    assert not engine.active
    assert engine.command == {"pitch": 0, "yaw": 0}
    assert engine.board.commands[-1] == (0, 0)


def test_stop_is_immediate_and_wrong_tab_cannot_renew_lease(engine):
    ready_engine(engine)
    previous = engine.heartbeat_at
    engine.heartbeat("other-client")
    assert engine.heartbeat_at == previous
    engine.stop()
    engine.control_tick(dt=0.02)
    assert not engine.active
    assert engine.board.commands[-1] == (0, 0)


def test_manual_impulse_expires_without_browser_input(engine):
    engine.connect(simulated=True)
    engine.jog(0, 1, "test-client-123")
    engine.control_tick()
    assert engine.command["yaw"] > 0
    engine.jog_until = time.monotonic() - 0.01
    engine.control_tick()
    assert engine.command["yaw"] == 0
    assert engine.owner is None


def test_manual_input_renews_until_owning_tab_releases(engine):
    engine.connect(simulated=True)
    owner = "test-client-123"
    engine.jog(1, 0, owner)
    engine.control_tick()
    assert engine.command["pitch"] > 0
    deadline = engine.jog_until
    engine.jog(0, -1, owner)
    assert engine.jog_until > deadline
    engine.stop_jog("other-client")
    engine.control_tick()
    assert engine.command["yaw"] < 0
    engine.stop_jog(owner)
    assert engine.command == {"pitch": 0.0, "yaw": 0.0}
    assert engine.board.commands[-1] == (0, 0)


def test_telemetry_failure_disconnects_and_stops(engine, monkeypatch):
    ready_engine(engine)
    def broken_read():
        raise TimeoutError("Board unplugged")
    monkeypatch.setattr(engine.board, "read_angles", broken_read)
    engine.last_poll = 0
    engine.control_tick()
    assert not engine.active
    assert engine.board is None
    assert engine.command == {"pitch": 0, "yaw": 0}


def test_web_api_permissions_threshold_and_selected_target_flow(engine):
    from fastapi.testclient import TestClient
    from control_panel.server import create_app
    app = create_app(engine=engine, autostart_camera=False)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.post("/api/connect", json={"simulated": True}).status_code == 403
        client.headers["X-Control-Token"] = app.state.control_token
        assert client.post("/api/connect", json={"simulated": True}).status_code == 200
        assert client.post("/api/settings", json={"confidence": .34}).status_code == 422
        assert client.post("/api/start", json={"client":"test-client-123"}).status_code == 409
        now = time.monotonic()
        with engine.lock:
            engine.camera_status = "live"
            engine.frame_at = now
            for instant in (now-.06,now-.03,now):
                engine.tracker.update([detection()],instant)
            target = engine.tracker.confirmed(now)[0]["id"]
        assert client.post("/api/target",json={"track_id":target}).status_code == 200
        assert client.post("/api/start",json={"client":"test-client-123"}).status_code == 200
        assert client.post("/api/stop",json={}).json()["tracking"]["active"] is False
        assert client.post("/api/stop",headers={"Sec-Fetch-Site":"cross-site"},json={}).status_code == 403
        assert client.get("/api/state",headers={"Host":"evil.example"}).status_code == 400


def test_settings_reject_nonfinite_values():
    from pydantic import ValidationError
    from control_panel.schemas import SettingsInput
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            SettingsInput(max_speed=value)


def test_calibrated_aim_tracks_spot_instead_of_camera_center_at_any_resolution():
    settings = MotionSettings(deadband_px=2, aim_x_percent=5, aim_y_percent=10)
    for width, height in ((640, 480), (1280, 960)):
        motion = MotionControl()
        x, y = width * .55, height * .60
        target = {'box': [x-5, y-5, x+5, y+5]}
        pitch, yaw, ex, ey = motion.compute(target, (width,height), (0,0), settings, .02)
        assert ex == pytest.approx(0)
        assert ey == pytest.approx(0)
        assert pitch == yaw == 0
        _, _, ex, ey = motion.compute({'box':[width/2-5,height/2-5,width/2+5,height/2+5]},
                                     (width,height), (0,0), settings, .02)
        assert ex == pytest.approx(-width*.05)
        assert ey == pytest.approx(-height*.10)


def test_calibration_stops_motion_preserves_limits_and_rejects_invalid_spots(engine):
    from fastapi.testclient import TestClient
    from control_panel.server import create_app
    app = create_app(engine=engine, autostart_camera=False)
    with TestClient(app) as client:
        ready_engine(engine)
        previous = engine.settings.public()
        assert client.post('/api/calibration',json={'aim_x_percent':1,'aim_y_percent':3}).status_code == 403
        client.headers['X-Control-Token'] = app.state.control_token
        assert client.post('/api/calibration',json={'aim_x_percent':41,'aim_y_percent':0}).status_code == 422
        result = client.post('/api/calibration',json={'aim_x_percent':1,'aim_y_percent':3})
        assert result.status_code == 200
        assert result.json()['tracking']['active'] is False
        assert engine.command == {'pitch':0,'yaw':0}
        for key, value in previous.items():
            if not key.startswith('aim_'):
                assert getattr(engine.settings,key) == value
        assert engine.settings.aim_y_percent == 3
        assert client.post('/api/calibration',json={'aim_x_percent':0,'aim_y_percent':0}).status_code == 200
        assert engine.settings.aim_x_percent == engine.settings.aim_y_percent == 0


def test_motor_commands_and_state_are_confirmed_by_board(monkeypatch):
    import serial
    fake = FakeSerial()
    monkeypatch.setattr(serial, "Serial", lambda **kwargs: fake)
    board = SimpleBGC("COM_TEST")
    assert board.read_motor_state() is True
    board.configure()
    board.set_motors(False)
    assert board.motors_on is False
    assert not board.configured
    assert (MOTORS_OFF, b"\x02") in fake.commands
    board.set_motors(True)
    assert board.motors_on is True
    assert (MOTORS_ON, b"") in fake.commands
    board.close()


def test_motor_ack_without_state_change_is_not_reported_as_success(monkeypatch):
    import serial
    fake = FakeSerial()
    fake.apply_motor_commands = False
    monkeypatch.setattr(serial, "Serial", lambda **kwargs: fake)
    board = SimpleBGC("COM_TEST")
    with pytest.raises(RuntimeError, match="состояние моторов не изменилось"):
        board.set_motors(False)
    assert board.motors_on is True
    board.close()


def test_power_off_stops_tracking_and_on_never_resumes_it(engine):
    ready_engine(engine)
    engine.set_motors(False)
    assert not engine.active
    assert engine.owner is None
    assert engine.snapshot()["controller"]["motors_enabled"] is False
    commands_after_off = len(engine.board.commands)
    engine.control_tick()
    assert len(engine.board.commands) == commands_after_off
    with pytest.raises(ValueError, match="включите моторы"):
        engine.jog(1,0,"test-client-123")
    engine.set_motors(True)
    engine.control_tick()
    assert engine.snapshot()["controller"]["motors_enabled"] is True
    assert not engine.active
    assert engine.command == {"pitch":0,"yaw":0}


def test_motor_api_requires_connection_and_explicit_boolean(engine):
    from fastapi.testclient import TestClient
    from control_panel.server import create_app
    app=create_app(engine=engine,autostart_camera=False)
    with TestClient(app) as client:
        assert client.post('/api/motors',json={'enabled':False}).status_code==403
        client.headers['X-Control-Token']=app.state.control_token
        assert client.post('/api/motors',json={'enabled':False}).status_code==409
        assert client.post('/api/connect',json={'simulated':True}).status_code==200
        assert client.post('/api/motors',json={'enabled':'true'}).status_code==422
        assert client.post('/api/motors',json={'enabled':False}).json()['controller']['motors_enabled'] is False
        result=client.post('/api/motors',json={'enabled':True}).json()
        assert result['controller']['motors_enabled'] is True
        assert result['tracking']['active'] is False


def test_limit_reason_reports_actual_stall_instead_of_claiming_motion(engine):
    ready_engine(engine)
    now=time.monotonic()
    engine.tracker.clear()
    for t in (now-.06,now-.03,now):
        engine.tracker.update([detection((205,236,225,256))],t)
    engine.selected_id=next(iter(engine.tracker.tracks))
    engine.home['yaw']=55
    engine.angles['yaw']=0
    engine.last_poll=now
    engine.motion.reset()
    engine.control_tick()
    snapshot=engine.snapshot()
    assert snapshot['tracking']['command']['yaw']==0
    assert snapshot['tracking']['limited_axes']==['Yaw']
    assert 'Ограничение Yaw' in snapshot['tracking']['reason']
    assert 'Цель ещё не в центре' in snapshot['tracking']['reason']
    assert engine.settings.yaw_limit==45


def test_power_on_references_limits_once_but_resume_preserves_reference(engine):
    engine.connect(simulated=True)
    engine.set_motors(False)
    engine.board.yaw=60
    engine.board.pitch=10
    engine.set_motors(True)
    assert engine.home['yaw']==pytest.approx(60)
    assert engine.home['pitch']==pytest.approx(10)
    reference=engine.home.copy()
    engine.board.yaw=65
    engine.set_motors(True)
    assert engine.home==reference  # Idempotent ON must not shift the travel envelope.
    engine.stop()
    assert engine.home==reference
