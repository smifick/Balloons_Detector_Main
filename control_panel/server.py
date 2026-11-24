from __future__ import annotations

import asyncio
import hmac
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .engine import Engine
from .schemas import AimInput, CameraInput, ClientInput, ConnectionInput, JogInput, MotorInput, SettingsInput, TargetInput

STATIC = Path(__file__).parent / "static"


def create_app(engine=None, autostart_camera=True):
    engine = engine or Engine()
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        if autostart_camera:
            engine.start_camera(0)
        yield
        engine.close()

    app = FastAPI(title="Balloon Control · SimpleBGC", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.state.control_token = token
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_control(request: Request, call_next):
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Разрешено только локальное окно панели"}, status_code=403)
        if request.method == "POST":
            if not hmac.compare_digest(request.headers.get("x-control-token", ""), token):
                return JSONResponse({"detail": "Обновите страницу панели"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8").replace("__CONTROL_TOKEN__", token)

    @app.get("/api/state")
    def state():
        return engine.snapshot()

    @app.get("/api/ports")
    def ports():
        from serial.tools import list_ports
        found = [{"port": port.device, "description": port.description,
                  "usb": bool(port.vid), "preferred": port.vid == 0x10C4 and port.pid == 0xEA60}
                 for port in list_ports.comports()]
        return sorted(found, key=lambda port: (not port["preferred"], not port["usb"], port["port"]))

    @app.get("/api/events")
    async def events(request: Request, client: str = ""):
        async def stream():
            last_frame = None
            last_sent = 0.0
            loop = asyncio.get_running_loop()
            try:
                while not await request.is_disconnected():
                    snapshot = engine.snapshot(with_frame=True)
                    frame = snapshot["frame_number"] if snapshot["frame"] else None
                    now = loop.time()
                    if frame != last_frame or now - last_sent >= 0.08:
                        if frame == last_frame:
                            snapshot["frame"] = None
                        last_frame, last_sent = frame, now
                        yield "data: " + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n\n"
                    await asyncio.sleep(0.02)
            finally:
                with engine.lock:
                    if client and engine.owner == client:
                        engine.stop("Окно управления отключено")
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/connect")
    def connect(body: ConnectionInput):
        if not body.simulated:
            from serial.tools import list_ports
            if body.port not in {port.device for port in list_ports.comports()}:
                raise ValueError("Выберите доступный COM-порт")
        try:
            engine.connect(**body.model_dump())
        except (OSError, TimeoutError, RuntimeError) as error:
            raise HTTPException(409, str(error))
        return engine.snapshot()

    @app.post("/api/disconnect")
    def disconnect():
        engine.disconnect()
        return engine.snapshot()

    @app.post("/api/motors")
    def motors(body: MotorInput):
        try:
            engine.set_motors(body.enabled)
        except (OSError, TimeoutError, RuntimeError) as error:
            engine.stop("Не удалось переключить моторы")
            raise HTTPException(409, str(error))
        return engine.snapshot()

    @app.post("/api/target")
    def target(body: TargetInput):
        engine.select(body.track_id, body.client)
        return engine.snapshot()

    @app.post("/api/start")
    def start(body: ClientInput):
        try:
            engine.arm(body.client)
        except (OSError, TimeoutError, RuntimeError) as error:
            engine.stop("Не удалось начать наведение")
            raise HTTPException(409, str(error))
        return engine.snapshot()

    @app.post("/api/stop")
    def stop():
        engine.stop()
        return engine.snapshot()

    @app.post("/api/heartbeat")
    def heartbeat(body: ClientInput):
        engine.heartbeat(body.client)
        return {"ok": True}

    @app.post("/api/jog")
    def jog(body: JogInput):
        if abs(body.pitch) + abs(body.yaw) != 1:
            raise ValueError("Выберите одно направление")
        try:
            engine.jog(**body.model_dump())
        except (OSError, TimeoutError, RuntimeError) as error:
            engine.stop("Проверка направления остановлена")
            raise HTTPException(409, str(error))
        return engine.snapshot()

    @app.post("/api/jog/stop")
    def stop_jog(body: ClientInput):
        engine.stop_jog(body.client)
        return {"ok": True}

    @app.post("/api/home")
    def home():
        engine.set_home()
        return engine.snapshot()

    @app.post("/api/settings")
    def settings(body: SettingsInput):
        engine.apply_settings(body.model_dump())
        return engine.snapshot()

    @app.post("/api/calibration")
    def calibration(body: AimInput):
        engine.calibrate_aim(**body.model_dump())
        return engine.snapshot()

    @app.post("/api/camera/start")
    def camera_start(body: CameraInput):
        engine.start_camera(body.source)
        return engine.snapshot()

    @app.post("/api/camera/stop")
    def camera_stop():
        engine.stop_camera()
        return engine.snapshot()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
