from pydantic import BaseModel, ConfigDict, Field


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SettingsInput(Input):
    confidence: float = Field(default=0.35, ge=0.35, le=0.99)
    max_speed: float = Field(default=6, ge=0.5, le=30)
    gain_yaw: float = Field(default=8, ge=0.5, le=40)
    gain_pitch: float = Field(default=8, ge=0.5, le=40)
    deadband_px: float = Field(default=7, ge=1, le=50)
    acceleration: float = Field(default=25, ge=1, le=120)
    yaw_limit: float = Field(default=45, ge=5, le=120)
    pitch_limit: float = Field(default=30, ge=5, le=75)
    invert_yaw: bool = False
    invert_pitch: bool = False
    aim_x_percent: float = Field(default=0, ge=-40, le=40)
    aim_y_percent: float = Field(default=0, ge=-40, le=40)
    braking_lookahead: float = Field(default=0.28, ge=0, le=0.35)


class ConnectionInput(Input):
    port: str = Field(default="", max_length=128)
    baud: int = Field(default=115200, ge=9600, le=921600)
    protocol: int = Field(default=2, ge=1, le=2)
    simulated: bool = False


class ClientInput(Input):
    client: str = Field(min_length=8, max_length=80)


class JogInput(ClientInput):
    pitch: int = Field(default=0, ge=-1, le=1)
    yaw: int = Field(default=0, ge=-1, le=1)


class TargetInput(Input):
    track_id: int | None = Field(default=None, ge=1)
    client: str | None = Field(default=None, min_length=8, max_length=80)


class CameraInput(Input):
    source: int = Field(default=0, ge=0, le=9)


class MotorInput(Input):
    enabled: bool = Field(strict=True)


class AimInput(Input):
    aim_x_percent: float = Field(ge=-40, le=40)
    aim_y_percent: float = Field(ge=-40, le=40)
