from pydantic import BaseModel, Field


class ReprocessParams(BaseModel):
    variant: str = "standard"
    depth_intensity: float = Field(default=1.0, ge=0.0, le=2.0)
    blur_radius: float = Field(default=1.0, ge=0.0, le=10.0)
    contrast: float = Field(default=1.1, ge=0.0, le=3.0)
    edge_enhancement: float = Field(default=0.3, ge=0.0, le=1.0)
    invert: bool = False
