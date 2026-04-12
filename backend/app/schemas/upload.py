from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    status: str = "queued"
    estimated_seconds: int = 20
