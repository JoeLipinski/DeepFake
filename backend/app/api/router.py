from fastapi import APIRouter
from app.api import health, upload, jobs, export

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(upload.router, tags=["upload"])
api_router.include_router(jobs.router, tags=["jobs"])
api_router.include_router(export.router, tags=["export"])
