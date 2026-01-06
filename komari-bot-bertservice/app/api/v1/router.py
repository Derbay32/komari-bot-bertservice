"""API v1 路由模块"""

from fastapi import APIRouter

from app.api.v1 import endpoints

api_router = APIRouter()


# 注册端点
api_router.include_router(endpoints.router, tags=["scoring"])
