"""错误处理中间件"""

import sentry_sdk

from app.config import settings
from app.utils.logger import logger
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class APIError(Exception):
    """API 错误基类"""

    def __init__(self, message: str, code: str = "api_error", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code


class APIValidationError(APIError):
    """API 验证错误"""

    def __init__(self, message: str):
        super().__init__(message, code="validation_error", status_code=422)


class ModelNotLoadedError(APIError):
    """模型未加载错误"""

    def __init__(self):
        super().__init__(
            "模型未加载，请稍后重试",
            code="model_not_loaded",
            status_code=503,
        )


def add_exception_handlers(app: FastAPI) -> None:
    """添加异常处理器

    Args:
        app: FastAPI 应用实例
    """

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """处理 API 错误"""
        logger.error(
            "api_error",
            code=exc.code,
            message=exc.message,
            path=request.url.path,
        )

        # 捕获到 Sentry（如果启用）
        if settings.sentry_enabled:
            sentry_sdk.capture_exception(exc)

        return JSONResponse(
            status_code=exc.status_code,
            content={"data": None, "error": {"message": exc.message, "code": exc.code}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """处理请求验证错误"""
        errors = exc.errors()
        logger.warning(
            "validation_error",
            errors=errors,
            path=request.url.path,
        )

        # 捕获到 Sentry（如果启用）
        if settings.sentry_enabled:
            sentry_sdk.capture_exception(exc)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "data": None,
                "error": {
                    "message": "请求验证失败",
                    "code": "validation_error",
                    "details": errors,
                },
            },
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """处理未捕获的异常"""
        logger.error(
            "unhandled_error",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
            exc_info=True,
        )

        # 捕获到 Sentry（如果启用）
        if settings.sentry_enabled:
            sentry_sdk.capture_exception(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "data": None,
                "error": {"message": "内部服务器错误", "code": "internal_error"},
            },
        )
