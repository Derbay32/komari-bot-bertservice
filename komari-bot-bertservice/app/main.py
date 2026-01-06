"""FastAPI 主应用模块"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.api.v1.router import api_router
from app.config import settings
from app.middleware.error_handler import add_exception_handlers
from app.middleware.metrics import REGISTRY, MetricsMiddleware, generate_latest
from app.services.inference_engine import ONNXInferenceEngine
from app.utils.logger import logger

# 类型别名
type AppState = FastAPI.state  # type: ignore[attr-defined]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期

    启动时加载模型，关闭时清理资源

    Args:
        app: FastAPI 应用实例
    """
    # 启动
    logger.info("application_starting", version=settings.app_version)

    # 检查模型文件是否存在
    model_path = Path(settings.model_path).resolve()
    tokenizer_path = Path(settings.tokenizer_path).resolve()

    # 路径遍历防护：确保路径在允许的目录内
    allowed_model_dirs = {
        Path("models").resolve(),
        Path("/app/models").resolve(),
    }

    # 验证模型路径
    model_path_valid = any(
        str(model_path).startswith(str(allowed_dir))
        for allowed_dir in allowed_model_dirs
    )
    if not model_path_valid:
        logger.critical(
            "invalid_model_path",
            model_path=str(model_path),
            allowed_dirs=[str(d) for d in allowed_model_dirs],
            message="模型路径不在允许的目录内",
        )
        raise ValueError(
            f"Model path must be within allowed directories: {model_path}"
        )

    # 验证分词器路径
    tokenizer_path_valid = any(
        str(tokenizer_path).startswith(str(allowed_dir))
        for allowed_dir in allowed_model_dirs
    )
    if not tokenizer_path_valid:
        logger.critical(
            "invalid_tokenizer_path",
            tokenizer_path=str(tokenizer_path),
            allowed_dirs=[str(d) for d in allowed_model_dirs],
            message="分词器路径不在允许的目录内",
        )
        raise ValueError(
            f"Tokenizer path must be within allowed directories: {tokenizer_path}"
        )

    # 模型文件缺失时快速失败
    if not model_path.exists():
        logger.critical(
            "model_file_not_found",
            model_path=settings.model_path,
            message="模型文件不存在，无法启动服务",
        )
        raise RuntimeError(
            f"Required model file not found: {settings.model_path}. "
            "Service cannot start without model."
        )

    if not tokenizer_path.exists():
        logger.critical(
            "tokenizer_not_found",
            tokenizer_path=settings.tokenizer_path,
            message="分词器不存在，无法启动服务",
        )
        raise RuntimeError(
            f"Required tokenizer not found: {settings.tokenizer_path}. "
            "Service cannot start without tokenizer."
        )

    # 加载推理引擎
    logger.info("loading_model", model_path=str(model_path))
    try:
        app.state.inference_engine = ONNXInferenceEngine(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            use_gpu=settings.use_gpu,
            cache_size=settings.cache_size,
            enable_parallel=settings.enable_parallel,
            workers=settings.workers,
            onnx_threads=settings.onnx_threads,
        )
        logger.info(
            "model_loaded",
            model_path=str(model_path),
            threads=app.state.inference_engine.num_threads,
        )
    except Exception as e:
        logger.critical(
            "model_loading_failed",
            model_path=str(model_path),
            error=str(e),
            error_type=type(e).__name__,
        )
        raise RuntimeError(f"Failed to load inference engine: {e}") from e

    yield

    # 关闭
    logger.info("application_shutting_down")

    # 清理推理引擎资源
    if hasattr(app.state, "inference_engine"):
        try:
            # 释放 ONNX 会话资源
            del app.state.inference_engine
            logger.info("inference_engine_cleaned_up")
        except Exception as e:
            logger.warning(
                "inference_engine_cleanup_error",
                error=str(e),
            )


# 创建应用
app = FastAPI(
    title=settings.app_name,
    description="聊天消息重要性评分服务",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 指标中间件
app.add_middleware(MetricsMiddleware)

# 异常处理
add_exception_handlers(app)

# 路由
app.include_router(api_router, prefix="/api/v1")


# 健康检查
@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str | bool | float]:
    """健康检查端点（包含模型就绪状态验证）

    Returns:
        包含服务状态和模型就绪状态的字典
    """
    model_loaded = hasattr(app.state, "inference_engine")
    model_ready = False

    if model_loaded:
        try:
            # 执行推理测试验证模型可用性
            import time
            start = time.time()
            score, category, confidence = app.state.inference_engine.score(
                "健康检查测试", "上下文"
            )
            inference_time = (time.time() - start) * 1000

            # 验证返回值有效性
            model_ready = (
                isinstance(score, float) and
                0.0 <= score <= 1.0 and
                category in ["low_value", "normal", "interrupt"] and
                isinstance(confidence, float) and
                0.0 <= confidence <= 1.0
            )

            if model_ready:
                return {
                    "status": "healthy",
                    "model_loaded": True,
                    "model_ready": True,
                    "inference_test_passed": True,
                    "inference_time_ms": round(inference_time, 2),
                    "version": settings.app_version,
                }
        except Exception as e:
            logger.warning(
                "health_check_inference_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "status": "unhealthy",
                "model_loaded": True,
                "model_ready": False,
                "error": "Model inference test failed",
                "version": settings.app_version,
            }

    return {
        "status": "unhealthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_ready": False,
        "version": settings.app_version,
    }


# Prometheus 指标
@app.get("/metrics", tags=["monitoring"])
async def metrics() -> Response:
    """Prometheus 指标端点

    Returns:
        Prometheus 格式的指标数据
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain",
    )
