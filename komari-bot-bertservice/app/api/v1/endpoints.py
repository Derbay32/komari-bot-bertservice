"""API v1 端点实现"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import settings
from app.middleware.metrics import (
    batch_size,
    cache_size,
    inference_duration,
)
from app.middleware.rate_limit import check_rate_limit_middleware
from app.models.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreRequest,
    ScoreResponse,
)
from app.services.inference_engine import ONNXInferenceEngine
from app.utils.logger import logger

router = APIRouter()


def get_inference_engine(request: Request) -> ONNXInferenceEngine:
    """依赖注入：获取推理引擎

    Args:
        request: FastAPI 请求对象

    Returns:
        推理引擎实例

    Raises:
        HTTPException: 如果模型未加载
    """
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请稍后重试",
        )
    return engine


@router.post("/score", response_model=ScoreResponse)
async def score_message(
    request: ScoreRequest,
    engine: ONNXInferenceEngine = Depends(get_inference_engine),
    _rate_limit: None = Depends(check_rate_limit_middleware),
) -> ScoreResponse:
    """单条消息评分

    对单条聊天消息进行重要性评分。

    **context 字段说明：**
    - 用于提供对话历史上下文，帮助模型更准确地理解消息
    - 如果提供，模型会将 context 和 message 用 [SEP] 拼接处理
    - 示例：输入 "昨天下雨了 [SEP] 今天天气真好啊"
    - 可以为空或 null，仅使用 message 进行分类

    **请求示例：**
    ```json
    {
        "message": "今天天气真好啊",
        "context": "昨天下雨了",
        "user_id": "user_123",
        "group_id": "group_456"
    }
    ```

    Args:
        request: 评分请求
        engine: 推理引擎

    Returns:
        评分响应
    """
    start_time = time.time()

    try:
        # 调用推理引擎
        with inference_duration.labels(method="single").time():
            score, category, confidence = engine.score(
                request.message,
                request.context or "",
            )

        processing_time = (time.time() - start_time) * 1000  # ms

        # 记录日志
        logger.info(
            "message_scored",
            message_length=len(request.message),
            score=score,
            category=category,
            confidence=confidence,
            processing_time_ms=processing_time,
            user_id=request.user_id,
            group_id=request.group_id,
        )

        # 更新缓存指标
        cache_size.set(len(engine._cache))

        return ScoreResponse(
            score=score,
            category=category,  # type: ignore[arg-type]
            confidence=confidence,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("scoring_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score/batch", response_model=BatchScoreResponse)
async def score_messages_batch(
    request: BatchScoreRequest,
    engine: ONNXInferenceEngine = Depends(get_inference_engine),
    _rate_limit: None = Depends(check_rate_limit_middleware),
) -> BatchScoreResponse:
    """批量消息评分

    对多条聊天消息进行批量重要性评分。

    **context 字段说明：**
    - 每条消息可以单独设置 context（对话历史上下文）
    - 如果提供，模型会将 context 和 message 用 [SEP] 拼接处理
    - 可以为空或 null，仅使用 message 进行分类

    **请求示例：**
    ```json
    {
        "messages": [
            {
                "message": "今天天气真好啊",
                "context": "昨天下雨了"
            },
            {
                "message": "哈哈哈"
            }
        ]
    }
    ```

    Args:
        request: 批量评分请求
        engine: 推理引擎

    Returns:
        批量评分响应
    """
    # 验证批量大小
    if len(request.messages) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "batch_size_exceeded",
                "message": f"Batch size {len(request.messages)} exceeds maximum {settings.max_batch_size}",
                "max_batch_size": settings.max_batch_size,
            },
        )

    start_time = time.time()

    # 记录批量大小
    batch_size.observe(len(request.messages))

    try:
        # 批量推理
        with inference_duration.labels(method="batch").time():
            results = engine.score_batch(
                [
                    {"message": msg.message, "context": msg.context or ""}
                    for msg in request.messages
                ]
            )

        total_time = (time.time() - start_time) * 1000  # ms

        # 构建响应
        response_results = []
        for (score, category, confidence), original in zip(results, request.messages):
            response_results.append(
                ScoreResponse(
                    score=score,
                    category=category,  # type: ignore[arg-type]
                    confidence=confidence,
                    processing_time_ms=0,  # 批量模式不记录单项时间
                )
            )

        # 记录日志
        logger.info(
            "batch_scored",
            batch_size=len(request.messages),
            total_processing_time_ms=total_time,
        )

        # 更新缓存指标
        cache_size.set(len(engine._cache))

        return BatchScoreResponse(
            results=response_results,
            total_processing_time_ms=total_time,
        )

    except Exception as e:
        logger.error("batch_scoring_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
