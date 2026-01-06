"""Pydantic 数据模型

定义请求和响应的数据结构
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# 类型别名
type ScoreCategory = Literal["low_value", "normal", "interrupt"]


class ScoreRequest(BaseModel):
    """评分请求模型

    请求示例：
        ```json
        {
            "message": "今天天气真好啊",
            "context": "昨天下雨了",
            "user_id": "user_123",
            "group_id": "group_456"
        }
        ```

    关于 context 字段：
        - 用于提供对话历史上下文，帮助模型更准确地理解消息
        - 如果提供，模型会将 context 和 message 用 [SEP] 拼接
        - 示例：输入 "昨天下雨了 [SEP] 今天天气真好啊"
        - 可以为空或 null，仅使用 message 进行分类
    """

    message: str = Field(..., min_length=1, max_length=500, description="待评分的消息")
    context: str | None = Field(
        default=None,
        max_length=500,
        description="对话上下文（可选，为空则仅使用 message 分类）"
    )
    # 防止日志注入：只允许字母、数字、下划线和连字符
    user_id: str | None = Field(
        default=None,
        max_length=64,
        pattern="^[a-zA-Z0-9_-]*$",
        description="用户 ID（仅字母、数字、下划线、连字符）"
    )
    group_id: str | None = Field(
        default=None,
        max_length=64,
        pattern="^[a-zA-Z0-9_-]*$",
        description="群组 ID（仅字母、数字、下划线、连字符）"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "今天天气真好啊",
                    "context": "昨天下雨了",
                    "user_id": "user_123",
                    "group_id": "group_456",
                }
            ]
        }
    }


class ScoreResponse(BaseModel):
    """评分响应模型"""

    score: float = Field(..., ge=0.0, le=1.0, description="连续评分值 (0.0-1.0)")
    category: ScoreCategory = Field(..., description="分类标签")
    confidence: float = Field(..., ge=0.0, le=1.0, description="模型置信度")
    processing_time_ms: float = Field(..., description="处理耗时（毫秒）")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "score": 0.65,
                    "category": "normal",
                    "confidence": 0.92,
                    "processing_time_ms": 45.3,
                }
            ]
        }
    }


class BatchScoreRequest(BaseModel):
    """批量评分请求模型"""

    messages: list[ScoreRequest] = Field(
        ..., min_length=1, max_length=50, description="消息列表"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_length(cls, v: list[ScoreRequest]) -> list[ScoreRequest]:
        """验证消息列表长度

        Args:
            v: 消息列表

        Returns:
            验证后的消息列表

        Raises:
            ValueError: 如果超过最大长度
        """
        if len(v) > 50:
            raise ValueError("最多支持 50 条消息批量评分")
        return v


class BatchScoreResponse(BaseModel):
    """批量评分响应模型"""

    results: list[ScoreResponse] = Field(..., description="评分结果列表")
    total_processing_time_ms: float = Field(..., description="总处理耗时（毫秒）")


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    version: str = Field(..., description="服务版本")
