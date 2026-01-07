"""配置管理模块

使用环境变量和 pydantic-settings 管理应用配置
"""

import os
from typing import Literal

# 类型别名
type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings:
    """应用配置类

    通过环境变量加载配置，提供默认值
    """

    # 模型配置
    model_path: str = os.getenv("MODEL_PATH", "models/tiny_bert_scoring.onnx")
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "models/tokenizer")
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() == "true"

    # 推理配置
    cache_size: int = int(os.getenv("CACHE_SIZE", "1024"))
    enable_parallel: bool = os.getenv("ENABLE_PARALLEL", "true").lower() == "true"
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "50"))

    # API 配置
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "4"))
    reload: bool = os.getenv("RELOAD", "false").lower() == "true"

    # ONNX Runtime 线程配置（每个 worker）
    onnx_threads: int | None = None  # None 表示自动计算

    # 日志配置
    log_level: LogLevel = os.getenv("LOG_LEVEL", "INFO").upper()  # type: ignore[assignment]

    # CORS 配置（默认为空，需要显式配置）
    # 生产环境必须配置具体的允许来源
    cors_origins: list[str] = (
        os.getenv("CORS_ORIGINS", "").split(",")
        if os.getenv("CORS_ORIGINS")
        else []
    )

    # 应用信息
    app_name: str = "BERT Scoring Service"
    app_version: str = "1.0.0"

    # Gemini API 配置
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
    gemini_timeout: int = int(os.getenv("GEMINI_TIMEOUT", "30"))
    gemini_retry_attempts: int = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "3"))
    gemini_retry_delay: float = float(os.getenv("GEMINI_RETRY_DELAY", "1.0"))

    # Sentry 错误追踪配置
    sentry_dsn: str | None = os.getenv("SENTRY_DSN")
    sentry_environment: str = os.getenv(
        "SENTRY_ENVIRONMENT",
        "production" if os.getenv("RELOAD") != "true" else "development",
    )
    sentry_traces_sample_rate: float = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    sentry_profiles_sample_rate: float = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))

    # 心跳监控配置
    heartbeat_url: str | None = os.getenv("HEARTBEAT_URL")
    heartbeat_interval: int = int(os.getenv("HEARTBEAT_INTERVAL", "30"))

    # 监控配置
    deployment: str = os.getenv("DEPLOYMENT", "production")
    instance_id: str = os.getenv("INSTANCE_ID", "bert-1")

    @property
    def sentry_enabled(self) -> bool:
        """检查 Sentry 是否启用

        只有当 SENTRY_DSN 有值时才启用 Sentry
        """
        return bool(self.sentry_dsn)

    @property
    def heartbeat_enabled(self) -> bool:
        """检查心跳是否启用

        只有当 HEARTBEAT_URL 有值时才启用心跳
        """
        return bool(self.heartbeat_url)

    @property
    def max_length(self) -> int:
        """最大序列长度"""
        return 128


# 全局配置实例
settings = Settings()
