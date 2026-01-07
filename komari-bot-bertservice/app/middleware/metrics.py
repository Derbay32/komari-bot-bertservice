"""Prometheus 监控指标中间件"""

import time

from fastapi import Request
from prometheus_client import (  # noqa: F401
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# 指标定义
request_count = Counter(
    "bert_scoring_requests_total",
    "Total number of scoring requests",
    ["endpoint", "status", "deployment"],
)

request_duration = Histogram(
    "bert_scoring_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "deployment"],
)

inference_duration = Histogram(
    "bert_scoring_inference_duration_seconds",
    "Model inference duration in seconds",
    ["method", "provider"],  # single, batch + provider
)

# 缓存指标
cache_hits = Counter(
    "bert_scoring_cache_hits_total",
    "Total number of cache hits",
)

cache_misses = Counter(
    "bert_scoring_cache_misses_total",
    "Total number of cache misses",
)

cache_size = Gauge(
    "bert_scoring_cache_size",
    "Current cache size (number of entries)",
)

cache_hit_ratio = Gauge(
    "bert_scoring_cache_hit_ratio",
    "Cache hit ratio (0.0 - 1.0)",
)

# 批量处理指标
batch_size_histogram = Histogram(
    "bert_scoring_batch_size",
    "Batch size distribution",
    buckets=[1, 2, 4, 8, 16, 32, 50],
)

# 错误追踪
inference_errors = Counter(
    "bert_scoring_inference_errors_total",
    "Total number of inference errors",
    ["error_type", "provider"],
)

# 活跃请求
active_requests = Gauge(
    "bert_scoring_active_requests",
    "Number of active requests being processed",
    ["endpoint"],
)

# 评分分布
score_distribution = Histogram(
    "bert_scoring_score_distribution",
    "Distribution of predicted scores",
    ["category"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# 模型加载时间
model_load_duration = Histogram(
    "bert_scoring_model_load_duration_seconds",
    "Model loading duration in seconds",
    ["success"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus 指标中间件"""

    def __init__(self, app, deployment: str | None = None):
        """初始化中间件

        Args:
            app: ASGI 应用
            deployment: 部署环境标签，如果为 None 则从 settings 读取
        """
        super().__init__(app)
        if deployment is None:
            # 延迟导入避免循环依赖
            from app.config import settings

            self.deployment = settings.deployment
        else:
            self.deployment = deployment

    async def dispatch(self, request: Request, call_next):
        """处理请求并记录指标

        Args:
            request: HTTP 请求
            call_next: 下一个中间件/路由处理器

        Returns:
            HTTP 响应
        """
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        # 记录指标
        request_count.labels(
            endpoint=request.url.path,
            status=response.status_code,
            deployment=self.deployment,
        ).inc()

        request_duration.labels(
            endpoint=request.url.path,
            deployment=self.deployment,
        ).observe(duration)

        return response


# 别名以避免与参数名冲突
batch_size = batch_size_histogram
