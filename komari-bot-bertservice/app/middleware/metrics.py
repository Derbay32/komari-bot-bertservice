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
    ["endpoint", "status"],
)

request_duration = Histogram(
    "bert_scoring_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
)

inference_duration = Histogram(
    "bert_scoring_inference_duration_seconds",
    "Model inference duration in seconds",
    ["method"],  # single, batch
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
    "Current cache size",
)

# 批量处理指标
batch_size_histogram = Histogram(
    "bert_scoring_batch_size",
    "Batch size distribution",
    buckets=[1, 2, 4, 8, 16, 32, 50],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus 指标中间件"""

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
        ).inc()

        request_duration.labels(endpoint=request.url.path).observe(duration)

        return response


# 别名以避免与参数名冲突
batch_size = batch_size_histogram
