"""速率限制中间件

使用基于 IP 的简单速率限制器，防止 DoS 攻击
"""

import time
from collections import defaultdict
from fastapi import Request, HTTPException
from app.utils.logger import logger


class RateLimiter:
    """基于 IP 的速率限制器

    使用滑动窗口算法实现速率限制
    """

    def __init__(self, requests_per_minute: int = 60):
        """初始化速率限制器

        Args:
            requests_per_minute: 每分钟允许的最大请求数
        """
        self.requests_per_minute = requests_per_minute
        # {ip: [(timestamp1, timestamp2, ...)]}
        self._requests: defaultdict[str, list[float]] = defaultdict(list)
        # 清理过期记录的时间窗口（秒）
        self._window_size = 60

    def _clean_old_requests(self, ip: str, current_time: float) -> None:
        """清理过期的请求记录

        Args:
            ip: 客户端 IP
            current_time: 当前时间戳
        """
        window_start = current_time - self._window_size
        # 保留窗口内的请求
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > window_start
        ]

    def check_rate_limit(self, ip: str) -> bool:
        """检查是否超过速率限制

        Args:
            ip: 客户端 IP

        Returns:
            True 如果未超过限制，False 如果超过限制
        """
        current_time = time.time()

        # 清理过期记录
        self._clean_old_requests(ip, current_time)

        # 检查当前窗口内的请求数
        request_count = len(self._requests[ip])

        if request_count >= self.requests_per_minute:
            logger.warning(
                "rate_limit_exceeded",
                ip=ip,
                request_count=request_count,
                limit=self.requests_per_minute,
            )
            return False

        # 记录本次请求
        self._requests[ip].append(current_time)
        return True


# 全局速率限制器实例
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """获取速率限制器实例（懒加载）

    Returns:
        速率限制器实例
    """
    global _rate_limiter
    if _rate_limiter is None:
        # 从环境变量读取速率限制配置，默认为每分钟 60 次
        import os
        rpm = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        _rate_limiter = RateLimiter(requests_per_minute=rpm)
    return _rate_limiter


async def check_rate_limit_middleware(request: Request) -> None:
    """速率限制中间件检查函数

    在需要速率限制的端点中使用 Depends 注入此函数

    Args:
        request: FastAPI 请求对象

    Raises:
        HTTPException: 如果超过速率限制
    """
    # 获取客户端 IP（考虑代理）
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # 取第一个 IP（客户端 IP）
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    # 检查速率限制
    rate_limiter = get_rate_limiter()
    if not rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {rate_limiter.requests_per_minute} requests per minute.",
            },
        )
