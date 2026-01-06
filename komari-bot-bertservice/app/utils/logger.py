"""日志配置模块

使用 structlog 进行结构化日志记录
"""

import logging
import sys

import structlog
from structlog.types import EventDict, Processor

from app.config import settings

# 类型别名
type Logger = structlog.stdlib.BoundLogger


def _add_app_context(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """添加应用上下文到日志

    Args:
        _logger: 日志记录器（未使用）
        _method_name: 方法名（未使用）
        event_dict: 事件字典

    Returns:
        更新后的事件字典
    """
    event_dict["app"] = settings.app_name
    event_dict["version"] = settings.app_version
    return event_dict


def _configure_stdlib_logging(level: str) -> None:
    """配置标准库日志

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # 禁用过于详细的日志
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _configure_structlog(level: str) -> list[Processor]:
    """配置 structlog 处理器

    Args:
        level: 日志级别

    Returns:
        处理器列表
    """
    processors: list[Processor] = [
        # 过滤低于指定级别的日志
        structlog.stdlib.filter_by_level,
        # 添加日志级别
        structlog.stdlib.add_log_level,
        # 添加时间戳
        structlog.processors.TimeStamper(fmt="iso"),
        # 添加调用位置
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        # 添加应用上下文
        _add_app_context,
        # 格式化异常
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        # JSON 格式输出（生产环境）
        structlog.processors.JSONRenderer(),
    ]

    # 开发环境使用更易读的格式
    if settings.log_level in ("DEBUG", "INFO"):
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            _add_app_context,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    return processors


def setup_logger(module_name: str) -> Logger:
    """设置并返回日志记录器

    Args:
        module_name: 模块名称

    Returns:
        配置好的日志记录器
    """
    # 配置标准库日志
    _configure_stdlib_logging(settings.log_level)

    # 配置 structlog
    processors = _configure_structlog(settings.log_level)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(module_name)


# 导出全局日志记录器
logger = setup_logger(__name__)
