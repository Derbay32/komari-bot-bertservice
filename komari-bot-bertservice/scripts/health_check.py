#!/usr/bin/env python
"""服务健康检查脚本

支持单次检查和持续监控模式，CI/CD 友好的退出码。
"""

import argparse
import logging
import sys
import time
from typing import Literal

import requests

# =============================================================================
# 类型定义
# =============================================================================

type ExitCode = Literal[0, 1, 2]

type CheckResult = dict[str, str | bool | int]


# =============================================================================
# 工具函数
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志

    Args:
        log_level: 日志级别

    Returns:
        配置好的 logger
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def check_health(base_url: str, timeout: int) -> CheckResult:
    """检查健康端点

    Args:
        base_url: 服务基础 URL
        timeout: 请求超时（秒）

    Returns:
        检查结果字典
    """
    logger = logging.getLogger(__name__)
    url = f"{base_url.rstrip('/')}/health"

    try:
        response = requests.get(url, timeout=timeout)

        result: CheckResult = {
            "status": "unhealthy" if response.status_code != 200 else "healthy",
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

        if response.status_code == 200:
            data = response.json()
            result["model_loaded"] = data.get("model_loaded", False)
            result["version"] = data.get("version", "unknown")

            if logger.level <= logging.DEBUG:
                logger.debug(f"Response: {data}")

        return result

    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "status_code": 0,
            "success": False,
        }

    except requests.exceptions.ConnectionError:
        return {
            "status": "connection_error",
            "status_code": 0,
            "success": False,
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "status": "error",
            "status_code": 0,
            "success": False,
        }


def check_model(base_url: str, timeout: int) -> CheckResult:
    """检查模型推理端点

    Args:
        base_url: 服务基础 URL
        timeout: 请求超时（秒）

    Returns:
        检查结果字典
    """
    logger = logging.getLogger(__name__)
    url = f"{base_url.rstrip('/')}/api/v1/score"

    try:
        response = requests.post(
            url,
            json={"message": "测试消息"},
            timeout=timeout,
        )

        result: CheckResult = {
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

        if response.status_code == 200:
            data = response.json()
            result["has_score"] = "score" in data
            result["has_category"] = "category" in data

            if logger.level <= logging.DEBUG:
                logger.debug(f"Response: {data}")

        return result

    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "status_code": 0,
            "success": False,
        }

    except requests.exceptions.ConnectionError:
        return {
            "status": "connection_error",
            "status_code": 0,
            "success": False,
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "status": "error",
            "status_code": 0,
            "success": False,
        }


# =============================================================================
# 主函数
# =============================================================================

def main() -> ExitCode:
    """主函数

    Returns:
        退出码：0=成功，1=健康检查失败，2=网络错误
    """
    parser = argparse.ArgumentParser(
        description="Health check for BERT scoring service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Service base URL",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Request timeout in seconds",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Check interval in seconds (continuous mode)",
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous monitoring mode",
    )

    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Maximum consecutive failures before exit",
    )

    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Also check model inference endpoint",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    consecutive_failures = 0
    check_count = 0

    try:
        while True:
            check_count += 1

            if args.verbose:
                print(f"\n--- Check #{check_count} ---")

            # 健康检查
            health_result = check_health(args.base_url, args.timeout)

            if health_result["success"]:
                consecutive_failures = 0

                # 打印结果
                model_status = "loaded" if health_result.get("model_loaded") else "not loaded"
                version = health_result.get("version", "unknown")
                logger.info(f"Health: OK (model={model_status}, version={version})")

                # 模型检查
                if args.check_model:
                    model_result = check_model(args.base_url, args.timeout)
                    if model_result["success"]:
                        logger.info("Model inference: OK")
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Model inference: {model_result['status']}")

            else:
                consecutive_failures += 1
                logger.warning(f"Health: {health_result['status']}")

            # 检查是否退出
            if consecutive_failures >= args.max_failures:
                logger.error(
                    f"Too many consecutive failures ({consecutive_failures}), exiting"
                )
                return 1

            # 持续模式
            if args.continuous:
                time.sleep(args.interval)
            else:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
