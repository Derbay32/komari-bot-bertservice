"""打分分类日志服务

每天生成一个独立的 JSONL 格式日志文件，记录每一次打分的详细信息。
支持：
- 按日期分割文件（score_log_YYYY-MM-DD.jsonl）
- 异步安全写入（asyncio.Lock）
- 自动清理超过保留期的旧文件
"""

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiofiles

from app.utils.logger import logger


class ScoreLogger:
    """打分分类日志记录器

    将每次打分结果以 JSONL 格式写入按日期分割的文件中，
    并在初始化时清理超过保留期的旧日志文件。

    Attributes:
        log_dir: 日志文件存储目录
        retention_days: 日志保留天数
    """

    def __init__(self, log_dir: str | Path, retention_days: int = 180) -> None:
        """初始化日志记录器

        Args:
            log_dir: 日志文件存储目录（不存在时自动创建）
            retention_days: 日志保留天数，超过此天数的文件将被删除
        """
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self._lock = asyncio.Lock()
        # 确保目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "score_logger_initialized",
            log_dir=str(self.log_dir),
            retention_days=self.retention_days,
        )

    def _get_log_path(self, target_date: date | None = None) -> Path:
        """获取指定日期的日志文件路径

        Args:
            target_date: 目标日期，None 表示今天

        Returns:
            日志文件的完整路径
        """
        d = target_date or date.today()
        return self.log_dir / f"score_log_{d.isoformat()}.jsonl"

    async def log(self, record: dict[str, Any]) -> None:
        """写入一条打分日志记录

        使用 asyncio.Lock 保证并发安全，以 JSONL 格式追加写入。

        Args:
            record: 日志记录字典，应包含 timestamp、score、category 等字段
        """
        line = json.dumps(record, ensure_ascii=False) + "\n"
        log_path = self._get_log_path()

        async with self._lock:
            try:
                async with aiofiles.open(log_path, mode="a", encoding="utf-8") as f:
                    await f.write(line)
            except OSError as e:
                logger.error(
                    "score_log_write_failed",
                    log_path=str(log_path),
                    error=str(e),
                )

    async def cleanup_old_logs(self) -> None:
        """清理超过保留期的旧日志文件

        扫描日志目录，删除文件名中日期超过 retention_days 的 JSONL 文件。
        文件命名格式须为 score_log_YYYY-MM-DD.jsonl，否则跳过。
        """
        cutoff_date = date.today() - timedelta(days=self.retention_days)
        deleted_count = 0

        for log_file in self.log_dir.glob("score_log_*.jsonl"):
            # 从文件名中解析日期
            stem = log_file.stem  # e.g. "score_log_2025-08-27"
            date_str = stem.removeprefix("score_log_")
            try:
                file_date = date.fromisoformat(date_str)
            except ValueError:
                # 文件名不符合格式，跳过
                continue

            if file_date < cutoff_date:
                try:
                    log_file.unlink()
                    deleted_count += 1
                    logger.debug(
                        "score_log_deleted",
                        file=log_file.name,
                        file_date=date_str,
                        cutoff_date=cutoff_date.isoformat(),
                    )
                except OSError as e:
                    logger.warning(
                        "score_log_delete_failed",
                        file=log_file.name,
                        error=str(e),
                    )

        logger.info(
            "score_log_cleanup_done",
            deleted_count=deleted_count,
            cutoff_date=cutoff_date.isoformat(),
            retention_days=self.retention_days,
        )

    async def close(self) -> None:
        """关闭日志记录器（当前无持久句柄，预留给未来扩展）"""
        logger.info("score_logger_closed")


def make_score_record(
    *,
    message: str,
    message_length: int,
    score: float,
    category: str,
    confidence: float,
    processing_time_ms: float,
    source: str,
    user_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """构造标准的打分日志记录字典

    Args:
        message: 原始消息文本
        message_length: 消息字符长度
        score: 连续评分值 (0.0-1.0)
        category: 分类标签（low_value / normal / interrupt）
        confidence: 模型置信度
        processing_time_ms: 处理耗时（毫秒）
        source: 来源，"single" 或 "batch"
        user_id: 用户 ID（可选）
        group_id: 群组 ID（可选）

    Returns:
        可直接写入日志的字典
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "message_length": message_length,
        "score": score,
        "category": category,
        "confidence": confidence,
        "processing_time_ms": processing_time_ms,
        "user_id": user_id,
        "group_id": group_id,
        "source": source,
    }
