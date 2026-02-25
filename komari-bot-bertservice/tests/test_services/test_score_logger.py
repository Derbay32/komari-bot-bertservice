"""ScoreLogger 单元测试

测试覆盖：
- 日志写入（文件创建、内容格式、JSONL 规范）
- 旧文件清理（180 天保留逻辑）
- 并发写入安全性
"""

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """临时日志目录"""
    d = tmp_path / "score_logs"
    d.mkdir()
    return d


@pytest.fixture
async def score_logger(log_dir: Path):
    """ScoreLogger 实例"""
    from app.services.score_logger import ScoreLogger

    sl = ScoreLogger(log_dir=log_dir, retention_days=180)
    yield sl
    await sl.close()


def make_record(**kwargs) -> dict:
    """构造测试用日志记录"""
    from app.services.score_logger import make_score_record

    defaults = dict(
        message_length=10,
        score=0.65,
        category="normal",
        confidence=0.92,
        processing_time_ms=42.0,
        source="single",
        user_id="u1",
        group_id="g1",
    )
    defaults.update(kwargs)
    return make_score_record(**defaults)


# =============================================================================
# 初始化测试（同步）
# =============================================================================

class TestScoreLoggerInit:
    """ScoreLogger 初始化测试"""

    def test_creates_log_dir(self, tmp_path: Path) -> None:
        """测试：不存在的目录会被自动创建"""
        from app.services.score_logger import ScoreLogger

        new_dir = tmp_path / "nested" / "logs"
        assert not new_dir.exists()
        ScoreLogger(log_dir=new_dir)
        assert new_dir.is_dir()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """测试：接受字符串路径"""
        from app.services.score_logger import ScoreLogger

        sl = ScoreLogger(log_dir=str(tmp_path / "str_logs"))
        assert sl.log_dir.is_dir()

    def test_default_retention_days(self, tmp_path: Path) -> None:
        """测试：默认保留 180 天"""
        from app.services.score_logger import ScoreLogger

        sl = ScoreLogger(log_dir=tmp_path)
        assert sl.retention_days == 180

    def test_custom_retention_days(self, tmp_path: Path) -> None:
        """测试：可自定义保留天数"""
        from app.services.score_logger import ScoreLogger

        sl = ScoreLogger(log_dir=tmp_path, retention_days=30)
        assert sl.retention_days == 30


# =============================================================================
# 写入测试（异步）
# =============================================================================

class TestScoreLoggerWrite:
    """日志写入功能测试"""

    async def test_log_creates_file(self, score_logger, log_dir: Path) -> None:
        """测试：写入一条日志后，文件被创建"""
        await score_logger.log(make_record())

        today_str = date.today().isoformat()
        log_file = log_dir / f"score_log_{today_str}.jsonl"
        assert log_file.exists()

    async def test_log_writes_valid_json(self, score_logger, log_dir: Path) -> None:
        """测试：写入的内容是合法 JSON"""
        await score_logger.log(make_record(score=0.9, category="interrupt"))

        today_str = date.today().isoformat()
        log_file = log_dir / f"score_log_{today_str}.jsonl"
        content = log_file.read_text(encoding="utf-8").strip()
        record = json.loads(content)

        assert record["score"] == 0.9
        assert record["category"] == "interrupt"

    async def test_log_multiple_records_one_per_line(self, score_logger, log_dir: Path) -> None:
        """测试：多条记录各占一行（JSONL 格式）"""
        for i in range(5):
            await score_logger.log(make_record(score=float(i) * 0.1))

        today_str = date.today().isoformat()
        log_file = log_dir / f"score_log_{today_str}.jsonl"
        lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 5

        for line in lines:
            obj = json.loads(line)
            assert "score" in obj

    async def test_log_record_has_required_fields(self, score_logger, log_dir: Path) -> None:
        """测试：日志记录包含所有必须字段"""
        record = make_record(source="batch", user_id="u42", group_id="g99")
        await score_logger.log(record)

        today_str = date.today().isoformat()
        log_file = log_dir / f"score_log_{today_str}.jsonl"
        content = log_file.read_text(encoding="utf-8").strip()
        obj = json.loads(content)

        required_fields = {
            "timestamp", "message_length", "score", "category",
            "confidence", "processing_time_ms", "source",
        }
        assert required_fields.issubset(obj.keys())
        assert obj["source"] == "batch"
        assert obj["user_id"] == "u42"
        assert obj["group_id"] == "g99"

    async def test_log_appends_to_existing_file(self, score_logger, log_dir: Path) -> None:
        """测试：追加写入不覆盖现有行"""
        await score_logger.log(make_record(score=0.1))
        await score_logger.log(make_record(score=0.9))

        today_str = date.today().isoformat()
        log_file = log_dir / f"score_log_{today_str}.jsonl"
        lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2

        scores = [json.loads(ln)["score"] for ln in lines]
        assert 0.1 in scores
        assert 0.9 in scores


# =============================================================================
# 文件命名测试（同步）
# =============================================================================

class TestScoreLoggerFilenaming:
    """日志文件命名测试"""

    def test_get_log_path_today(self, score_logger) -> None:
        """测试：默认返回今天的日志路径"""
        path = score_logger._get_log_path()
        assert path.name == f"score_log_{date.today().isoformat()}.jsonl"

    def test_get_log_path_specific_date(self, score_logger) -> None:
        """测试：指定日期返回对应路径"""
        target = date(2025, 8, 27)
        path = score_logger._get_log_path(target)
        assert path.name == "score_log_2025-08-27.jsonl"


# =============================================================================
# 清理测试（异步）
# =============================================================================

class TestScoreLoggerCleanup:
    """旧文件清理功能测试"""

    def _create_log_file(self, log_dir: Path, target_date: date) -> Path:
        """在指定日期创建日志文件"""
        log_file = log_dir / f"score_log_{target_date.isoformat()}.jsonl"
        log_file.write_text('{"score": 0.5}\n', encoding="utf-8")
        return log_file

    async def test_cleanup_deletes_old_files(self, score_logger, log_dir: Path) -> None:
        """测试：超过保留期的文件被删除"""
        old_date = date.today() - timedelta(days=181)
        old_file = self._create_log_file(log_dir, old_date)
        assert old_file.exists()

        await score_logger.cleanup_old_logs()
        assert not old_file.exists()

    async def test_cleanup_keeps_recent_files(self, score_logger, log_dir: Path) -> None:
        """测试：保留期内的文件不被删除"""
        recent_file = self._create_log_file(log_dir, date.today() - timedelta(days=10))
        today_file = self._create_log_file(log_dir, date.today())

        await score_logger.cleanup_old_logs()

        assert recent_file.exists()
        assert today_file.exists()

    async def test_cleanup_keeps_boundary_file(self, score_logger, log_dir: Path) -> None:
        """测试：恰好等于保留天数边界的文件被保留"""
        boundary_date = date.today() - timedelta(days=180)
        boundary_file = self._create_log_file(log_dir, boundary_date)

        await score_logger.cleanup_old_logs()
        assert boundary_file.exists()

    async def test_cleanup_ignores_unmatched_files(self, score_logger, log_dir: Path) -> None:
        """测试：不匹配命名格式的文件不受影响"""
        other_file = log_dir / "random_file.txt"
        other_file.write_text("data", encoding="utf-8")

        await score_logger.cleanup_old_logs()
        assert other_file.exists()

    async def test_cleanup_mixed_old_and_new(self, score_logger, log_dir: Path) -> None:
        """测试：同时存在新旧文件时，只删除旧文件"""
        old_file = self._create_log_file(log_dir, date.today() - timedelta(days=200))
        new_file = self._create_log_file(log_dir, date.today() - timedelta(days=5))

        await score_logger.cleanup_old_logs()

        assert not old_file.exists()
        assert new_file.exists()


# =============================================================================
# 并发写入测试（异步）
# =============================================================================

class TestScoreLoggerConcurrency:
    """并发写入安全性测试"""

    async def test_concurrent_writes_no_data_loss(self, score_logger, log_dir: Path) -> None:
        """测试：并发写入不丢失任何记录"""
        n = 50
        tasks = [score_logger.log(make_record(score=float(i) / n)) for i in range(n)]
        await asyncio.gather(*tasks)

        today_str = date.today().isoformat()
        log_file = log_dir / f"score_log_{today_str}.jsonl"
        lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == n

    async def test_concurrent_writes_all_valid_json(self, score_logger, log_dir: Path) -> None:
        """测试：并发写入后每一行都是合法 JSON（无损坏行）"""
        tasks = [score_logger.log(make_record()) for _ in range(30)]
        await asyncio.gather(*tasks)

        today_str = date.today().isoformat()
        log_file = log_dir / f"score_log_{today_str}.jsonl"
        lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

        for line in lines:
            obj = json.loads(line)
            assert "score" in obj
