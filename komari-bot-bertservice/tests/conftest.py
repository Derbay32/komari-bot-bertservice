"""Pytest 配置和共享 fixtures"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.main import app
from app.services.inference_engine import ONNXInferenceEngine

# =============================================================================
# 全局 pytest 配置
# =============================================================================

def pytest_configure(config):
    """Pytest 初始化配置"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )


@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return settings


# =============================================================================
# Mock fixtures
# =============================================================================

@pytest.fixture
def mock_tokenizer():
    """Mock 分词器"""
    tokenizer = MagicMock()

    def mock_encode(text: str) -> dict:
        """模拟编码，返回固定形状的数组"""
        # 返回形状 (1, 128) 的数组
        input_ids = np.random.randint(0, 1000, size=(1, 128), dtype=np.int64)
        attention_mask = np.ones((1, 128), dtype=np.int64)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    tokenizer.encode = mock_encode
    return tokenizer


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX Runtime 会话"""
    session = MagicMock()

    # 模拟推理输出：返回 logits
    # 形状: (1, 3) - 1个样本，3个类别
    mock_logits = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)

    def mock_run(output_names, input_feed):
        """模拟推理运行"""
        return [mock_logits]

    session.run = mock_run

    # 模拟输入输出名称
    session.get_inputs.return_value = [MagicMock(name="input_ids"), MagicMock(name="attention_mask")]
    session.get_outputs.return_value = [MagicMock(name="output")]

    return session


@pytest.fixture
def mock_inference_engine(mock_tokenizer):
    """Mock 推理引擎"""
    engine = MagicMock(spec=ONNXInferenceEngine)

    # 设置默认返回值
    engine.score.return_value = (0.65, "normal", 0.92)
    engine.score_batch.return_value = [
        (0.65, "normal", 0.92),
        (0.15, "low_value", 0.88),
    ]
    engine._cache = {}
    engine.cache_size = 1024

    return engine


# =============================================================================
# FastAPI 测试客户端
# =============================================================================

@pytest.fixture
def test_client():
    """FastAPI 测试客户端"""
    return TestClient(app)


@pytest.fixture
def test_client_with_mock_engine(mock_inference_engine):
    """带有 mock 推理引擎的测试客户端"""
    # 设置 mock 引擎到 app state
    app.state.inference_engine = mock_inference_engine
    return TestClient(app)


# =============================================================================
# 测试数据 fixtures
# =============================================================================

@pytest.fixture
def sample_messages():
    """示例消息数据"""
    return [
        {
            "message": "哈哈哈",
            "context": "",
            "expected_category": "low_value",
        },
        {
            "message": "今天天气真好啊",
            "context": "昨天下雨了",
            "expected_category": "normal",
        },
        {
            "message": "我需要帮助解决这个问题",
            "context": "有人知道怎么处理吗？",
            "expected_category": "interrupt",
        },
    ]


@pytest.fixture
def edge_case_messages():
    """边界情况测试数据"""
    return [
        {"message": "", "context": ""},  # 空消息
        {"message": "a" * 600, "context": ""},  # 超长消息
        {"message": "特殊字符：!@#$%^&*()", "context": "测试上下文"},
        {"message": "Emoji 测试 😂🎉", "context": "表情包"},
    ]


@pytest.fixture
def batch_test_data():
    """批量测试数据"""
    return {
        "messages": [
            {"message": "测试消息1", "context": "上下文1"},
            {"message": "测试消息2", "context": "上下文2"},
            {"message": "哈哈哈", "context": ""},
        ]
    }


# =============================================================================
# 路径 fixtures
# =============================================================================

@pytest.fixture
def mock_model_path(tmp_path):
    """临时模型路径"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "test_model.onnx"
    model_file.write_text("mock model content")
    return str(model_file)


@pytest.fixture
def mock_tokenizer_path(tmp_path):
    """临时分词器路径"""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "config.json").write_text("{}")
    (tokenizer_dir / "vocab.txt").write_text("vocab")
    return str(tokenizer_dir)


# =============================================================================
# 环境变量
# =============================================================================

@pytest.fixture(autouse=True)
def set_test_env_vars(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("MODEL_PATH", "/tmp/test_model.onnx")
    monkeypatch.setenv("TOKENIZER_PATH", "/tmp/test_tokenizer")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")  # 减少测试日志噪音
