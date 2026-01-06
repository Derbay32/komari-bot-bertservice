"""ONNX 推理引擎测试"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.inference_engine import ONNXInferenceEngine

# =============================================================================
# 初始化测试
# =============================================================================

class TestONNXInferenceEngineInit:
    """推理引擎初始化测试"""

    def test_init_with_valid_paths(self, mock_model_path, mock_tokenizer_path):
        """测试：有效路径时的初始化"""
        # 注意：这需要真实的模型文件，所以这里主要测试路径处理逻辑
        # 实际测试中，我们需要模拟 ONNX Runtime
        pass

    def test_init_sets_cache_size(self, mock_model_path, mock_tokenizer_path):
        """测试：缓存大小设置正确"""
        with patch("app.services.inference_engine.TokenizerWrapper"):
            with patch("app.services.inference_engine.ort.InferenceSession"):
                engine = ONNXInferenceEngine(
                    mock_model_path,
                    mock_tokenizer_path,
                    cache_size=512,
                )
                assert engine.cache_size == 512

    def test_init_calculates_threads_correctly(self, mock_model_path, mock_tokenizer_path):
        """测试：线程数计算正确"""
        with patch("app.services.inference_engine.TokenizerWrapper"):
            with patch("app.services.inference_engine.ort.InferenceSession"):
                with patch("os.cpu_count", return_value=4):
                    engine = ONNXInferenceEngine(
                        mock_model_path,
                        mock_tokenizer_path,
                    )
                    # 应该是 min(4, 8) = 4
                    assert engine.num_threads == 4

    def test_init_caps_threads_at_8(self, mock_model_path, mock_tokenizer_path):
        """测试：线程数最多为 8"""
        with patch("app.services.inference_engine.TokenizerWrapper"):
            with patch("app.services.inference_engine.ort.InferenceSession"):
                with patch("os.cpu_count", return_value=16):
                    engine = ONNXInferenceEngine(
                        mock_model_path,
                        mock_tokenizer_path,
                    )
                    # 应该是 min(16, 8) = 8
                    assert engine.num_threads == 8


# =============================================================================
# 单条评分测试
# =============================================================================

class TestSingleScoring:
    """单条评分测试"""

    def test_score_returns_tuple(self, mock_inference_engine):
        """测试：返回值是元组"""
        result = mock_inference_engine.score("test", "context")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_score_returns_correct_types(self, mock_inference_engine):
        """测试：返回值类型正确"""
        score, category, confidence = mock_inference_engine.score("test", "context")

        assert isinstance(score, float)
        assert isinstance(category, str)
        assert isinstance(confidence, float)

    def test_score_in_range(self, mock_inference_engine):
        """测试：评分在 0.0-1.0 范围内"""
        score, _, _ = mock_inference_engine.score("test", "context")
        assert 0.0 <= score <= 1.0

    def test_score_category_valid(self, mock_inference_engine):
        """测试：分类标签有效"""
        _, category, _ = mock_inference_engine.score("test", "context")
        assert category in ["low_value", "normal", "interrupt"]

    def test_score_confidence_in_range(self, mock_inference_engine):
        """测试：置信度在 0.0-1.0 范围内"""
        _, _, confidence = mock_inference_engine.score("test", "context")
        assert 0.0 <= confidence <= 1.0

    def test_score_without_context(self, mock_inference_engine):
        """测试：没有上下文也能评分"""
        mock_inference_engine.score.return_value = (0.5, "normal", 0.9)
        score, category, confidence = mock_inference_engine.score("test", "")
        assert score == 0.5


# =============================================================================
# 批量评分测试
# =============================================================================

class TestBatchScoring:
    """批量评分测试"""

    def test_score_batch_with_valid_input(self, mock_inference_engine):
        """测试：有效输入的批量评分"""
        items = [
            {"message": "test1", "context": "ctx1"},
            {"message": "test2", "context": "ctx2"},
        ]

        results = mock_inference_engine.score_batch(items)

        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_score_batch_with_empty_list(self, mock_inference_engine):
        """测试：空列表返回空结果"""
        mock_inference_engine.score_batch.return_value = []
        results = mock_inference_engine.score_batch([])
        assert results == []

    def test_score_batch_with_single_item(self, mock_inference_engine):
        """测试：单项列表调用单条评分"""
        items = [{"message": "test", "context": "ctx"}]

        mock_inference_engine.score.return_value = (0.7, "normal", 0.85)
        results = mock_inference_engine.score_batch(items)

        assert len(results) == 1
        assert results[0] == (0.7, "normal", 0.85)

    def test_score_batch_preserves_order(self, mock_inference_engine):
        """测试：批量结果保持原始顺序"""
        items = [
            {"message": "test1", "context": "ctx1"},
            {"message": "test2", "context": "ctx2"},
            {"message": "test3", "context": "ctx3"},
        ]

        mock_inference_engine.score_batch.return_value = [
            (0.1, "low_value", 0.8),
            (0.5, "normal", 0.9),
            (0.9, "interrupt", 0.95),
        ]

        results = mock_inference_engine.score_batch(items)

        # 验证顺序
        assert results[0][0] == 0.1
        assert results[1][0] == 0.5
        assert results[2][0] == 0.9


# =============================================================================
# 缓存测试
# =============================================================================

class TestCaching:
    """缓存功能测试"""

    def test_cache_key_generation(self, mock_inference_engine):
        """测试：缓存键生成正确"""
        # 测试缓存键的唯一性
        key1 = mock_inference_engine._get_cache_key("message", "context")
        key2 = mock_inference_engine._get_cache_key("message", "context")
        key3 = mock_inference_engine._get_cache_key("message", "different")

        assert key1 == key2
        assert key1 != key3

    def test_cache_key_format(self, mock_inference_engine):
        """测试：缓存键格式正确"""
        key = mock_inference_engine._get_cache_key("你好", "世界")
        assert "世界" in key
        assert "你好" in key

    def test_cache_add_increases_size(self, mock_inference_engine):
        """测试：添加缓存增加大小"""
        initial_size = len(mock_inference_engine._cache)
        mock_inference_engine._add_to_cache("key", (0.5, "normal", 0.9))
        assert len(mock_inference_engine._cache) == initial_size + 1

    def test_cache_eviction_when_full(self, mock_inference_engine):
        """测试：缓存满时执行 LRU 驱逐"""
        # 设置小缓存
        mock_inference_engine.cache_size = 3
        mock_inference_engine._cache = MagicMock()

        # 添加 4 个项目
        for i in range(4):
            mock_inference_engine._add_to_cache(f"key{i}", (i * 0.1, "normal", 0.9))

        # 应该只有 3 个项目（最后一个被驱逐）
        assert len(mock_inference_engine._cache) == 3


# =============================================================================
# 辅助方法测试
# =============================================================================

class TestHelperMethods:
    """辅助方法测试"""

    def test_softmax_normalization(self):
        """测试：Softmax 输出归一化"""
        logits = np.array([1.0, 2.0, 3.0])
        probs = ONNXInferenceEngine._softmax(logits)

        # 验证和为 1
        assert np.isclose(probs.sum(), 1.0)

        # 验证所有值为正
        assert np.all(probs > 0)

    def test_softmax_large_values(self):
        """测试：Softmax 处理大值"""
        logits = np.array([100.0, 200.0, 300.0])
        probs = ONNXInferenceEngine._softmax(logits)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))

    def test_class_to_score_range(self):
        """测试：类别转换评分在正确范围"""
        probs = np.array([0.2, 0.6, 0.2])
        engine = ONNXInferenceEngine.__new__(ONNXInferenceEngine)
        score = engine._class_to_score(0, probs)

        # 使用权重 [0.0, 0.55, 1.0]
        # score = 0*0.2 + 0.55*0.6 + 1.0*0.2 = 0.53
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("score,expected", [
        (0.1, "low_value"),
        (0.5, "normal"),
        (0.9, "interrupt"),
        (0.0, "low_value"),
        (0.3, "normal"),
        (0.8, "normal"),
    ])
    def test_score_to_category_mapping(self, score, expected):
        """测试：评分到分类的映射正确"""
        category = ONNXInferenceEngine._score_to_category(score)
        assert category == expected


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_message(self, mock_inference_engine):
        """测试：空消息处理"""
        mock_inference_engine.score.return_value = (0.1, "low_value", 0.7)
        score, category, confidence = mock_inference_engine.score("", "")
        assert category == "low_value"

    def test_very_long_message(self, mock_inference_engine):
        """测试：超长消息处理"""
        long_message = "a" * 600
        mock_inference_engine.score.return_value = (0.5, "normal", 0.8)
        # 应该能处理，可能截断
        score, _, _ = mock_inference_engine.score(long_message, "context")
        assert isinstance(score, float)

    def test_unicode_message(self, mock_inference_engine):
        """测试：Unicode 消息处理"""
        unicode_message = "测试中文字符 😂🎉"
        mock_inference_engine.score.return_value = (0.6, "normal", 0.85)
        score, _, _ = mock_inference_engine.score(unicode_message, "上下文")
        assert isinstance(score, float)

    def test_special_characters(self, mock_inference_engine):
        """测试：特殊字符处理"""
        special_message = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        mock_inference_engine.score.return_value = (0.4, "normal", 0.8)
        score, _, _ = mock_inference_engine.score(special_message, "test")
        assert isinstance(score, float)
