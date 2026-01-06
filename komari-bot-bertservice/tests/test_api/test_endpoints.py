"""API 端点集成测试"""

from unittest.mock import patch

# =============================================================================
# POST /api/v1/score 测试
# =============================================================================

class TestScoreEndpoint:
    """单条评分端点测试"""

    def test_score_endpoint_success(self, test_client_with_mock_engine):
        """测试：有效请求返回 200"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score",
            json={
                "message": "今天天气真好啊",
                "context": "昨天下雨了",
                "user_id": "user_123",
                "group_id": "group_456",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "category" in data
        assert "confidence" in data
        assert "processing_time_ms" in data

    def test_score_endpoint_minimal_request(self, test_client_with_mock_engine):
        """测试：最小请求（只有 message）"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score",
            json={"message": "测试消息"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "score" in data

    def test_score_endpoint_missing_message(self, test_client_with_mock_engine):
        """测试：缺少 message 字段返回 422"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score",
            json={"context": "上下文"}
        )

        assert response.status_code == 422

    def test_score_endpoint_message_too_long(self, test_client_with_mock_engine):
        """测试：消息过长返回 422"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score",
            json={"message": "a" * 600}  # 超过 500 限制
        )

        assert response.status_code == 422

    def test_score_endpoint_context_too_long(self, test_client_with_mock_engine):
        """测试：上下文过长返回 422"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score",
            json={"message": "测试", "context": "a" * 600}
        )

        assert response.status_code == 422

    def test_score_endpoint_response_schema(self, test_client_with_mock_engine):
        """测试：响应 schema 验证"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score",
            json={"message": "测试消息"}
        )

        assert response.status_code == 200
        data = response.json()

        # 验证字段存在和类型
        assert isinstance(data["score"], float)
        assert isinstance(data["category"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["processing_time_ms"], float)

        # 验证范围
        assert 0.0 <= data["score"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["category"] in ["low_value", "normal", "interrupt"]

    def test_score_endpoint_optional_fields(self, test_client_with_mock_engine):
        """测试：可选字段正确处理"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score",
            json={
                "message": "测试消息",
                "context": None,
                "user_id": None,
                "group_id": None,
            }
        )

        assert response.status_code == 200


# =============================================================================
# POST /api/v1/score/batch 测试
# =============================================================================

class TestBatchScoreEndpoint:
    """批量评分端点测试"""

    def test_batch_score_endpoint_success(self, test_client_with_mock_engine):
        """测试：有效批量请求返回 200"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score/batch",
            json={
                "messages": [
                    {"message": "消息1", "context": "上下文1"},
                    {"message": "消息2", "context": "上下文2"},
                    {"message": "哈哈哈", "context": ""},
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "total_processing_time_ms" in data
        assert len(data["results"]) == 3

    def test_batch_score_endpoint_empty_messages(self, test_client_with_mock_engine):
        """测试：空消息列表返回 422"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score/batch",
            json={"messages": []}
        )

        assert response.status_code == 422

    def test_batch_score_endpoint_too_many_messages(self, test_client_with_mock_engine):
        """测试：超过 50 条消息返回 422"""
        messages = [{"message": f"测试{i}", "context": ""} for i in range(51)]

        response = test_client_with_mock_engine.post(
            "/api/v1/score/batch",
            json={"messages": messages}
        )

        assert response.status_code == 422

    def test_batch_score_endpoint_single_message(self, test_client_with_mock_engine):
        """测试：单条消息的批量请求"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score/batch",
            json={"messages": [{"message": "测试消息", "context": ""}]}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1

    def test_batch_score_endpoint_response_order(self, test_client_with_mock_engine):
        """测试：批量结果顺序保持一致"""
        # Mock 返回不同的结果
        from app.services.inference_engine import ONNXInferenceEngine

        # 需要 patch 实际的方法来返回不同的结果
        with patch.object(
            ONNXInferenceEngine,
            "score",
            side_effect=[
                (0.1, "low_value", 0.8),
                (0.5, "normal", 0.9),
                (0.9, "interrupt", 0.95),
            ]
        ):
            response = test_client_with_mock_engine.post(
                "/api/v1/score/batch",
                json={
                    "messages": [
                        {"message": "msg1", "context": ""},
                        {"message": "msg2", "context": ""},
                        {"message": "msg3", "context": ""},
                    ]
                }
            )

        assert response.status_code == 200
        data = response.json()

        # 验证顺序
        assert data["results"][0]["score"] == 0.1
        assert data["results"][1]["score"] == 0.5
        assert data["results"][2]["score"] == 0.9

    def test_batch_score_endpoint_mixed_valid_fields(self, test_client_with_mock_engine):
        """测试：混合有/无可选字段的批量请求"""
        response = test_client_with_mock_engine.post(
            "/api/v1/score/batch",
            json={
                "messages": [
                    {"message": "消息1", "context": "ctx", "user_id": "u1"},
                    {"message": "消息2"},  # 最小字段
                    {"message": "消息3", "group_id": "g1"},
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3


# =============================================================================
# GET /health 测试
# =============================================================================

class TestHealthEndpoint:
    """健康检查端点测试"""

    def test_health_endpoint_without_model(self, test_client):
        """测试：没有模型时健康检查"""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert "version" in data

    def test_health_endpoint_with_model(self, test_client_with_mock_engine):
        """测试：有模型时健康检查"""
        response = test_client_with_mock_engine.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_endpoint_response_structure(self, test_client):
        """测试：健康检查响应结构"""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # 验证必需字段
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

        # 验证类型
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["version"], str)


# =============================================================================
# GET /metrics 测试
# =============================================================================

class TestMetricsEndpoint:
    """Prometheus 指标端点测试"""

    def test_metrics_endpoint_returns_text(self, test_client):
        """测试：指标端点返回文本格式"""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_metrics_endpoint_content(self, test_client):
        """测试：指标端点包含 Prometheus 格式数据"""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        content = response.text

        # Prometheus 格式的基本验证
        assert "bert_scoring" in content or len(content) > 0


# =============================================================================
# 错误处理测试
# =============================================================================

class TestErrorHandling:
    """错误处理测试"""

    def test_invalid_json_returns_422(self, test_client):
        """测试：无效 JSON 返回 422"""
        response = test_client.post(
            "/api/v1/score",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_method_not_allowed(self, test_client):
        """测试：不允许的 HTTP 方法"""
        response = test_client.get("/api/v1/score")

        assert response.status_code == 405

    def test_invalid_endpoint_returns_404(self, test_client):
        """测试：不存在的端点返回 404"""
        response = test_client.get("/api/v1/invalid")

        assert response.status_code == 404


# =============================================================================
# 模型未加载测试
# =============================================================================

class TestModelNotLoaded:
    """模型未加载场景测试"""

    def test_score_without_model_returns_503(self, test_client):
        """测试：没有模型时评分返回 503"""
        # 确保 app 没有模型
        if hasattr(test_client.app.state, "inference_engine"):
            delattr(test_client.app.state, "inference_engine")

        response = test_client.post(
            "/api/v1/score",
            json={"message": "测试消息"}
        )

        assert response.status_code == 503

    def test_batch_score_without_model_returns_503(self, test_client):
        """测试：没有模型时批量评分返回 503"""
        if hasattr(test_client.app.state, "inference_engine"):
            delattr(test_client.app.state, "inference_engine")

        response = test_client.post(
            "/api/v1/score/batch",
            json={"messages": [{"message": "测试", "context": ""}]}
        )

        assert response.status_code == 503
