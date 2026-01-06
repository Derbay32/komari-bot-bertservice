"""测试数据常量和工具函数"""

from typing import Literal

# =============================================================================
# 测试消息类别常量
# =============================================================================

type ScoreCategory = Literal["low_value", "normal", "interrupt"]


# =============================================================================
# 单条评分测试数据
# =============================================================================

# 低价值消息示例
LOW_VALUE_MESSAGES = [
    {"message": "哈哈哈", "context": "", "expected_category": "low_value"},
    {"message": "233", "context": "", "expected_category": "low_value"},
    {"message": "笑死我了", "context": "", "expected_category": "low_value"},
    {"message": "啊啊啊", "context": "", "expected_category": "low_value"},
    {"message": "www", "context": "", "expected_category": "low_value"},
]

# 正常消息示例
NORMAL_MESSAGES = [
    {
        "message": "今天天气真好啊",
        "context": "昨天下雨了",
        "expected_category": "normal",
    },
    {
        "message": "我觉得这个问题可以这样解决",
        "context": "刚才讨论的bug",
        "expected_category": "normal",
    },
    {
        "message": "大家吃饭了吗",
        "context": "中午了",
        "expected_category": "normal",
    },
    {
        "message": "这个功能我昨天实现了",
        "context": "用户认证",
        "expected_category": "normal",
    },
]

# 打断性消息示例
INTERRUPT_MESSAGES = [
    {
        "message": "我需要帮助解决这个问题",
        "context": "有人知道怎么处理吗？",
        "expected_category": "interrupt",
    },
    {
        "message": "服务器宕机了",
        "context": "生产环境",
        "expected_category": "interrupt",
    },
    {
        "message": "紧急通知",
        "context": "关于上线",
        "expected_category": "interrupt",
    },
]


# =============================================================================
# 批量评分测试数据
# =============================================================================

BATCH_TEST_DATA = {
    "small_batch": [
        {"message": "测试消息1", "context": "上下文1"},
        {"message": "测试消息2", "context": "上下文2"},
    ],
    "medium_batch": [
        {"message": f"测试消息{i}", "context": f"上下文{i}"} for i in range(10)
    ],
    "large_batch": [
        {"message": f"测试消息{i}", "context": f"上下文{i}"} for i in range(50)
    ],
    "mixed_batch": [
        {"message": "哈哈哈", "context": "", "user_id": "user1"},
        {"message": "今天天气真好", "context": "昨天", "user_id": "user2"},
        {"message": "我需要帮助", "context": "紧急", "user_id": "user3"},
    ],
}


# =============================================================================
# 边界情况测试数据
# =============================================================================

EDGE_CASE_MESSAGES = [
    {"message": "", "context": "", "description": "空消息"},
    {"message": "a" * 600, "context": "", "description": "超长消息"},
    {"message": " ", "context": " ", "description": "纯空格"},
    {"message": "\n\t\n", "context": "", "description": "纯换行和制表符"},
    {"message": "特殊字符：!@#$%^&*()", "context": "测试上下文", "description": "特殊字符"},
    {"message": "Emoji 测试 😂🎉🔥", "context": "表情包", "description": "Emoji"},
    {"message": "测试中文字符 你好世界", "context": "中文上下文", "description": "中文"},
    {"message": "Mix of English and 中文", "context": "Mixed context", "description": "混合语言"},
    {"message": "URL: https://example.com/path?query=value", "context": "", "description": "URL"},
]


# =============================================================================
# 无效请求数据
# =============================================================================

INVALID_REQUESTS = {
    "missing_message": {"context": "只有上下文"},
    "message_too_long": {"message": "a" * 600},
    "context_too_long": {"message": "测试", "context": "a" * 600},
    "wrong_type_message": {"message": 123},
    "empty_batch": {"messages": []},
    "oversized_batch": {"messages": [{"message": f"测试{i}", "context": ""} for i in range(51)]},
}


# =============================================================================
# 评分范围测试数据
# =============================================================================

SCORE_RANGE_DATA = [
    {"score": 0.0, "expected_category": "low_value"},
    {"score": 0.1, "expected_category": "low_value"},
    {"score": 0.2, "expected_category": "low_value"},
    {"score": 0.3, "expected_category": "normal"},
    {"score": 0.4, "expected_category": "normal"},
    {"score": 0.5, "expected_category": "normal"},
    {"score": 0.6, "expected_category": "normal"},
    {"score": 0.7, "expected_category": "normal"},
    {"score": 0.8, "expected_category": "normal"},
    {"score": 0.9, "expected_category": "interrupt"},
    {"score": 1.0, "expected_category": "interrupt"},
]


# =============================================================================
# 缓存测试数据
# =============================================================================

CACHE_TEST_DATA = {
    "cache_hits": [
        {"message": "重复消息1", "context": "相同上下文"},
        {"message": "重复消息1", "context": "相同上下文"},
        {"message": "重复消息1", "context": "相同上下文"},
    ],
    "cache_misses": [
        {"message": "不同消息1", "context": "上下文1"},
        {"message": "不同消息2", "context": "上下文2"},
        {"message": "不同消息3", "context": "上下文3"},
    ],
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_score_category(score: float) -> ScoreCategory:
    """根据评分返回分类标签

    Args:
        score: 0.0-1.0 之间的评分

    Returns:
        分类标签: "low_value", "normal", 或 "interrupt"
    """
    if score < 0.3:
        return "low_value"
    if score < 0.8:
        return "normal"
    return "interrupt"


def is_valid_score(score: float) -> bool:
    """检查评分是否在有效范围内

    Args:
        score: 要检查的评分

    Returns:
        评分是否在 0.0-1.0 范围内
    """
    return 0.0 <= score <= 1.0


def is_valid_confidence(confidence: float) -> bool:
    """检查置信度是否在有效范围内

    Args:
        confidence: 要检查的置信度

    Returns:
        置信度是否在 0.0-1.0 范围内
    """
    return 0.0 <= confidence <= 1.0
