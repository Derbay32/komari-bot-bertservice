#!/usr/bin/env python
"""模型验证脚本

验证 ONNX 模型和分词器的完整性及正确性。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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


def check_files_exist(model_path: str, tokenizer_path: str) -> bool:
    """检查文件是否存在

    Args:
        model_path: ONNX 模型路径
        tokenizer_path: 分词器路径

    Returns:
        文件是否存在
    """
    logger = logging.getLogger(__name__)

    # 检查模型文件
    model_file = Path(model_path)
    if not model_file.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    logger.info(f"✓ Model file exists: {model_path} ({model_file.stat().st_size / 1024 / 1024:.2f} MB)")

    # 检查分词器目录
    tokenizer_dir = Path(tokenizer_path)
    if not tokenizer_dir.exists():
        logger.error(f"Tokenizer directory not found: {tokenizer_path}")
        return False
    logger.info(f"✓ Tokenizer directory exists: {tokenizer_path}")

    # 检查必需文件
    required_files = [
        "config.json",
        "vocab.txt",
        "tokenizer_config.json",
    ]

    for file_name in required_files:
        file_path = tokenizer_dir / file_name
        if not file_path.exists():
            logger.error(f"Required file missing: {file_name}")
            return False
        logger.info(f"✓ {file_name} exists")

    return True


def load_onnx_model(model_path: str) -> ort.InferenceSession | None:
    """加载 ONNX 模型

    Args:
        model_path: ONNX 模型路径

    Returns:
        ONNX Runtime session，失败返回 None
    """
    logger = logging.getLogger(__name__)

    try:
        # 验证 ONNX 模型
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model validation passed")

        # 创建推理 session
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("✓ ONNX Runtime session created")

        # 打印模型信息
        logger.info(f"  Model inputs: {[i.name for i in session.get_inputs()]}")
        logger.info(f"  Model outputs: {[o.name for o in session.get_outputs()]}")

        return session

    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        return None


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerBase | None:
    """加载分词器

    Args:
        tokenizer_path: 分词器路径

    Returns:
        分词器，失败返回 None
    """
    logger = logging.getLogger(__name__)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("✓ Tokenizer loaded successfully")
        logger.info(f"  Vocab size: {tokenizer.vocab_size}")
        logger.info(f"  Model max length: {tokenizer.model_max_length}")

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None


def test_inference(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    test_messages: list[str],
) -> bool:
    """测试推理

    Args:
        session: ONNX Runtime session
        tokenizer: 分词器
        test_messages: 测试消息列表

    Returns:
        推理是否成功
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info("Running inference tests...")

        for i, message in enumerate(test_messages, 1):
            # 准备输入
            inputs = tokenizer(
                message,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="np",
            )

            # 推理
            outputs = session.run(
                None,
                {
                    "input_ids": np.asarray(inputs["input_ids"]).astype(np.int64),
                    "attention_mask": np.asarray(inputs["attention_mask"]).astype(np.int64),
                },
            )

            # 显式转换输出为 numpy 数组
            logits = np.asarray(outputs[0])
            # 数值稳定的 softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            score = float(0.0 * probs[0, 0] + 0.55 * probs[0, 1] + 1.0 * probs[0, 2])

            # 确定分类
            if score < 0.3:
                category = "low_value"
            elif score < 0.8:
                category = "normal"
            else:
                category = "interrupt"

            logger.info(f"  Test {i}:")
            logger.info(f"    Input: {message}")
            logger.info(f"    Output: logits shape={logits.shape}")
            logger.info(f"    Score: {score:.3f}")
            logger.info(f"    Category: {category}")

        return True

    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return False


def validate_model(
    model_path: str,
    tokenizer_path: str | None,
    verbose: bool = False,
) -> bool:
    """验证模型

    Args:
        model_path: ONNX 模型路径
        tokenizer_path: 分词器路径，None 则使用模型父目录
        verbose: 详细输出

    Returns:
        验证是否通过
    """
    # 确定分词器路径
    if tokenizer_path is None:
        tokenizer_path = str(Path(model_path).parent / "tokenizer")

    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    print("\n" + "=" * 50)
    print("MODEL VALIDATION")
    print("=" * 50 + "\n")

    # 1. 检查文件
    if not check_files_exist(model_path, tokenizer_path):
        return False

    print()

    # 2. 加载模型
    session = load_onnx_model(model_path)
    if session is None:
        return False

    print()

    # 3. 加载分词器
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        return False

    print()

    # 4. 测试推理
    test_messages = [
        "哈哈哈",
        "今天天气真好啊",
        "我需要帮助解决这个问题",
    ]

    if not test_inference(session, tokenizer, test_messages):
        return False

    print()
    print("=" * 50)
    print("All validations passed!")
    print("=" * 50)

    return True


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Validate ONNX model and tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/model.onnx",
        help="Path to ONNX model file",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer directory (default: <model-dir>/tokenizer)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    success = validate_model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
