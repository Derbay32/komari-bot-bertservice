#!/usr/bin/env python
"""将 PyTorch 模型导出为 ONNX 格式

支持 ONNX Runtime 优化和模型验证。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    max_length: int = 128,
    opset_version: int = 14,
) -> None:
    """导出模型到 ONNX 格式

    Args:
        model_path: PyTorch 模型路径
        output_path: ONNX 输出路径
        max_length: 最大序列长度
        opset_version: ONNX opset 版本
    """
    logger = setup_logging()
    logger.info(f"Loading model from {model_path}")

    # 加载模型和 tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    # 准备示例输入
    dummy_input = tokenizer(
        "这是一个测试消息",
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    logger.info(f"Exporting to ONNX (opset {opset_version})...")

    # 导出
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (
            dummy_input["input_ids"],
            dummy_input["attention_mask"],
        ),
        output_path,
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Model exported to {output_path}")

    # 保存 tokenizer
    tokenizer_dir = output_file.parent / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info(f"Tokenizer saved to {tokenizer_dir}")


def validate_onnx_model(
    onnx_path: str,
    max_length: int = 128,
) -> bool:
    """验证 ONNX 模型

    Args:
        onnx_path: ONNX 模型路径
        max_length: 最大序列长度

    Returns:
        验证是否通过
    """
    logger = setup_logging()
    logger.info(f"Validating ONNX model: {onnx_path}")

    try:
        # 加载 ONNX 模型
        model = onnx.load(onnx_path)

        # 检查模型
        onnx.checker.check_model(model)
        logger.info("ONNX model check passed")

        # 获取输入输出信息
        graph = model.graph
        logger.info(f"Inputs: {[i.name for i in graph.input]}")
        logger.info(f"Outputs: {[o.name for o in graph.output]}")

        # 使用 ONNX Runtime 测试推理
        logger.info("Testing ONNX Runtime inference...")

        # 创建推理 session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # 准备测试输入
        tokenizer = AutoTokenizer.from_pretrained(str(Path(onnx_path).parent / "tokenizer"))
        test_text = "这是一个测试消息"
        inputs = tokenizer(
            test_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )

        # 推理
        outputs = session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        # 显式转换输出为 numpy 数组
        result = np.asarray(outputs[0])
        logger.info(f"Output shape: {result.shape}")
        logger.info(f"Output: {result}")

        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def optimize_onnx_model(
    onnx_path: str,
    optimized_path: str | None = None,
) -> str:
    """优化 ONNX 模型

    Args:
        onnx_path: 原始 ONNX 模型路径
        optimized_path: 优化后的输出路径，None 则覆盖原文件

    Returns:
        优化后的模型路径
    """
    logger = setup_logging()
    logger.info(f"Optimizing ONNX model: {onnx_path}")

    from onnxruntime.transformers import optimizer

    if optimized_path is None:
        optimized_path = onnx_path

    # 优化模型
    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type="bert",
        num_heads=12,  # BERT-base
        hidden_size=768,  # BERT-base
        opt_level=1,  # 基础优化
    )

    optimized_model.save_model_to_file(optimized_path)

    logger.info(f"Optimized model saved to {optimized_path}")

    return optimized_path


def compare_models(
    torch_model_path: str,
    onnx_model_path: str,
    max_length: int = 128,
    tolerance: float = 1e-5,
) -> bool:
    """比较 PyTorch 和 ONNX 模型输出

    Args:
        torch_model_path: PyTorch 模型路径
        onnx_model_path: ONNX 模型路径
        max_length: 最大序列长度
        tolerance: 容忍误差

    Returns:
        输出是否一致
    """
    logger = setup_logging()
    logger.info("Comparing PyTorch and ONNX outputs...")

    import numpy as np

    # 加载 PyTorch 模型
    torch_model = AutoModelForSequenceClassification.from_pretrained(torch_model_path)
    torch_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(torch_model_path)

    # 准备输入
    test_text = "这是一个测试消息"
    inputs = tokenizer(
        test_text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    # PyTorch 推理
    with torch.no_grad():
        torch_outputs = torch_model(**inputs)
        torch_logits = torch_outputs.logits.cpu().numpy()

    logger.info(f"PyTorch output: {torch_logits}")

    # ONNX 推理
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
    )

    onnx_outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
        },
    )
    onnx_logits = onnx_outputs[0]

    logger.info(f"ONNX output: {onnx_logits}")

    # 比较
    diff = np.abs(torch_logits - onnx_logits).max()
    logger.info(f"Max difference: {diff}")

    if diff < tolerance:
        logger.info("Outputs match within tolerance!")
        return True
    else:
        logger.warning(f"Outputs differ by {diff} > {tolerance}")
        return False


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Export PyTorch BERT model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to PyTorch model (directory or HuggingFace model ID)",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./model.onnx",
        help="Output path for ONNX model",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    parser.add_argument(
        "--optimize",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Enable ONNX Runtime optimization",
    )

    parser.add_argument(
        "--validate",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Validate ONNX model after export",
    )

    parser.add_argument(
        "--compare",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Compare PyTorch and ONNX outputs",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # 导出 ONNX
    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        max_length=args.max_length,
        opset_version=args.opset_version,
    )

    # 验证
    if args.validate:
        if not validate_onnx_model(args.output_path, args.max_length):
            logger.error("ONNX validation failed!")
            sys.exit(1)

    # 优化
    if args.optimize:
        try:
            optimize_onnx_model(args.output_path)
        except ImportError:
            logger.warning("onnxruntime-transformers not installed, skipping optimization")
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

    # 比较
    if args.compare:
        if not compare_models(args.model_path, args.output_path, args.max_length):
            logger.warning("Model outputs differ significantly!")

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
