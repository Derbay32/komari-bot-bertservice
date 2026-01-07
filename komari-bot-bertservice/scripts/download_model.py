#!/usr/bin/env python
"""从 HuggingFace 下载预训练模型和分词器

支持公开和私有模型下载，包含完整性验证。
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

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


def download_model(
    model_name: str,
    output_dir: str,
    token: str | None = None,
) -> Path:
    """下载模型

    Args:
        model_name: HuggingFace 模型名称
        output_dir: 输出目录
        token: HuggingFace 访问令牌（私有模型）

    Returns:
        模型目录路径
    """
    logger = setup_logging()
    logger.info(f"正在下载模型: {model_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 使用 huggingface_hub 下载
    model_dir = snapshot_download(
        repo_id=model_name,
        local_dir=output_path,
        local_dir_use_symlinks=False,
        token=token,
    )

    logger.info(f"模型已下载到: {model_dir}")

    # 验证下载
    logger.info("正在验证下载...")
    try:
        # 验证配置文件
        config = AutoConfig.from_pretrained(model_dir)
        num_labels = len(config.id2label) if config.id2label is not None else config.num_labels
        logger.info(f"模型配置: {config.model_type}, {num_labels} 个标签")

        # 验证分词器
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info(f"分词器词汇量: {tokenizer.vocab_size}")

        logger.info("验证完成！")

    except Exception as e:
        logger.error(f"验证失败: {e}")
        sys.exit(1)

    return Path(model_dir)


def list_files(directory: Path) -> None:
    """列出目录中的文件

    Args:
        directory: 目录路径
    """
    logger = setup_logging()
    logger.info(f"{directory} 的内容:")

    for file in sorted(directory.rglob("*")):
        if file.is_file():
            size = file.stat().st_size
            size_mb = size / (1024 * 1024)
            logger.info(f"  {file.relative_to(directory)}: {size_mb:.2f} MB")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载预训练模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="hfl/chinese-bert-wwm-ext",
        help="HuggingFace 模型名称或 ID",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="模型文件输出目录",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace 访问令牌（用于私有模型）",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出下载的文件",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(args.log_level)

    # 下载模型
    model_dir = download_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        token=args.token,
    )

    # 列出文件
    if args.list:
        list_files(model_dir)


if __name__ == "__main__":
    main()
