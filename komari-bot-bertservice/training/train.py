#!/usr/bin/env python
"""BERT 模型微调脚本

支持自定义超参数、Early stopping、学习率调度、W&B 集成等功能。
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# 类型别名
type LabelMap = dict[str, int]
type MetricsDict = dict[str, float]


# =============================================================================
# 配置
# =============================================================================

@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "hfl/chinese-bert-wwm-ext"
    num_labels: int = 3
    max_length: int = 128
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False


@dataclass
class OptimizerConfig:
    """优化器配置"""
    weight_decay: float = 0.01
    adam_epsilon: float = 1.0e-8


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    type: str = "linear"
    num_warmup_steps: int | None = None


@dataclass
class EarlyStoppingConfig:
    """Early stopping 配置"""
    patience: int = 3
    min_delta: float = 0.001


@dataclass
class LoggingConfig:
    """日志配置"""
    steps: int = 100
    eval_steps: int = 500


@dataclass
class OutputConfig:
    """输出配置"""
    save_total_limit: int = 3
    save_strategy: str = "steps"
    save_steps: int = 500


@dataclass
class Config:
    """完整训练配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


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


def load_config(config_path: str | None) -> Config:
    """从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径，None 则使用默认配置

    Returns:
        配置对象
    """
    default_config = Config()

    if config_path is None or not os.path.exists(config_path):
        return default_config

    with Path(config_path).open() as f:
        config_data = yaml.safe_load(f)

    # 递归更新配置
    def update_config(target: Any, source: dict[str, Any]) -> Any:
        for key, value in source.items():
            if hasattr(target, key):
                attr = getattr(target, key)
                if hasattr(attr, "__dataclass_fields__"):
                    setattr(target, key, update_config(attr, value))
                else:
                    setattr(target, key, value)
        return target

    return update_config(default_config, config_data)


def load_training_data(data_path: str) -> Dataset:
    """加载训练数据

    Args:
        data_path: JSON 格式的训练数据路径

    Returns:
        HuggingFace Dataset 对象
    """
    with Path(data_path).open() as f:
        data = json.load(f)

    return Dataset.from_list(data)


def compute_metrics(eval_preds: EvalPrediction) -> MetricsDict:
    """计算评估指标

    Args:
        eval_preds: EvalPrediction 对象

    Returns:
        指标字典
    """
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)

    accuracy = float(accuracy_score(labels, predictions))

    return {
        "accuracy": accuracy,
    }


# =============================================================================
# 数据处理
# =============================================================================

class DataProcessor:
    """数据处理类

    处理文本分类数据的预处理和 tokenization
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        label_map: LabelMap | None = None,
    ):
        """初始化数据处理器

        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
            label_map: 标签映射，None 则自动推断
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or {
            "low_value": 0,
            "normal": 1,
            "interrupt": 2,
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def preprocess_function(self, examples: dict[str, list]) -> dict[str, list]:
        """预处理函数

        Args:
            examples: 包含 message 和 context 的字典

        Returns:
            tokenized 结果
        """
        # 拼接消息和上下文
        texts = []
        for message, context in zip(examples["message"], examples["context"]):
            if context and context.strip():
                texts.append(f"{context} [SEP] {message}")
            else:
                texts.append(message)

        # Tokenization
        tokenized = self.tokenizer(  # type: ignore[call-arg]
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        return tokenized

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """准备数据集

        Args:
            dataset: 原始数据集

        Returns:
            处理后的数据集
        """
        # 确保标签是整数
        if "label" not in dataset.column_names:
            raise ValueError("数据集必须包含 'label' 列")

        # Tokenization
        tokenized = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["message", "context"],
            desc="Tokenizing",
        )

        # 设置格式
        tokenized.set_format("torch")

        return tokenized


# =============================================================================
# 训练主函数
# =============================================================================

def train(
    data_path: str,
    output_dir: str,
    config: Config,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    """训练模型

    Args:
        data_path: 训练数据路径
        output_dir: 输出目录
        config: 训练配置
        wandb_project: W&B 项目名
        wandb_run_name: W&B 运行名
        resume_from_checkpoint: 检查点路径
    """
    logger = setup_logging()
    logger.info("Loading training data...")
    dataset = load_training_data(data_path)
    logger.info(f"Loaded {len(dataset)} examples")

    # 划分训练集和验证集
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # 加载 tokenizer 和模型
    logger.info(f"Loading model: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model_config = AutoConfig.from_pretrained(
        config.model.name,
        num_labels=config.model.num_labels,
        hidden_dropout_prob=config.model.dropout,
        attention_probs_dropout_prob=config.model.dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        config=model_config,
    )

    # 数据预处理
    logger.info("Preprocessing data...")
    processor = DataProcessor(tokenizer, config.model.max_length)
    train_dataset = processor.prepare_dataset(train_dataset)
    eval_dataset = processor.prepare_dataset(eval_dataset)

    # 训练参数
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size * 2,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.optimizer.weight_decay,
        adam_epsilon=config.optimizer.adam_epsilon,
        lr_scheduler_type=config.scheduler.type,
        fp16=config.training.fp16,
        logging_steps=config.logging.steps,
        eval_steps=config.logging.eval_steps,
        eval_strategy="steps",  # 必须与 save_strategy 匹配
        save_strategy=config.output.save_strategy,
        save_steps=config.output.save_steps,
        save_total_limit=config.output.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["wandb"] if wandb_project else [],
        run_name=wandb_run_name,
    )

    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping.patience,
        early_stopping_threshold=config.early_stopping.min_delta,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    # 恢复训练
    checkpoint = resume_from_checkpoint
    if checkpoint is None:
        last_checkpoint = get_last_checkpoint(str(output_path))
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
            logger.info(f"Resuming from checkpoint: {checkpoint}")

    # 训练
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 保存模型
    logger.info("Saving model...")
    trainer.save_model(str(output_path / "checkpoint-best"))
    tokenizer.save_pretrained(str(output_path / "checkpoint-best"))

    # 保存训练结果
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 最终评估
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("Training complete!")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT model for message scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必需参数
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data (JSON format)",
    )

    # 可选参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for model checkpoints",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config)",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (overrides config)",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.max_length is not None:
        config.model.max_length = args.max_length
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.fp16:
        config.training.fp16 = True

    # 设置日志
    setup_logging(args.log_level)

    # W&B 初始化
    if args.wandb:
        import wandb  # type: ignore

        wandb.init(
            project=args.wandb_project or "bert-scoring",
            name=args.wandb_run_name,
            config={
                "model": config.model.__dict__,
                "training": config.training.__dict__,
            },
        )

    # 训练
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        config=config,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.wandb_run_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
