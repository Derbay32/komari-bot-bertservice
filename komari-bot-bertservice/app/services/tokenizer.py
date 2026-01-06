"""分词器封装模块

提供高效的文本分词和编码功能
"""

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

# 类型别名
type TokenizedDict = dict[str, np.ndarray]


class TokenizerWrapper:
    """分词器封装（优化性能）

    提供高效的文本分词和编码功能
    """

    def __init__(self, tokenizer_path: str | Path) -> None:
        """初始化分词器

        Args:
            tokenizer_path: 分词器路径
        """
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.max_length = 128

    def encode(self, text: str) -> TokenizedDict:
        """编码文本为模型输入

        Args:
            text: 待编码的文本

        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

    def encode_batch(self, texts: list[str]) -> TokenizedDict:
        """批量编码文本

        Args:
            texts: 待编码的文本列表

        Returns:
            包含批量 input_ids 和 attention_mask 的字典
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
