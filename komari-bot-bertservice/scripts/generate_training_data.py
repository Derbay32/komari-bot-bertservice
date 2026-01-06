#!/usr/bin/env python
"""生成 BERT 模型训练数据

使用 Gemini API 对聊天消息进行自动标注，生成用于模型微调的训练数据。
支持随机采样、进度跟踪、错误重试等功能。
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Literal

from google import genai
from tqdm import tqdm

# 类型别名（Python 3.13）
type Label = Literal[0, 1, 2]
type ScoreCategory = Literal["low_value", "normal", "interrupt"]
type TrainingSample = dict[str, str | int]
type ChatMessage = dict[str, str]

# Gemini API Prompt 模板
SCORING_PROMPT = """你是一个聊天消息价值评估专家。请根据以下标准对消息进行评分。

## 评分标准

**Label 0 (low_value)** - 低价值消息:
- 分数范围: 0.0 - 0.3
- 特征: 纯表情、简短笑声、无实质内容
- 示例: "哈哈哈", "233", "笑死我了", "啊啊啊", "www", "😂😂😂"

**Label 1 (normal)** - 正常消息:
- 分数范围: 0.3 - 0.8
- 特征: 包含实质性内容的日常对话
- 示例: "今天天气真好啊", "我觉得这个问题可以这样解决", "大家吃饭了吗"

**Label 2 (interrupt)** - 打断性消息（小鞠知花相关）:
- 分数范围: 0.8 - 1.0
- 特征: 与轻小说《败犬女主太多了！》（負けヒロインが多すぎる!）中的角色"小鞠知花"（こまりちか）相关的内容
- 判断标准:
  * 提到"小鞠"、"知花"、"小鞠知花"等角色名称
  * 提到"败犬"、"败犬女主"、"败犬女主太多了"等作品相关词汇
  * 提到"文艺部"、"温水"、"温水和彦"等作品中的人物或组织
  * 提到与该作品相关的动画、小说、漫画等讨论
  * 表达对该角色或作品的喜爱、讨论、评价等情感
- 示例: "小鞠好可爱", "败犬女主太多了真好看", "我想和小鞠结婚", "小鞠是我老婆", "文艺部活动", "温水前辈"

## 任务

请评估以下消息，只返回标签数字（0、1 或 2），不要返回其他内容。

消息: {message}

标签:"""


class GeminiLabeler:
    """Gemini API 标注器

    使用 Google Gen AI SDK (google-genai) 对消息进行自动标注
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """初始化标注器

        Args:
            api_key: Gemini API key（None 则从环境变量读取）
            model: Gemini 模型名称
            temperature: 采样温度
            retry_attempts: 重试次数
            retry_delay: 重试延迟（秒）

        Raises:
            ValueError: API key 未配置
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured. "
                "Please set the GEMINI_API_KEY environment variable or provide it via --api-key."
            )

        self.model = model
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # 创建客户端
        self.client = genai.Client(api_key=self.api_key)

    def label_message(self, message: str) -> tuple[Label, ScoreCategory]:
        """标注单条消息

        Args:
            message: 待标注的消息

        Returns:
            (label, category) 元组

        Raises:
            RuntimeError: API 调用失败且重试耗尽
        """
        prompt = SCORING_PROMPT.format(message=message)

        for attempt in range(self.retry_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": self.temperature},
                )

                # 解析响应
                label_text = response.text.strip()
                label = int(label_text)

                if label not in (0, 1, 2):
                    print(f"[警告] Gemini 返回无效标签: {label_text}，消息: {message[:50]}")
                    label = 1  # 默认为 normal

                category = self._label_to_category(label)
                return label, category

            except Exception as e:
                print(f"[警告] Gemini API 调用失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")

                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Gemini API failed after {self.retry_attempts} attempts: {e}"
                    )

    @staticmethod
    def _label_to_category(label: Label) -> ScoreCategory:
        """将标签转换为分类名称

        Args:
            label: 标签数字

        Returns:
            分类名称
        """
        mapping: dict[Label, ScoreCategory] = {
            0: "low_value",
            1: "normal",
            2: "interrupt",
        }
        return mapping[label]

    def close(self):
        """关闭客户端连接"""
        if hasattr(self, "client"):
            self.client.close()


def load_chat_messages(input_file: Path) -> list[ChatMessage]:
    """加载聊天消息

    Args:
        input_file: 输入 JSON 文件路径

    Returns:
        消息列表

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 解析失败
    """
    print(f"[加载] 读取文件: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    print(f"[加载] 共 {len(messages)} 条消息")

    return messages


def sample_messages(
    messages: list[ChatMessage], sample_size: int, seed: int | None = None
) -> list[ChatMessage]:
    """随机采样消息

    Args:
        messages: 消息列表
        sample_size: 采样数量
        seed: 随机种子（用于可重现性）

    Returns:
        采样后的消息列表
    """
    if sample_size >= len(messages):
        print(f"[采样] 无需采样，使用全部 {len(messages)} 条消息")
        return messages

    if seed is not None:
        random.seed(seed)
        print(f"[采样] 使用随机种子: {seed}")

    sampled = random.sample(messages, sample_size)
    print(f"[采样] 从 {len(messages)} 条消息中随机采样 {sample_size} 条")

    return sampled


def generate_training_data(
    messages: list[ChatMessage],
    labeler: GeminiLabeler,
    batch_size: int = 10,
) -> list[TrainingSample]:
    """生成训练数据

    Args:
        messages: 消息列表
        labeler: Gemini 标注器
        batch_size: 批处理大小（用于进度条更新频率）

    Returns:
        训练样本列表
    """
    training_data: list[TrainingSample] = []

    # 统计计数器
    label_counts = {0: 0, 1: 0, 2: 0}

    print(f"\n[标注] 开始标注 {len(messages)} 条消息...")

    # 使用 tqdm 进度条
    with tqdm(total=len(messages), desc="标注进度", unit="条") as pbar:
        for idx, msg in enumerate(messages):
            text = msg.get("text", "")

            if not text or not text.strip():
                pbar.update(1)
                continue

            try:
                # 调用 Gemini API 标注
                label, category = labeler.label_message(text)

                # 构建训练样本
                sample: TrainingSample = {
                    "message": text,
                    "context": "",  # 保持为空
                    "label": label,
                }

                training_data.append(sample)
                label_counts[label] += 1

                # 更新进度条
                pbar.set_postfix(
                    {
                        "low_value": label_counts[0],
                        "normal": label_counts[1],
                        "interrupt": label_counts[2],
                    }
                )
                pbar.update(1)

                # 每 batch_size 条记录一次日志
                if (idx + 1) % batch_size == 0:
                    pass  # 进度条已足够

            except Exception as e:
                print(f"\n[错误] 标注失败 (索引 {idx}): {e}")
                print(f"       消息内容: {text[:50]}...")
                # 继续处理下一条
                pbar.update(1)

    print(f"\n[标注完成] 总样本数: {len(training_data)}")
    print(f"          标签分布: low_value={label_counts[0]}, normal={label_counts[1]}, interrupt={label_counts[2]}")

    return training_data


def save_training_data(data: list[TrainingSample], output_file: Path) -> None:
    """保存训练数据

    Args:
        data: 训练样本列表
        output_file: 输出文件路径
    """
    print(f"\n[保存] 写入文件: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[保存] 完成!")


def validate_training_data(data: list[TrainingSample]) -> bool:
    """验证训练数据质量

    Args:
        data: 训练样本列表

    Returns:
        验证是否通过
    """
    print(f"\n[验证] 验证 {len(data)} 条训练数据...")

    errors = []

    for idx, sample in enumerate(data):
        # 检查必需字段
        if "message" not in sample:
            errors.append(f"Sample {idx}: missing 'message' field")

        if "context" not in sample:
            errors.append(f"Sample {idx}: missing 'context' field")

        if "label" not in sample:
            errors.append(f"Sample {idx}: missing 'label' field")

        # 检查标签值
        label = sample.get("label")
        if label not in (0, 1, 2):
            errors.append(f"Sample {idx}: invalid label {label}")

        # 检查消息非空
        message = sample.get("message", "")
        if not message or not str(message).strip():
            errors.append(f"Sample {idx}: empty message")

    if errors:
        print(f"[验证失败] 发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前 10 个错误
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
        return False

    print("[验证] ✓ 所有数据验证通过!")
    return True


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="使用 Gemini API 生成 BERT 模型训练数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="输入聊天消息 JSON 文件路径（group_msg_processed.json）",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_file",
        default=None,
        help="输出训练数据文件路径（默认：./training_data.json）",
    )

    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=800,
        dest="sample_size",
        help="随机采样消息数量",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重现性）",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        dest="api_key",
        help="Gemini API key（默认从 GEMINI_API_KEY 环境变量读取）",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        dest="model",
        help="Gemini 模型名称",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        dest="temperature",
        help="采样温度（0.0 = 更确定性）",
    )

    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        dest="retry_attempts",
        help="API 调用失败时的重试次数",
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        dest="retry_delay",
        help="重试之间的延迟（秒）",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="跳过输出数据验证",
    )

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("训练数据生成脚本")
    print("=" * 60)

    try:
        # 1. 加载聊天消息
        messages = load_chat_messages(args.input_file)

        # 2. 随机采样
        sampled = sample_messages(messages, args.sample_size, args.seed)

        # 3. 初始化 Gemini 标注器
        print(f"\n[初始化] 使用 Gemini 模型: {args.model}")
        labeler = GeminiLabeler(
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
        )

        # 4. 生成训练数据
        training_data = generate_training_data(sampled, labeler)

        # 5. 关闭客户端
        labeler.close()

        # 6. 验证数据质量
        if not args.no_validate:
            if not validate_training_data(training_data):
                print("\n[错误] 数据验证失败，请检查输出")
                sys.exit(1)

        # 7. 保存训练数据
        output_file = args.output_file or Path("training_data.json")
        save_training_data(training_data, output_file)

        # 8. 输出统计信息
        print("\n" + "=" * 60)
        print("✓ 训练数据生成完成!")
        print("=" * 60)
        print(f"输出文件: {output_file}")
        print(f"总样本数: {len(training_data)}")

        # 打印标签分布
        label_counts = {0: 0, 1: 0, 2: 0}
        for sample in training_data:
            label_counts[sample["label"]] += 1

        print(f"\n标签分布:")
        print(
            f"  - low_value (0): {label_counts[0]} ({label_counts[0] / len(training_data) * 100:.1f}%)"
        )
        print(
            f"  - normal (1):    {label_counts[1]} ({label_counts[1] / len(training_data) * 100:.1f}%)"
        )
        print(
            f"  - interrupt (2): {label_counts[2]} ({label_counts[2] / len(training_data) * 100:.1f}%)"
        )
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n[错误] 文件不存在: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n[错误] JSON 解析失败: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[错误] 配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 未预期的错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
