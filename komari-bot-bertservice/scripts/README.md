# 实用工具脚本

本目录包含用于模型管理、性能测试、数据处理和健康检查的实用脚本。

## 依赖安装

部分脚本需要额外的依赖：

**Poetry（开发环境）：**

```bash
poetry install --with scripts
```

**pip（生产环境）：**

```bash
pip install huggingface-hub google-genai tqdm
```

## 可用脚本

### process_chat.py

处理 QQ 群聊天记录导出的 JSON 文件，生成可用于模型训练或推理的数据。

**功能：**

- 只保留纯文本消息（type 为 text）
- 过滤无效文本（指令、@提及、CQ 码等）
- 合并单个用户 15 秒内发送的连续消息
- 导出为简化的 JSON 格式

**用法：**

```bash
python scripts/process_chat.py input.json -o output.json
```

**参数：**

| 参数              | 说明                       | 默认值                      |
| ----------------- | -------------------------- | --------------------------- |
| `input_file`      | 输入 JSON 文件路径（必需） | -                           |
| `-o, --output`    | 输出 JSON 文件路径         | `输入文件名_processed.json` |
| `-t, --threshold` | 合并阈值（秒）             | `15`                        |

**示例：**

```bash
# 使用默认输出文件名
python scripts/process_chat.py chat.json

# 指定输出文件
python scripts/process_chat.py chat.json -o processed_chat.json

# 自定义合并阈值（30 秒）
python scripts/process_chat.py chat.json -t 30
```

**输入格式：**

```json
{
  "messages": [
    {
      "type": "type_1",
      "sender": { "uid": "123", "name": "用户名" },
      "timestamp": 1234567890000,
      "content": { "text": "消息内容" },
      "recalled": false,
      "system": false
    }
  ]
}
```

**输出格式：**

```json
{
  "messages": [{ "sender_name": "用户名", "text": "合并后的消息内容" }]
}
```

---

### generate_training_data.py

使用 Gemini API 对聊天消息进行自动标注，生成用于模型微调的训练数据。

**功能：**

- 调用 Gemini 2.5 Flash API 自动标注消息
- 三分类标签：low_value (0), normal (1), interrupt (2)
- 支持随机采样、进度跟踪、错误重试
- 自动验证数据质量

**用法：**

```bash
python scripts/generate_training_data.py input.json -o training_data.json
```

**参数：**

| 参数                | 说明                               | 默认值                           |
| ------------------- | ---------------------------------- | -------------------------------- |
| `input_file`        | 输入聊天消息 JSON 文件路径（必需） | -                                |
| `-o, --output`      | 输出训练数据文件路径               | `./training_data.json`           |
| `-n, --sample-size` | 随机采样消息数量                   | `800`                            |
| `--seed`            | 随机种子（用于可重现性）           | `None`                           |
| `--api-key`         | Gemini API key                     | 从环境变量 `GEMINI_API_KEY` 读取 |
| `--model`           | Gemini 模型名称                    | `gemini-2.5-flash-lite`          |
| `--temperature`     | 采样温度（0.0 = 更确定性）         | `0.0`                            |
| `--retry-attempts`  | API 调用失败时的重试次数           | `3`                              |
| `--retry-delay`     | 重试之间的延迟（秒）               | `1.0`                            |
| `--no-validate`     | 跳过输出数据验证                   | `false`                          |

**评分标准：**

| 标签            | 分数范围  | 特征                                           | 示例                                   |
| --------------- | --------- | ---------------------------------------------- | -------------------------------------- |
| `low_value` (0) | 0.0 - 0.3 | 纯表情、简短笑声、无实质内容                   | "哈哈哈", "233", "😂😂😂"              |
| `normal` (1)    | 0.3 - 0.8 | 包含实质性内容的日常对话                       | "今天天气真好啊", "我觉得可以这样解决" |
| `interrupt` (2) | 0.8 - 1.0 | 小鞠知花相关内容（轻小说《败犬女主太多了！》） | "小鞠好可爱", "败犬女主太多了真好看"   |

**示例：**

```bash
# 使用默认设置
python scripts/generate_training_data.py chat.json

# 自定义采样数量和随机种子
python scripts/generate_training_data.py chat.json -n 1000 --seed 42

# 指定输出文件
python scripts/generate_training_data.py chat.json -o data/train.json

# 使用自定义 API key
python scripts/generate_training_data.py chat.json --api-key YOUR_API_KEY
```

**输出格式：**

```json
[
  {
    "message": "今天天气真好啊",
    "context": "",
    "label": 1
  },
  {
    "message": "小鞠好可爱",
    "context": "",
    "label": 2
  }
]
```

---

### download_model.py

从 HuggingFace 下载预训练模型和分词器。

**用法：**

```bash
python scripts/download_model.py --model-name hfl/chinese-bert-wwm-ext --output-dir ./models
```

**参数：**

| 参数           | 说明                             | 默认值                     |
| -------------- | -------------------------------- | -------------------------- |
| `--model-name` | HuggingFace 模型名称             | `hfl/chinese-bert-wwm-ext` |
| `--output-dir` | 输出目录                         | `./models`                 |
| `--token`      | HuggingFace 访问令牌（私有模型） | `null`                     |
| `--list`       | 列出下载的文件                   | `false`                    |
| `--log-level`  | 日志级别                         | `INFO`                     |

**示例：**

```bash
# 下载默认模型
python scripts/download_model.py

# 下载自定义模型
python scripts/download_model.py --model-name bert-base-chinese --output-dir ./custom-model

# 下载私有模型（需要 token）
python scripts/download_model.py --model-name org/private-model --token hf_xxx

# 下载后列出文件
python scripts/download_model.py --list
```

---

### benchmark.py

性能基准测试脚本，用于测试推理吞吐量和延迟。

**用法：**

```bash
python scripts/benchmark.py --model-path ./models/model.onnx
```

**参数：**

| 参数               | 说明          | 默认值                |
| ------------------ | ------------- | --------------------- |
| `--model-path`     | ONNX 模型路径 | `./models/model.onnx` |
| `--tokenizer-path` | 分词器路径    | 与模型相同            |
| `--batch-size`     | 批次大小      | `1`                   |
| `--num-requests`   | 请求数量      | `100`                 |
| `--num-warmup`     | 预热请求数    | `10`                  |
| `--max-length`     | 最大序列长度  | `128`                 |
| `--enable-cache`   | 启用缓存测试  | `true`                |
| `--log-level`      | 日志级别      | `INFO`                |

**示例：**

```bash
# 基础测试
python scripts/benchmark.py --model-path ./models/model.onnx

# 高负载测试
python scripts/benchmark.py --model-path ./models/model.onnx --batch-size 16 --num-requests 1000

# 批量测试
python scripts/benchmark.py --model-path ./models/model.onnx --batch-size 32 --num-requests 500
```

**输出示例：**

```
==================================================
BENCHMARK RESULTS
==================================================

### Single Request Latency ###
  min: 10.23 ms
  max: 18.45 ms
  mean: 12.50 ms
  median: 12.10 ms
  p50: 12.10 ms
  p90: 14.20 ms
  p95: 14.80 ms
  p99: 16.80 ms
  stdev: 1.85 ms

### Throughput ###
  Total time: 8.03 s
  Total requests: 1000
  Requests/sec: 124.72
  Batch size: 16

### Cache Effectiveness ###
  Unique requests: 50
  Repeat requests: 50
  Speedup: 8.20x
  Potential hit rate: 50.0%

==================================================
```

---

### health_check.py

服务健康检查脚本，支持持续监控模式和 CI/CD 集成。

**用法：**

```bash
python scripts/health_check.py --base-url http://localhost:8000
```

**参数：**

| 参数             | 说明                     | 默认值                  |
| ---------------- | ------------------------ | ----------------------- |
| `--base-url`     | 服务基础 URL             | `http://localhost:8000` |
| `--timeout`      | 请求超时（秒）           | `5`                     |
| `--interval`     | 检查间隔（秒，持续模式） | `10`                    |
| `--continuous`   | 持续监控模式             | `false`                 |
| `--max-failures` | 最大失败次数（退出）     | `3`                     |
| `--check-model`  | 同时检查模型推理端点     | `false`                 |
| `--verbose`      | 详细输出                 | `false`                 |

**示例：**

```bash
# 单次检查
python scripts/health_check.py --base-url http://localhost:8000

# 持续监控
python scripts/health_check.py --base-url http://localhost:8000 --continuous --interval 30

# CI/CD 集成（失败时非零退出码）
python scripts/health_check.py --base-url http://localhost:8000 --max-failures 1

# 包含模型推理检查
python scripts/health_check.py --base-url http://localhost:8000 --check-model

# 详细模式
python scripts/health_check.py --base-url http://localhost:8000 --verbose
```

**退出码：**

| 退出码 | 说明                 |
| ------ | -------------------- |
| `0`    | 健康检查通过         |
| `1`    | 健康检查失败         |
| `2`    | 键盘中断（用户终止） |

---

## 工作流示例

### 完整训练数据准备流程

```bash
# 1. 处理原始聊天记录
python scripts/process_chat.py raw_chat.json -o processed_chat.json

# 2. 生成训练数据（需要设置 GEMINI_API_KEY 环境变量，-n 为随机采样数量，默认 800）
export GEMINI_API_KEY=your_api_key
python scripts/generate_training_data.py processed_chat.json -n 800 -o training_data.json

# 3. 训练模型
python training/train.py --data-path training_data.json --output-dir ./output

# 4. 导出 ONNX 模型
python training/export_onnx.py --model-path ./output/checkpoint-best --output-path ./models/model.onnx

# 5. 性能测试
python scripts/benchmark.py --model-path ./models/model.onnx

# 6. 部署后健康检查
python scripts/health_check.py --base-url http://localhost:8000 --check-model
```

---

## 开发新脚本

遵循以下约定：

1. **参数解析**：使用 `argparse` 并提供 `--help`
2. **日志**：使用 `logging` 模块，支持 `--log-level`
3. **退出码**：使用适当的标准退出码
4. **文档**：在文件顶部添加文档字符串
5. **类型注解**：使用 Python 3.13 类型别名语法

**模板：**

```python
#!/usr/bin/env python
"""脚本描述

用法:
    python scripts/script_name.py [options]
"""

import argparse
import logging
import sys
from pathlib import Path

# 类型别名（Python 3.13）
type Config = dict[str, str | int]


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


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="脚本描述",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging(args.log_level)

    # 脚本逻辑
    logger.info("Running script...")


if __name__ == "__main__":
    main()
```
