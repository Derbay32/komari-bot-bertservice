# 实用工具脚本

本目录包含用于模型管理、性能测试和健康检查的实用脚本。

## 可用脚本

### download_model.py

从 HuggingFace 下载预训练模型和分词器。

**用法：**

```bash
python scripts/download_model.py --model-name hfl/chinese-bert-wwm-ext --output-dir ./models
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-name` | HuggingFace 模型名称 | `hfl/chinese-bert-wwm-ext` |
| `--output-dir` | 输出目录 | `./models` |
| `--token` | HuggingFace 访问令牌（私有模型） | `null` |

**示例：**

```bash
# 下载默认模型
python scripts/download_model.py

# 下载自定义模型
python scripts/download_model.py --model-name bert-base-chinese --output-dir ./custom-model

# 下载私有模型（需要 token）
python scripts/download_model.py --model-name org/private-model --token hf_xxx
```

### benchmark.py

性能基准测试脚本，用于测试推理吞吐量和延迟。

**用法：**

```bash
python scripts/benchmark.py --model-path ./models/model.onnx
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | ONNX 模型路径 | `./models/model.onnx` |
| `--tokenizer-path` | 分词器路径 | 与模型相同 |
| `--batch-size` | 批次大小 | `1` |
| `--num-requests` | 请求数量 | `100` |
| `--num-warmup` | 预热请求数 | `10` |
| `--max-length` | 最大序列长度 | `128` |
| `--enable-cache` | 启用缓存测试 | `true` |

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
=== Benchmark Results ===

Single request latency:
  Mean: 12.5ms
  P50: 12.1ms
  P95: 14.2ms
  P99: 16.8ms

Throughput:
  Requests/sec: 79.8
  Batch (16): 1247.2 requests/sec

Cache effectiveness:
  Hit rate: 85.3%
  Speedup: 8.2x
```

### health_check.py

服务健康检查脚本，支持持续监控模式和 CI/CD 集成。

**用法：**

```bash
python scripts/health_check.py --base-url http://localhost:8000
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base-url` | 服务基础 URL | `http://localhost:8000` |
| `--timeout` | 请求超时（秒） | `5` |
| `--interval` | 检查间隔（秒，持续模式） | `10` |
| `--continuous` | 持续监控模式 | `false` |
| `--max-failures` | 最大失败次数（退出） | `3` |
| `--verbose` | 详细输出 | `false` |

**示例：**

```bash
# 单次检查
python scripts/health_check.py --base-url http://localhost:8000

# 持续监控
python scripts/health_check.py --base-url http://localhost:8000 --continuous --interval 30

# CI/CD 集成（失败时非零退出码）
python scripts/health_check.py --base-url http://localhost:8000 --max-failures 1

# 详细模式
python scripts/health_check.py --base-url http://localhost:8000 --verbose
```

**退出码：**

- `0`: 健康检查通过
- `1`: 健康检查失败
- `2`: 网络错误

## 开发新脚本

遵循以下约定：

1. **参数解析**：使用 `argparse` 并提供 `--help`
2. **日志**：使用 `logging` 模块，支持 `--log-level`
3. **退出码**：使用适当的标准退出码
4. **文档**：在文件顶部添加文档字符串

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


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=getattr(logging, log_level.upper()),
    )
    return logging.getLogger(__name__)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="脚本描述")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    # 脚本逻辑
    logger.info("Running script...")


if __name__ == "__main__":
    main()
```
