#!/usr/bin/env python
"""性能基准测试脚本

测试推理吞吐量和延迟，支持批量处理和缓存效果测试。
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str | None = None,
) -> tuple[ort.InferenceSession, PreTrainedTokenizerBase]:
    """加载 ONNX 模型和分词器

    Args:
        model_path: ONNX 模型路径
        tokenizer_path: 分词器路径，None 则使用模型父目录

    Returns:
        (ONNX session, tokenizer) 元组
    """
    logger = setup_logging()
    logger.info(f"Loading model from {model_path}")

    # ONNX Runtime session
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    # Tokenizer
    if tokenizer_path is None:
        tokenizer_path = str(Path(model_path).parent / "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    logger.info("Model and tokenizer loaded")

    return session, tokenizer


def prepare_inputs(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[str],
    max_length: int = 128,
) -> dict[str, np.ndarray]:
    """准备模型输入

    Args:
        tokenizer: 分词器
        messages: 消息列表
        max_length: 最大序列长度

    Returns:
        输入字典
    """
    encoded = tokenizer(
        messages,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    return {
        "input_ids": np.asarray(encoded["input_ids"]).astype(np.int64),
        "attention_mask": np.asarray(encoded["attention_mask"]).astype(np.int64),
    }


def run_inference(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    """运行推理

    Args:
        session: ONNX Runtime session
        inputs: 输入字典

    Returns:
        输出 logits
    """
    outputs = session.run(None, inputs)
    return np.asarray(outputs[0])


def calculate_percentiles(data: list[float]) -> dict[str, float]:
    """计算百分位数

    Args:
        data: 数据列表

    Returns:
        百分位数字典
    """
    if not data:
        return {}

    return {
        "min": min(data),
        "max": max(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "p50": np.percentile(data, 50),
        "p90": np.percentile(data, 90),
        "p95": np.percentile(data, 95),
        "p99": np.percentile(data, 99),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0.0,
    }


# =============================================================================
# 基准测试
# =============================================================================

def benchmark_latency(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int,
    num_warmup: int,
    max_length: int,
) -> dict[str, Any]:
    """测试单请求延迟

    Args:
        session: ONNX Runtime session
        tokenizer: 分词器
        num_requests: 请求数量
        num_warmup: 预热请求数
        max_length: 最大序列长度

    Returns:
        延迟统计字典
    """
    logger = setup_logging()
    logger.info("Benchmarking latency...")

    # 测试消息
    test_messages = ["这是一个测试消息"] * num_requests
    inputs = prepare_inputs(tokenizer, test_messages, max_length)

    # 预热
    for _ in range(num_warmup):
        run_inference(session, inputs)

    # 测试
    latencies: list[float] = []

    for i in range(num_requests):
        start = time.perf_counter()
        run_inference(session, inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

        if (i + 1) % 10 == 0:
            logger.debug(f"Completed {i + 1}/{num_requests} requests")

    return calculate_percentiles(latencies)


def benchmark_throughput(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_requests: int,
    max_length: int,
) -> dict[str, Any]:
    """测试吞吐量

    Args:
        session: ONNX Runtime session
        tokenizer: 分词器
        batch_size: 批次大小
        num_requests: 总请求数
        max_length: 最大序列长度

    Returns:
        吞吐量统计字典
    """
    logger = setup_logging()
    logger.info(f"Benchmarking throughput (batch_size={batch_size})...")

    # 计算批次数
    num_batches = num_requests // batch_size

    # 测试消息
    test_messages = ["这是一个测试消息"] * (num_batches * batch_size)
    inputs = prepare_inputs(tokenizer, test_messages, max_length)

    # 分批
    input_ids_batches = np.array_split(inputs["input_ids"], num_batches)
    attention_mask_batches = np.array_split(inputs["attention_mask"], num_batches)

    # 测试
    start = time.perf_counter()

    for input_ids, attention_mask in zip(input_ids_batches, attention_mask_batches):
        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        run_inference(session, batch_inputs)

    end = time.perf_counter()

    total_time = end - start
    total_requests = num_batches * batch_size
    throughput = total_requests / total_time

    return {
        "total_time_s": total_time,
        "total_requests": total_requests,
        "requests_per_second": throughput,
        "batch_size": batch_size,
    }


def benchmark_cache(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int,
    max_length: int,
) -> dict[str, Any]:
    """测试缓存效果

    Args:
        session: ONNX Runtime session
        tokenizer: 分词器
        num_requests: 请求数量
        max_length: 最大序列长度

    Returns:
        缓存统计字典
    """
    logger = setup_logging()
    logger.info("Benchmarking cache effectiveness...")

    # 重复消息和唯一消息
    repeat_ratio = 0.5
    num_unique = int(num_requests * (1 - repeat_ratio))
    num_repeat = num_requests - num_unique

    unique_messages = [f"这是唯一的测试消息 {i}" for i in range(num_unique)]
    repeat_messages = ["重复消息"] * num_repeat

    # 准备输入
    unique_inputs = prepare_inputs(tokenizer, unique_messages, max_length)
    repeat_inputs = prepare_inputs(tokenizer, repeat_messages, max_length)

    # 测试唯一消息
    start = time.perf_counter()
    for _ in range(num_unique):
        run_inference(session, unique_inputs)
    unique_time = time.perf_counter() - start

    # 测试重复消息（模拟缓存命中）
    start = time.perf_counter()
    for _ in range(num_repeat):
        run_inference(session, repeat_inputs)
    repeat_time = time.perf_counter() - start

    # 速度比
    unique_avg = unique_time / num_unique
    repeat_avg = repeat_time / num_repeat
    speedup = unique_avg / repeat_avg if repeat_avg > 0 else 1.0

    return {
        "unique_requests": num_unique,
        "repeat_requests": num_repeat,
        "unique_avg_latency_ms": unique_avg * 1000,
        "repeat_avg_latency_ms": repeat_avg * 1000,
        "speedup": speedup,
        "potential_hit_rate": repeat_ratio,
    }


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Performance benchmark for ONNX inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/model.onnx",
        help="Path to ONNX model",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (default: <model-dir>/tokenizer)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for throughput test",
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests for latency test",
    )

    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup requests",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--enable-cache",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Enable cache benchmark",
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
    setup_logging(args.log_level)

    # 加载模型
    session, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)

    # 延迟测试
    latency_stats = benchmark_latency(
        session,
        tokenizer,
        args.num_requests,
        args.num_warmup,
        args.max_length,
    )

    # 吞吐量测试
    throughput_stats = benchmark_throughput(
        session,
        tokenizer,
        args.batch_size,
        args.num_requests * 10,  # 更多请求用于吞吐量测试
        args.max_length,
    )

    # 缓存测试
    cache_stats: dict[str, Any] = {}
    if args.enable_cache:
        cache_stats = benchmark_cache(
            session,
            tokenizer,
            args.num_requests,
            args.max_length,
        )

    # 打印结果
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    print("\n### Single Request Latency ###")
    for key, value in latency_stats.items():
        print(f"  {key}: {value:.2f} ms")

    print("\n### Throughput ###")
    print(f"  Total time: {throughput_stats['total_time_s']:.2f} s")
    print(f"  Total requests: {throughput_stats['total_requests']}")
    print(f"  Requests/sec: {throughput_stats['requests_per_second']:.2f}")
    print(f"  Batch size: {throughput_stats['batch_size']}")

    if cache_stats:
        print("\n### Cache Effectiveness ###")
        print(f"  Unique requests: {cache_stats['unique_requests']}")
        print(f"  Repeat requests: {cache_stats['repeat_requests']}")
        print(f"  Speedup: {cache_stats['speedup']:.2f}x")
        print(f"  Potential hit rate: {cache_stats['potential_hit_rate']:.1%}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
