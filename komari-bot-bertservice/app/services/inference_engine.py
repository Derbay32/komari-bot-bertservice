"""ONNX 推理引擎模块

提供高性能的 BERT 模型推理，支持单条和批量评分
包含以下优化：
- LRU 缓存：相同输入直接返回缓存结果
- 真正的批量推理：一次推理处理多条数据
- 动态线程配置：根据 CPU 核心数自动调整
- 内存优化：使用 freed_model_after_inference
"""

import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort

from app.middleware.metrics import (
    cache_hits,
    cache_misses,
    inference_errors,
)
from app.middleware.metrics import (
    cache_size as cache_size_metric,
)
from app.services.tokenizer import TokenizerWrapper
from app.utils.logger import logger

# 类型别名
type ScoreResult = tuple[float, str, float]  # (score, category, confidence)
type InputDict = dict[str, np.ndarray]


def _calculate_optimal_threads(workers: int, cpu_count: int | None = None) -> int:
    """计算每个 worker 的最优线程数

    Args:
        workers: worker 进程数
        cpu_count: CPU 核心数（None 则自动检测）

    Returns:
        每个 worker 的线程数
    """
    cpu_count = cpu_count or os.cpu_count() or 4
    # 每个 worker 分配的线程数，避免过度订阅
    threads_per_worker = max(1, cpu_count // workers)
    # 限制最大线程数为 4（避免过多线程导致上下文切换开销）
    return min(threads_per_worker, 4)


class ONNXInferenceEngine:
    """ONNX 推理引擎（性能优化版）"""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        use_gpu: bool = False,
        cache_size: int = 1024,
        enable_parallel: bool = True,
        workers: int = 1,
        onnx_threads: int | None = None,
    ) -> None:
        """初始化推理引擎

        Args:
            model_path: ONNX 模型路径
            tokenizer_path: 分词器路径
            use_gpu: 是否使用 GPU
            cache_size: LRU 缓存大小
            enable_parallel: 是否启用并行执行
            workers: worker 进程数（用于计算线程数）
            onnx_threads: 显式指定 ONNX 线程数（None 则自动计算）
        """
        self.model_path = str(model_path)
        self.tokenizer = TokenizerWrapper(str(tokenizer_path))
        self.cache_size = cache_size
        self.enable_parallel = enable_parallel

        # 计算 ONNX Runtime 线程数（避免过度订阅 CPU）
        if onnx_threads is not None:
            self.num_threads = onnx_threads
        else:
            self.num_threads = _calculate_optimal_threads(workers)

        # 配置 ONNX Runtime 提供者
        if use_gpu:
            # 验证 CUDA 可用性
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available_providers:
                logger.warning(
                    "cuda_requested_but_unavailable",
                    available_providers=available_providers,
                    message="CUDA 请求但不可用，将使用 CPU",
                )
                providers = ["CPUExecutionProvider"]
                self.provider = "CPUExecutionProvider"
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.provider = "CUDAExecutionProvider"
                logger.info("using_gpu", providers=providers)
        else:
            providers = ["CPUExecutionProvider"]
            self.provider = "CPUExecutionProvider"

        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
            sess_options=self._create_session_options(),
        )

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.attention_mask_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

        # LRU 缓存（用于相同输入的快速响应）
        self._cache: OrderedDict[str, ScoreResult] = OrderedDict()
        self._cache_lock = Lock()  # 线程安全锁

        # 预热
        self._warmup()

        # 初始化缓存大小指标
        cache_size_metric.set(0)

        logger.info(
            "inference_engine_initialized",
            model_path=self.model_path,
            num_threads=self.num_threads,
            cache_size=self.cache_size,
            parallel=enable_parallel,
            provider=self.provider,
        )

    def _create_session_options(self) -> ort.SessionOptions:
        """优化会话选项

        Returns:
            配置好的 SessionOptions
        """
        opts = ort.SessionOptions()
        # 动态线程数
        opts.inter_op_num_threads = self.num_threads
        opts.intra_op_num_threads = self.num_threads

        # 并行执行模式
        if self.enable_parallel:
            opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 全部图优化
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 启用内存优化
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        return opts

    def _get_cache_key(self, message: str, context: str) -> str:
        """生成缓存键

        Args:
            message: 消息内容
            context: 上下文信息

        Returns:
            缓存键
        """
        return f"{context}|{message}"

    def _get_from_cache(self, cache_key: str) -> ScoreResult | None:
        """从缓存获取结果（线程安全）

        Args:
            cache_key: 缓存键

        Returns:
            缓存的结果或 None
        """
        with self._cache_lock:
            if cache_key in self._cache:
                # LRU: 移到末尾
                self._cache.move_to_end(cache_key)
                # 记录缓存命中
                cache_hits.inc()
                return self._cache[cache_key]
        # 记录缓存未命中
        cache_misses.inc()
        return None

    def _add_to_cache(self, cache_key: str, result: ScoreResult) -> None:
        """添加结果到缓存（线程安全）

        Args:
            cache_key: 缓存键
            result: 评分结果
        """
        with self._cache_lock:
            self._cache[cache_key] = result
            # LRU: 超过容量时删除最旧的
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            # 更新缓存大小指标
            cache_size_metric.set(len(self._cache))

    def _warmup(self) -> None:
        """预热模型（第一次推理较慢）"""
        _ = self.score("预热文本", "上下文")

    def score(self, message: str, context: str) -> ScoreResult:
        """对消息进行评分（带缓存）

        Args:
            message: 待评分的消息
            context: 上下文信息

        Returns:
            (score, category, confidence) 元组
        """
        # 检查缓存
        cache_key = self._get_cache_key(message, context)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # 构建输入
        input_text = f"{context} [SEP] {message}"
        inputs = self.tokenizer.encode(input_text)

        # 推理
        try:
            outputs = self.session.run(
                [self.output_name],
                {
                    self.input_name: inputs["input_ids"],
                    self.attention_mask_name: inputs["attention_mask"],
                },
            )  # type: ignore[assignment]
        except Exception:
            # 记录推理错误
            inference_errors.labels(
                error_type="inference_error", provider=self.provider
            ).inc()
            raise

        # 处理输出（ONNX Runtime 已返回 numpy 数组）
        logits = outputs[0][0]  # type: ignore[index]  # [num_classes]
        probs = self._softmax(logits)
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        # 转换为连续评分
        score = self._class_to_score(predicted_class, probs)

        # 分类
        category = self._score_to_category(score)

        result = (score, category, confidence)

        # 添加到缓存
        self._add_to_cache(cache_key, result)

        return result

    def score_batch(self, items: list[dict[str, str]]) -> list[ScoreResult]:
        """批量评分（真正的批量推理）

        将所有批次打包为一次推理，显著提升吞吐量

        Args:
            items: 包含 message 和 context 的字典列表

        Returns:
            评分结果列表
        """
        if not items:
            return []

        # 小批量直接使用单个推理
        if len(items) == 1:
            return [self.score(items[0]["message"], items[0].get("context", ""))]

        # 分离缓存命中和未命中的项
        cache_results: dict[int, ScoreResult] = {}
        uncached_items: list[tuple[int, dict[str, str]]] = []

        for idx, item in enumerate(items):
            cache_key = self._get_cache_key(item["message"], item.get("context", ""))
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                cache_results[idx] = cached
            else:
                uncached_items.append((idx, item))

        # 如果全部命中缓存
        if not uncached_items:
            return [cache_results[i] for i in range(len(items))]

        # 批量处理未缓存的项
        batch_results = self._batch_inference([item for _, item in uncached_items])

        # 合并结果
        results: list[ScoreResult] = [None] * len(items)  # type: ignore[assignment]

        # 填入缓存结果
        for idx, result in cache_results.items():
            results[idx] = result

        # 填入新计算结果并更新缓存
        for (idx, item), result in zip(uncached_items, batch_results):
            results[idx] = result
            cache_key = self._get_cache_key(item["message"], item.get("context", ""))
            self._add_to_cache(cache_key, result)

        return results  # type: ignore[return-value]

    def _batch_inference(self, items: list[dict[str, str]]) -> list[ScoreResult]:
        """真正的批量推理

        Args:
            items: 未缓存的项列表

        Returns:
            评分结果列表
        """
        # 构建批量输入
        input_texts = [
            f"{item.get('context', '')} [SEP] {item['message']}" for item in items
        ]

        # 批量编码（使用优化的批量方法）
        encoded_batch = self.tokenizer.encode_batch(input_texts)
        batch_input_ids = encoded_batch["input_ids"]
        batch_attention_mask = encoded_batch["attention_mask"]

        # 批量推理
        try:
            outputs = self.session.run(
                [self.output_name],
                {
                    self.input_name: batch_input_ids,
                    self.attention_mask_name: batch_attention_mask,
                },
            )  # type: ignore[assignment]
        except Exception:
            # 记录推理错误
            inference_errors.labels(
                error_type="batch_inference_error", provider=self.provider
            ).inc()
            raise

        # 处理批量输出（ONNX Runtime 已返回 numpy 数组）
        batch_logits = outputs[0]  # type: ignore[index]  # [batch_size, num_classes]
        results: list[ScoreResult] = []

        for logits in batch_logits:  # type: ignore[assignment]
            probs = self._softmax(logits)
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            score = self._class_to_score(predicted_class, probs)
            category = self._score_to_category(score)
            results.append((score, category, confidence))

        return results

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax 归一化

        Args:
            logits: 原始 logits

        Returns:
            归一化后的概率分布
        """
        exp_x = np.exp(logits - np.max(logits))
        return exp_x / exp_x.sum()

    def _class_to_score(self, _predicted_class: int, probs: np.ndarray) -> float:
        """将类别转换为连续评分

        使用加权平均获得更平滑的评分：
        score = 0 * P(low_value) + 0.55 * P(normal) + 1.0 * P(interrupt)

        Args:
            _predicted_class: 预测的类别（未使用，使用加权平均）
            probs: 类别概率分布

        Returns:
            连续评分值 (0.0 - 1.0)
        """
        weights = np.array([0.0, 0.55, 1.0])
        return float(np.dot(probs, weights))

    @staticmethod
    def _score_to_category(score: float) -> str:
        """将评分映射到分类

        Args:
            score: 连续评分值

        Returns:
            分类标签
        """
        match score:
            case s if s < 0.3:
                return "low_value"
            case s if s < 0.8:
                return "normal"
            case _:
                return "interrupt"
