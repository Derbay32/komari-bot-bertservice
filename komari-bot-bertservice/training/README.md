# BERT 模型训练

本目录包含用于微调和导出 BERT 模型的脚本。

## 目录结构

```
training/
├── __init__.py              # 模块初始化
├── README.md                # 本文档
├── requirements-train.txt   # 训练依赖
├── train.py                 # 微调主脚本
├── export_onnx.py           # ONNX 导出脚本
└── configs/
    ├── __init__.py          # 配置模块
    └── default.yaml         # 默认训练配置
```

## 快速开始

### 1. 安装训练依赖

**使用 Poetry（开发环境推荐）：**

```bash
# 安装训练依赖组
poetry install --with train
```

**使用 pip（生产环境）：**

```bash
# 安装训练依赖
pip install -r training/requirements-train.txt
```

### 2. 准备训练数据

准备 JSON 格式的训练数据：

```json
[
  {
    "message": "哈哈哈",
    "context": "",
    "label": 0
  },
  {
    "message": "今天天气真好啊",
    "context": "昨天下雨了",
    "label": 1
  },
  {
    "message": "我需要帮助解决这个问题",
    "context": "有人知道怎么处理吗？",
    "label": 2
  }
]
```

**字段说明：**

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 待评分的消息内容 |
| `context` | string | 是 | 对话上下文（可为空字符串） |
| `label` | int | 是 | 分类标签（0/1/2） |

**关于 `context` 字段：**

`context` 字段用于提供对话历史上下文，帮助模型更准确地理解消息。在训练和推理时，如果 context 存在，会与 message 用 `[SEP]` token 拼接：

```
输入格式："{context} [SEP] {message}"
示例："昨天下雨了 [SEP] 今天天气真好啊"
```

**使用场景：**

- **当前实现**：context 保持为空，仅使用 message 进行分类
- **未来扩展**：可从聊天记录中提取历史消息作为 context，实现上下文感知分类

**上下文感知的优势：**

| 场景 | 无 Context | 有 Context |
|------|-----------|-----------|
| "对啊" | 难以判断 | "小鞠好可爱" + "对啊" → interrupt |
| "真的吗" | 模糊 | "明天会下雨" + "真的吗" → normal |

标签映射：
- `0`: low_value（低价值）
- `1`: normal（正常）
- `2`: interrupt（打断性）

### 3. 微调模型

```bash
python training/train.py \
  --data-path /path/to/train.json \
  --output-dir ./output \
  --config training/configs/default.yaml
```

### 4. 导出 ONNX

```bash
python training/export_onnx.py \
  --model-path ./output/checkpoint-best \
  --output-path ./models/model.onnx
```

## 训练脚本

### train.py

模型微调主脚本，支持：

- 自定义超参数配置
- 学习率调度
- Early stopping
- 梯度累积
- 混合精度训练
- W&B 集成（可选）

**主要参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-path` | 训练数据路径（必需） | - |
| `--output-dir` | 输出目录 | `./output` |
| `--config` | 配置文件路径 | `configs/default.yaml` |
| `--epochs` | 训练轮数 | 从配置文件读取 |
| `--batch-size` | 批次大小 | 从配置文件读取 |
| `--learning-rate` | 学习率 | 从配置文件读取 |
| `--max-length` | 最大序列长度 | `128` |
| `--wandb` | 启用 W&B | `False` |

**示例：**

```bash
# 基础训练
python training/train.py \
  --data-path ./data/train.json \
  --output-dir ./output

# 自定义参数训练
python training/train.py \
  --data-path ./data/train.json \
  --output-dir ./output \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --max-length 256

# 使用 W&B 监控
python training/train.py \
  --data-path ./data/train.json \
  --output-dir ./output \
  --wandb \
  --wandb-project "bert-scoring"
```

### export_onnx.py

将训练好的模型导出为 ONNX 格式。

**主要参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | PyTorch 模型路径（必需） | - |
| `--output-path` | ONNX 输出路径 | `./model.onnx` |
| `--max-length` | 最大序列长度 | `128` |
| `--optimize` | 启用 ONNX 优化 | `True` |

**示例：**

```bash
# 基础导出
python training/export_onnx.py \
  --model-path ./output/checkpoint-best \
  --output-path ./models/model.onnx

# 自定义序列长度
python training/export_onnx.py \
  --model-path ./output/checkpoint-best \
  --output-path ./models/model-256.onnx \
  --max-length 256

# 禁用优化（调试用）
python training/export_onnx.py \
  --model-path ./output/checkpoint-best \
  --output-path ./models/model.onnx \
  --optimize false
```

## 配置文件

### default.yaml

默认训练配置文件，包含模型、训练和优化器设置。

```yaml
model:
  name: "hfl/chinese-bert-wwm-ext"
  num_labels: 3
  max_length: 128
  dropout: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2.0e-5
  warmup_ratio: 0.1
  gradient_accumulation_steps: 1
  fp16: false

optimizer:
  weight_decay: 0.01
  adam_epsilon: 1.0e-8

scheduler:
  type: "linear"
  num_warmup_steps: null

early_stopping:
  patience: 3
  min_delta: 0.001

logging:
  steps: 100
  eval_steps: 500

output:
  save_total_limit: 3
  save_strategy: "steps"
  save_steps: 500
```

## 训练技巧

### 数据准备

1. **数据平衡**：确保三个类别的样本数量相对均衡
2. **数据质量**：移除重复、错误或低质量样本
3. **数据增强**：可以对少量样本进行同义词替换等增强

### 超参数调整

1. **学习率**：通常在 `1e-5` 到 `5e-5` 之间
2. **批次大小**：根据 GPU 内存调整，可用梯度累积模拟更大批次
3. **最大长度**：通常 128-256 足够，更长会增加计算开销

### 过拟合处理

如果训练集表现好但测试集表现差：

- 增加 `dropout` 值
- 减少 `epochs`
- 增加 `weight_decay`
- 使用更多训练数据

### 欠拟合处理

如果训练集和测试集表现都不好：

- 减少 `dropout`
- 增加 `epochs`
- 减小 `weight_decay`
- 使用更大的预训练模型

## 验证导出的模型

导出 ONNX 后，使用验证脚本测试：

```bash
python -m models.tests.model_validation \
  --model-path ./models/model.onnx \
  --tokenizer-path ./models/tokenizer
```

## 故障排查

### CUDA 内存不足

```bash
# 减小批次大小
python training/train.py --data-path ./data/train.json --batch-size 16

# 或使用梯度累积
python training/train.py --data-path ./data/train.json --batch-size 8 \
  --gradient-accumulation-steps 4
```

### 训练不收敛

- 检查数据标签是否正确
- 尝试不同的学习率
- 增加训练轮数
- 检查数据质量

### ONNX 导出失败

- 确保模型路径正确
- 检查 transformers 版本兼容性
- 尝试禁用优化：`--optimize false`

## 参考资源

- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [Chinese BERT 模型](https://huggingface.co/hfl/chinese-bert-wwm-ext)
