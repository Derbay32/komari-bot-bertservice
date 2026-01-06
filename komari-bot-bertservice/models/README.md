# 模型目录

本目录用于存放 ONNX 模型文件和分词器。

## 目录结构

```
models/
├── README.md              # 本文档
├── model.onnx             # ONNX 模型文件
├── tokenizer/             # 分词器文件
│   ├── config.json
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── tests/
    └── model_validation.py  # 模型验证脚本
```

## 获取模型

### 方法 1: 使用下载脚本

使用提供的脚本从 HuggingFace 下载预训练模型。

**注意**：需要先安装脚本依赖：
- Poetry（开发环境）：`poetry install --with scripts`
- pip（生产环境）：`pip install huggingface-hub`

```bash
# 下载默认模型
python scripts/download_model.py --output-dir ./models

# 下载自定义模型
python scripts/download_model.py --model-name bert-base-chinese --output-dir ./models
```

### 方法 2: 从 HuggingFace 手动下载

1. 访问 [HuggingFace Model Hub](https://huggingface.co/models)
2. 选择合适的中文 BERT 模型，例如：
   - `hfl/chinese-bert-wwm-ext` - 推荐
   - `hfl/chinese-roberta-wwm-ext` - RoBERTa 版本
   - `bert-base-chinese` - Google 官方

3. 下载模型文件并使用 ONNX 导出脚本转换：

```bash
# 导出为 ONNX
python training/export_onnx.py \
  --model-path hfl/chinese-bert-wwm-ext \
  --output-path ./models/model.onnx
```

### 方法 3: 训练自定义模型

使用自己的数据微调模型：

```bash
# 1. 准备训练数据 (JSON 格式)
cat > train.json << EOF
[
  {"message": "哈哈哈", "context": "", "label": 0},
  {"message": "今天天气真好", "context": "昨天", "label": 1},
  {"message": "需要帮助", "context": "紧急", "label": 2}
]
EOF

# 数据格式说明：
# - message: 待评分的消息内容（必需）
# - context: 对话上下文（必需，可为空字符串）
# - label: 分类标签 0=low_value, 1=normal, 2=interrupt（必需）
#
# 训练时会将 context 和 message 用 [SEP] 拼接：
# - 有 context: "昨天 [SEP] 今天天气真好"
# - 无 context: "哈哈哈"

# 2. 训练模型
python training/train.py \
  --data-path train.json \
  --output-dir ./output \
  --epochs 10

# 3. 导出 ONNX
python training/export_onnx.py \
  --model-path ./output/checkpoint-best \
  --output-path ./models/model.onnx
```

## 模型文件

### model.onnx

主要的 ONNX 模型文件，包含：

- **输入**: `input_ids`, `attention_mask`
- **输出**: `logits` (shape: `[batch_size, 3]`)
- **动态批次**: 支持可变批次大小

### tokenizer/

分词器目录，包含以下文件：

- `config.json` - 模型配置
- `vocab.txt` - 词汇表
- `tokenizer_config.json` - 分词器配置
- `special_tokens_map.json` - 特殊 token 映射

## 验证模型

使用验证脚本测试模型完整性：

```bash
python -m models.tests.model_validation \
  --model-path ./models/model.onnx \
  --tokenizer-path ./models/tokenizer
```

验证脚本会检查：

1. 文件完整性
2. 模型加载
3. 分词器加载
4. 推理测试
5. 输出格式验证

**预期输出：**

```
=== Model Validation ===

✓ Model file exists
✓ Tokenizer directory exists
✓ Model loaded successfully
✓ Tokenizer loaded successfully
✓ Inference test passed
  Input: "这是一个测试"
  Output: logits shape=(1, 3)
  Score: 0.65
  Category: normal

All validations passed!
```

## 配置服务

在 `app/config.py` 或环境变量中配置模型路径：

```python
# .env
MODEL_PATH=/path/to/models/model.onnx
TOKENIZER_PATH=/path/to/models/tokenizer
```

或

```bash
export MODEL_PATH=/path/to/models/model.onnx
export TOKENIZER_PATH=/path/to/models/tokenizer
```

## 模型性能

### 推荐模型

| 模型 | 参数量 | 速度 | 准确率 | 推荐场景 |
|------|--------|------|--------|----------|
| `hfl/chinese-bert-wwm-ext` | 110M | 中等 | 高 | 通用场景，推荐 |
| `hfl/chinese-roberta-wwm-ext` | 110M | 中等 | 更高 | 需要更高准确率 |
| `hfl/chinese-bert-base` | 110M | 快 | 中等 | 资源受限 |
| `tiny-bert` | 15M | 很快 | 中等 | 极致性能要求 |

### 性能优化

1. **量化**: 将 FP32 模型量化为 INT8
2. **图优化**: 使用 ONNX Runtime 优化
3. **缓存**: 启用 LRU 缓存减少重复推理
4. **批处理**: 使用批量端点提高吞吐量

## 故障排查

### 模型加载失败

**错误**: `FileNotFoundError: model.onnx not found`

**解决方案**:
```bash
# 检查文件是否存在
ls -lh ./models/model.onnx

# 或使用下载脚本
python scripts/download_model.py --output-dir ./models
```

### 分词器错误

**错误**: `Can't load tokenizer for 'models/tokenizer'`

**解决方案**:
```bash
# 检查分词器文件
ls -lh ./models/tokenizer/

# 确保包含所有必需文件
# - config.json
# - vocab.txt
# - tokenizer_config.json
# - special_tokens_map.json
```

### 推理错误

**错误**: `Invalid input shape`

**解决方案**:
- 检查输入是否正确 tokenized
- 验证 `max_length` 设置与导出时一致
- 确保 `input_ids` 和 `attention_mask` 形状一致

### 内存不足

**错误**: `OOM: CUDA out of memory`

**解决方案**:
- 减小 `batch_size`
- 减小 `max_length`
- 使用量化模型
- 增加系统内存

## 模型更新

### 更新预训练模型

```bash
# 1. 备份现有模型
mv models/model.onnx models/model.onnx.bak

# 2. 下载新版本
python scripts/download_model.py --model-name new-model-name --output-dir ./models

# 3. 验证新模型
python -m models.tests.model_validation --model-path ./models/model.onnx
```

### 更新微调模型

```bash
# 1. 重新训练（如有新数据）
python training/train.py --data-path new_data.json --output-dir ./output

# 2. 导出新模型
python training/export_onnx.py \
  --model-path ./output/checkpoint-best \
  --output-path ./models/model.onnx

# 3. 验证
python -m models.tests.model_validation --model-path ./models/model.onnx
```

## 相关文档

- [训练文档](../training/README.md)
- [导出脚本](../training/export_onnx.py)
- [下载脚本](../scripts/download_model.py)
- [验证脚本](tests/model_validation.py)
