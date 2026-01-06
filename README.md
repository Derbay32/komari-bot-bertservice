# Komari Bot - BERT Scoring Service

自用小鞠 bot 聊天文本评分组件 - 基于 BERT 模型的实时推理 API

## 项目结构

```
komari-bert-service/
├── app/
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理
│   ├── models/
│   │   └── schemas.py          # Pydantic 数据模型
│   ├── services/
│   │   ├── inference_engine.py # ONNX 推理引擎
│   │   └── tokenizer.py        # 分词器封装
│   ├── api/
│   │   └── v1/
│   │       ├── router.py       # 路由定义
│   │       └── endpoints.py    # 端点实现
│   ├── middleware/
│   │   ├── error_handler.py    # 错误处理
│   │   └── metrics.py          # Prometheus 指标
│   └── utils/
│       └── logger.py           # 日志配置
├── models/                      # 模型文件目录
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 快速开始

### 开发环境（使用 Poetry）

Poetry 仅用于开发环境，管理开发、测试和训练依赖：

```bash
# 安装所有依赖（包括开发和测试）
poetry install

# 仅安装主依赖
poetry install --only main

# 启动开发服务器
poetry run uvicorn app.main:app --reload

# 运行测试
poetry run pytest

# 代码格式化
poetry run ruff check .
poetry run ruff format .
```

### 生产环境（使用 pip）

生产部署使用 pip 和 `requirements-prod.txt`（仅包含运行时依赖）：

```bash
# 安装依赖
pip install -r requirements-prod.txt

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 使用 Docker

Docker 使用 pip 和 `requirements-prod.txt` 安装依赖：

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## API 端点

### 健康检查

```bash
GET /health
```

响应：

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 单条评分

```bash
POST /api/v1/score
Content-Type: application/json

{
  "message": "今天天气真好啊",
  "context": "昨天下雨了",
  "user_id": "user_123",
  "group_id": "group_456"
}
```

响应：

```json
{
  "score": 0.65,
  "category": "normal",
  "confidence": 0.92,
  "processing_time_ms": 45.3
}
```

### 批量评分

```bash
POST /api/v1/score/batch
Content-Type: application/json

{
  "messages": [
    {"message": "哈哈哈", "context": "上一句"},
    {"message": "今天天气真好", "context": "昨天下雨了"}
  ]
}
```

响应：

```json
{
  "results": [
    {
      "score": 0.15,
      "category": "low_value",
      "confidence": 0.88,
      "processing_time_ms": 0
    },
    {
      "score": 0.72,
      "category": "normal",
      "confidence": 0.9,
      "processing_time_ms": 0
    }
  ],
  "total_processing_time_ms": 52.8
}
```

### Prometheus 指标

```bash
GET /metrics
```

返回 Prometheus 格式的监控指标。

## 配置

通过环境变量配置：

### 模型配置

| 变量             | 默认值                          | 说明            |
| ---------------- | ------------------------------- | --------------- |
| `MODEL_PATH`     | `models/tiny_bert_scoring.onnx` | ONNX 模型路径   |
| `TOKENIZER_PATH` | `models/tokenizer`              | 分词器路径      |
| `USE_GPU`        | `false`                         | 是否使用 GPU    |

### 推理配置

| 变量                  | 默认值  | 说明           |
| --------------------- | ------- | -------------- |
| `CACHE_SIZE`          | `1024`  | LRU 缓存大小   |
| `ENABLE_PARALLEL`     | `true`  | 是否并行推理   |
| `MAX_BATCH_SIZE`      | `50`    | 最大批处理大小 |

### API 配置

| 变量           | 默认值    | 说明            |
| -------------- | --------- | --------------- |
| `API_HOST`     | `0.0.0.0` | API 监听地址    |
| `API_PORT`     | `8000`    | API 监听端口    |
| `WORKERS`      | `4`       | 工作进程数      |
| `CORS_ORIGINS` | (空)      | CORS 允许的来源 |

### 日志配置

| 变量        | 默认值 | 说明     |
| ----------- | ------ | -------- |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### Sentry 错误追踪（可选）

启用 Sentry/Glitchtip 错误追踪：

| 变量                        | 默认值    | 说明                          |
| --------------------------- | --------- | ----------------------------- |
| `SENTRY_DSN`                | (空)      | Sentry DSN（必须设置才能启用）|
| `SENTRY_ENVIRONMENT`        | 自动检测  | 环境名称（production/development）|
| `SENTRY_TRACES_SAMPLE_RATE` | `0.1`     | 性能追踪采样率（0.0-1.0）     |
| `SENTRY_PROFILES_SAMPLE_RATE` | `0.0`   | 性能分析采样率（0.0-1.0）     |

**示例：**
```bash
# 启用 Sentry 错误追踪和性能监控
export SENTRY_DSN="https://your-dsn@sentry.io/project-id"
export SENTRY_ENVIRONMENT="production"
export SENTRY_TRACES_SAMPLE_RATE="0.1"
```

### 心跳监控（可选）

启用 Glitchtip 心跳监控：

| 变量                 | 默认值 | 说明                    |
| -------------------- | ------ | ----------------------- |
| `HEARTBEAT_URL`      | (空)   | 心跳端点 URL            |
| `HEARTBEAT_INTERVAL` | `30`   | 心跳间隔（秒）          |

**示例：**
```bash
# 启用每 30 秒的心跳监控
export HEARTBEAT_URL="https://heartbeat.glitchtip.com/bproject-id/uuid"
export HEARTBEAT_INTERVAL="30"
```

## 开发

### 运行测试

```bash
poetry run pytest
```

### 代码格式化

```bash
# 使用 ruff 进行格式化和 lint
poetry run ruff check .
poetry run ruff format .
```

### 类型检查

```bash
poetry run mypy app/
```

## 许可证

MIT License
