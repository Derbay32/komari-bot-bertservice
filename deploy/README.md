# BERT 评分服务 - 生产环境部署指南

## 前置要求

- Docker 20.10+
- Docker Compose 2.0+

## 部署步骤

### 1. 下载部署包

从 GitHub Releases 下载对应版本的部署包：

```bash
wget https://github.com/derbay32/komari-bot/releases/latest/download/bert-scoring-deploy-v*.tar.gz
```

### 2. 解压部署包

```bash
mkdir -p /opt/bert-scoring
cd /opt/bert-scoring
tar xzf bert-scoring-deploy-*.tar.gz
```

### 3. 准备模型文件

将模型文件放置到 `models/` 目录：

```bash
# 目录结构应该是：
models/
├── bert_scoring.onnx      # 模型文件
└── tokenizer/              # 分词器文件
    ├── vocab.txt
    ├── tokenizer_config.json
    └── ...
```

### 4. 配置环境变量

复制并编辑环境变量文件：

```bash
cp .env.example .env.prod
vi .env.prod  # 或使用你喜欢的编辑器
```

**必须修改的配置：**

- `INSTANCE_ID` - 实例唯一标识（用于区分不同部署）
- `GF_SECURITY_ADMIN_PASSWORD` - Grafana 管理员密码（生产环境务必修改）

**可选配置：**

- `SENTRY_DSN` - Sentry/Glitchtip 错误追踪 DSN（如不使用请留空）
- `HEARTBEAT_URL` - 心跳监控 Webhook URL（如不使用请留空）

### 5. 启动服务

```bash
# 拉取最新镜像
docker-compose pull

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f bert-scoring
```

### 6. 验证部署

```bash
# 检查 API 服务
curl http://localhost:8000/api/v1/docs

# 检查 Prometheus
curl http://localhost:9090

# 检查 Grafana
curl http://localhost:3000
```

## 常见问题

### 模型文件不存在

如果服务启动失败并提示模型文件不存在：

1. 检查 `models/` 目录是否包含 `bert_scoring.onnx`
2. 检查 `.env.prod` 中的 `MODEL_PATH` 配置是否正确（默认为 `/app/models/bert_scoring.onnx`）

### 服务无法启动

1. 检查日志：`docker-compose logs bert-scoring`
2. 检查端口占用：`netstat -tulpn | grep 8000`
3. 检查环境变量配置

### 监控服务无法访问

1. 确认 `monitoring/` 目录包含所有配置文件
2. 检查配置文件路径是否正确
3. 查看服务日志：`docker-compose logs prometheus`

## 更新服务

### 更新 Docker 镜像

```bash
docker-compose pull
docker-compose up -d
```

### 更新配置

编辑配置文件后重启服务：

```bash
docker-compose up -d
```

## 数据持久化

监控数据存储在 `./data/` 目录：

- `./data/prometheus-data/` - Prometheus 数据
- `./data/grafana-data/` - Grafana 数据
- `./data/alertmanager-data/` - AlertManager 数据

建议定期备份这些目录。
