# ============================================
# 阶段 1: 构建阶段
# ============================================
FROM python:3.13-slim AS builder

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /build

# 复制生产环境依赖文件
COPY requirements-prod.txt ./

# 使用 pip 安装依赖到系统目录
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# ============================================
# 阶段 2: 运行时阶段
# ============================================
FROM python:3.13-slim AS runtime

# 安装运行时依赖（最小化安装）
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --chown=appuser:appuser komari-bot-bertservice/app ./app

# 创建模型目录（但不复制模型文件，由生产环境通过 volume 挂载）
RUN mkdir -p /app/models && chown -R appuser:appuser /app

# 创建日志目录
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令（使用 uvloop 提升性能）
# 工作进程数通过环境变量配置
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]
