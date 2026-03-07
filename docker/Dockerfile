# 多阶段构建
FROM python:3.11-slim as builder

WORKDIR /build

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到用户目录
RUN pip install --user --no-cache-dir -r requirements.txt

# 运行阶段
FROM python:3.11-slim

# 创建非root用户
RUN useradd -m -u 1000 appuser

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /root/.local /home/appuser/.local

# 复制应用代码
COPY --chown=appuser:appuser . .

# 设置环境变量
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]