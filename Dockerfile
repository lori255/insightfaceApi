# 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 更新并安装必要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 单独安装 numpy（为编译提供支持）
RUN pip install --no-cache-dir numpy

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露服务端口
EXPOSE 8066

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8066"]
