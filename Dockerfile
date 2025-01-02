# 基础镜像
FROM python:3.9-slim

# 设置清华源为APT源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-backports main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list

# 更新并安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器
COPY . /app

# 设置清华源为Pip源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --upgrade pip

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露服务端口
EXPOSE 8066

# 启动服务
CMD ["python", "main.py"]
