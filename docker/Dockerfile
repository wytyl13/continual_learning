# 瘦身核心 1：使用 runtime 镜像，抛弃笨重的 C++ 编译工具链
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /continual_learning

# 瘦身核心 2：精简系统依赖，用 no-install-recommends 拒绝捆绑包，安装完立刻清空 apt 缓存
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# 瘦身核心 3：使用 --no-cache-dir，坚决不留几十兆甚至上百兆的 pip 缓存垃圾
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONPATH=/continual_learning
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_ENABLE_HF_TRANSFER=0

CMD ["tail", "-f", "/dev/null"]