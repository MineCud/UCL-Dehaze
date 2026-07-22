# UCL-Dehaze GPU training image (China-friendly)
#
# Uses a small Ubuntu base + cu118 wheels (CUDA libs are inside the wheels).
# Host still needs NVIDIA driver + docker --gpus.
#
# Build:
#   docker build -t ucl-dehaze:cuda118 .
# Or override mirrors:
#   docker build -t ucl-dehaze:cuda118 \
#     --build-arg BASE_IMAGE=docker.m.daocloud.io/library/ubuntu:22.04 \
#     --build-arg TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu118 \
#     .

ARG BASE_IMAGE=docker.m.daocloud.io/library/ubuntu:22.04
ARG TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu118
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com

FROM ${BASE_IMAGE}

ARG TORCH_INDEX
ARG PIP_INDEX
ARG PIP_TRUSTED_HOST

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# Prefer China mirrors; fall back to official if a package is missing there.
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchvision==0.16.2 \
        -i ${TORCH_INDEX} \
        --trusted-host ${PIP_TRUSTED_HOST} \
        --extra-index-url https://download.pytorch.org/whl/cu118 || \
    pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN grep -vE '^(torch|torchvision)([=<>]|$)' requirements.txt > /tmp/reqs.txt && \
    pip install -r /tmp/reqs.txt \
        -i ${PIP_INDEX} \
        --trusted-host ${PIP_TRUSTED_HOST}

COPY . .

CMD ["bash"]
