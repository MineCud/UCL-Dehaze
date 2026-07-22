# UCL-Dehaze GPU training image
# Default base is NVIDIA CUDA (often easier than pytorch/* on restricted networks).
# If pull fails, rebuild with a mirror, e.g.:
#   docker build -t ucl-dehaze:cuda118 \
#     --build-arg BASE_IMAGE=docker.m.daocloud.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 .

ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

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

# CUDA wheels first (do not install CPU torch from PyPI)
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN grep -vE '^(torch|torchvision)([=<>]|$)' requirements.txt > /tmp/reqs.txt && \
    pip install -r /tmp/reqs.txt

COPY . .

CMD ["bash"]
