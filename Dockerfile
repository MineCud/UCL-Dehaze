# UCL-Dehaze GPU training image (China-friendly)
#
# Recommended (fast): download wheels on the host first, then build.
#   bash download_torch_wheels.sh
#   docker build -t ucl-dehaze:cuda118 .
#
# Host needs NVIDIA driver + docker --gpus for training.

ARG BASE_IMAGE=docker.m.daocloud.io/library/ubuntu:22.04
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com

FROM ${BASE_IMAGE}

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
    && python3 -m venv /opt/venv \
    && pip install --upgrade pip

# Local wheels (copied from host ./wheels) — avoids slow in-build download
COPY wheels/ /tmp/wheels/
RUN if ls /tmp/wheels/torch*.whl >/dev/null 2>&1; then \
      echo "Installing torch from local wheels..." && \
      pip install /tmp/wheels/torch*.whl /tmp/wheels/torchvision*.whl && \
      rm -rf /tmp/wheels; \
    else \
      echo "ERROR: ./wheels/*.whl not found." && \
      echo "On the host run: bash download_torch_wheels.sh" && \
      exit 1; \
    fi

COPY requirements.txt .
RUN grep -vE '^(torch|torchvision)([=<>]|$)' requirements.txt > /tmp/reqs.txt && \
    pip install -r /tmp/reqs.txt \
        -i ${PIP_INDEX} \
        --trusted-host ${PIP_TRUSTED_HOST}

COPY . .

CMD ["bash"]
