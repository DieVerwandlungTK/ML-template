# Reference: https://docs.astral.sh/uv/guides/integration/docker/
# Development version with UV tools and CUDA support

# Define CUDA version as build argument with default value
ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

# Start with CUDA base image from NVIDIA with Python
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} as base

# タイムゾーン設定を事前に指定して対話プロンプトを回避
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Python 3.12と必要なパッケージをインストール
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    locales \
    curl \
    build-essential \
    git \
    sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy UV from the official image (more efficient than installing via pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ホストのユーザー名、ユーザーID、グループIDをビルド引数として受け取る
ARG USERNAME=appuser
ARG USER_ID=1000
ARG GROUP_ID=1000

# ユーザーとグループを指定されたIDで作成
RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -u ${USER_ID} -g ${USERNAME} -s /bin/bash -m ${USERNAME}

# Allow user to run SSH daemon without password
RUN apt-get update && apt-get install -y sudo openssh-server && \
    mkdir -p /run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "Port 2222" >> /etc/ssh/sshd_config && \
    mkdir -p /etc/sudoers.d && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/nopasswd && \
    chmod 0440 /etc/sudoers.d/nopasswd

# Create cache directories for UV and Hugging Face with proper ownership
RUN mkdir -p /home/$USERNAME/.cache/uv && chown -R $USERNAME:$USERNAME /home/$USERNAME
RUN mkdir -p /home/$USERNAME/.cache/huggingface && chown -R $USERNAME:$USERNAME /home/$USERNAME

# Install additional development tools and SSH server
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    procps \
    vim \
    build-essential \
    openssh-server \
    sudo \
    && mkdir -p /run/sshd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project for development
COPY . /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES all \
    NVIDIA_DRIVER_CAPABILITIES compute,utility

# Use non-root user for better security
USER $USERNAME
