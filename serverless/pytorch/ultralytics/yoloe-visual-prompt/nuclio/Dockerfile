FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN uv pip install --system --no-cache -U \
    ultralytics \
    opencv-python-headless \
    pillow \
    pyyaml \
    numpy \
    redis \
    huggingface-hub

WORKDIR /opt/nuclio

# Download YOLOE model
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('jameslahm/yoloe', 'yoloe-11l-seg.pt', local_dir='/opt/nuclio')"

# Copy handler code
COPY main.py /opt/nuclio/main.py
COPY model_handler.py /opt/nuclio/model_handler.py

# Environment variables
ENV MODEL_PATH=/opt/nuclio/yoloe-11l-seg.pt
ENV REDIS_HOST=cvat_redis_ondisk
ENV REDIS_PORT=6666
ENV VPE_TTL_DAYS=30