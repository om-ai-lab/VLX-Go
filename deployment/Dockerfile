ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE_IMAGE}

WORKDIR /app
COPY requirements.txt ./
COPY deployment/requirements.txt ./deployment/requirements.txt
# Preserve the compatible PyTorch/CUDA versions provided by the base image.
RUN python -c 'from importlib.metadata import version; from pathlib import Path; Path("/tmp/framework-constraints.txt").write_text("".join(f"{name}=={version(name)}\n" for name in ("torch", "torchvision")))' \
    && python -m pip install --no-cache-dir -c /tmp/framework-constraints.txt -r deployment/requirements.txt \
    && rm /tmp/framework-constraints.txt

COPY vlxgo/ ./vlxgo/
COPY deployment/__init__.py deployment/engine_server.py deployment/run_engine_server.sh ./deployment/
ENV PYTHONUNBUFFERED=1 \
    HF_MODEL_DIR=/app/resources/vlxgo \
    DINOV3_MODEL_PATH=/app/resources/dinov3-vits16-pretrain-lvd1689m \
    SIGLIP_MODEL_PATH=/app/resources/siglip-so400m-patch14-384 \
    HF_LOCAL_FILES_ONLY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/vlxgo-hf-cache \
    HF_MODULES_CACHE=/tmp/vlxgo-hf-cache/modules \
    VLXGO_ATTN_IMPL=sdpa \
    PORT=8233 \
    CUDA_VISIBLE_DEVICES=0
EXPOSE 8233
CMD ["bash", "deployment/run_engine_server.sh"]
