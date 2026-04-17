FROM python:3.12-slim
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# === Torch installation — GPU (CUDA 12.1) with CPU fallback ===
# TORCH_INDEX_URL can be overridden in docker-compose build args:
#   CPU only: https://download.pytorch.org/whl/cpu
#   CUDA 12.1: https://download.pytorch.org/whl/cu121  (default)
#   CUDA 12.4: https://download.pytorch.org/whl/cu124
#   ROCm 6.0:  https://download.pytorch.org/whl/rocm6.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url ${TORCH_INDEX_URL}

# Now install everything else (torch already satisfied, won't be re-installed)
COPY shared /shared
COPY services/learner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/learner/main.py .

VOLUME ["/models", "/workspace"]
EXPOSE 8016

# Single worker — model stays in memory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8016", "--workers", "1", "--timeout-keep-alive", "300"]
