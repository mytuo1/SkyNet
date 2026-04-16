FROM python:3.12-slim
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# === CRITICAL: Install Torch CPU wheel FIRST with correct index ===
# Must use --index-url (not --extra-index-url) to avoid pulling CUDA wheels
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Now install everything else (torch already satisfied, won't be re-installed)
COPY shared /shared
COPY services/learner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/learner/main.py .

VOLUME ["/models", "/workspace"]
EXPOSE 8016

# Single worker — model stays in memory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8016", "--workers", "1", "--timeout-keep-alive", "300"]
