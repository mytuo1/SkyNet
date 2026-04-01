FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gcc g++ curl git && rm -rf /var/lib/apt/lists/*
COPY shared /shared
COPY services/learner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/learner/main.py .
VOLUME ["/models", "/workspace"]
EXPOSE 8016
# Single worker — model lives in process memory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8016", "--workers", "1", "--timeout-keep-alive", "300"]
