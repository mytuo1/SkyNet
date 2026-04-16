FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential curl && rm -rf /var/lib/apt/lists/*
COPY shared /shared
COPY services/causal-graph/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/causal-graph/main.py .
EXPOSE 8012
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012", "--workers", "1"]
