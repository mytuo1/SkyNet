FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/embed-queue/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/embed-queue/main.py .

EXPOSE 8021
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8021", "--workers", "1", "--timeout-keep-alive", "300"]
