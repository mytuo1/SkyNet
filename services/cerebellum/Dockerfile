FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/cerebellum/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/cerebellum/main.py .
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
