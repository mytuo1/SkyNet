FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/working-memory/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/working-memory/main.py .
EXPOSE 8005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"]
