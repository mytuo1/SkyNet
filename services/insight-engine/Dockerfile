FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/insight-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/insight-engine/main.py .
EXPOSE 8020
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020", "--workers", "1"]
