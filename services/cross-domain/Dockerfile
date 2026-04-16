FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/cross-domain/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/cross-domain/main.py .
EXPOSE 8008
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008", "--workers", "1"]
