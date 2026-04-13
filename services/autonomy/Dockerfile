FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/autonomy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/autonomy/main.py .
EXPOSE 8018
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8018", "--workers", "2"]
