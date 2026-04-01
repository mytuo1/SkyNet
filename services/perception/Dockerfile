FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/perception/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/perception/main.py .
EXPOSE 8006
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "2"]
