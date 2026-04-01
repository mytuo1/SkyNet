FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/thalamus/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/thalamus/main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
