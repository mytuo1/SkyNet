FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/self-update/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/self-update/main.py .
EXPOSE 8011
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011", "--workers", "2"]
