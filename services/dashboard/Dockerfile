FROM python:3.12-slim

WORKDIR /app

COPY shared /shared
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/dashboard/main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
