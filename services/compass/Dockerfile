FROM python:3.12-slim
WORKDIR /app
COPY shared /shared
COPY services/compass/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/compass/main.py .
EXPOSE 8009
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009", "--workers", "1"]
