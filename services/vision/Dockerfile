FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxext6     libxrender-dev libgomp1 v4l-utils curl && rm -rf /var/lib/apt/lists/*
COPY shared /shared
COPY services/vision/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/vision/main.py .
EXPOSE 8017
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8017", "--workers", "1"]
