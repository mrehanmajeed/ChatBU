FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/.hf_cache

# Tesseract enables OCR for scanned PDFs
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN SECRET_KEY=build-only python manage.py collectstatic --noinput

EXPOSE 8000

# One worker: each worker loads its own copy of the embedding model and the in-memory rate limiter
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn Unibot.wsgi:application --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 180"]
