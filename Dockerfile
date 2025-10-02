FROM python:3.11-slim

ARG ENV=dev
ENV APP_ENV=${ENV}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    UVICORN_WORKERS=8

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libstdc++6 \
    libgomp1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*


# install python deps (copy reqs dulu biar cache efektif)
COPY requirements.pip.txt .
RUN python -m pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.pip.txt

# copy source
COPY . .

# copy env file sesuai profile
# (default .env.dev, override saat build)
COPY .env.${APP_ENV} /app/.env

EXPOSE ${PORT}

# Jalankan dengan uvicorn dulu (lebih mudah debug). Stabil? Silakan ganti ke Gunicorn (lihat komentar di bawah).
#CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}

# === Alternatif (setelah stabil), pakai Gunicorn:
CMD exec gunicorn app.main:app \
    --workers ${UVICORN_WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT} \
    --timeout 180
