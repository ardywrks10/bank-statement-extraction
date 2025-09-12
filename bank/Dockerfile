# Gunakan base image Python
FROM python:3.10-slim

# Install dependency sistem (untuk OpenCV dan pdf2image)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Buat direktori kerja
WORKDIR /app

# Copy requirements.txt dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua source code
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Jalankan FastAPI dengan Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
