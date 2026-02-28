# 1. En güncel stabil Python sürümüne geçtik
FROM python:3.12-slim

# 2. Çalışma dizini
WORKDIR /app

# 3. Ortam değişkenleri (PYTHONPATH'e /app ekleyerek import hatalarını çözüyoruz)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# 4. Sistem bağımlılıkları (gcc ve döküman derleyiciler bazen pydantic/uvloop için gerekir)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Bağımlılıkları kopyala
COPY requirements.txt .

# 6. PIP ve Paket Kurulumu (Kritik: google-genai v1.2.0+ zorlaması yapıyoruz)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Uygulama kodunu kopyala
# (.dockerignore dosyanızda .env, __pycache__, .git olduğundan emin olun)
COPY . .

# 8. Güvenlik: Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# 9. Cloud Run Portu
EXPOSE 8080

# 10. Health Check (Bağımlılıksız, standart kütüphane ile)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/docs')"

# 11. Başlatma komutu
# Not: Eğer main.py dosyan app/ klasörünün içindeyse 'app.main:app' doğrudur.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
