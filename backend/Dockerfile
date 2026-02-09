FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    EXIFTOOL_PATH=/usr/bin/exiftool \
    DJI_IRP_PATH=/app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_irp \
    LD_LIBRARY_PATH=/app/tools/dji-tsdk/utility/bin/linux/release_x64 \
    YOLO_DEVICE=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    exiftool \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN set -eux; \
    python -m pip install --upgrade --retries 15 --timeout 120 pip; \
    attempt=1; \
    until [ "$attempt" -gt 4 ]; do \
      python -m pip install --retries 15 --timeout 120 --progress-bar off --prefer-binary -r requirements.txt && break; \
      echo "pip install failed (attempt ${attempt}/4), retrying in 20s..."; \
      if [ "$attempt" -eq 4 ]; then exit 1; fi; \
      attempt=$((attempt + 1)); \
      sleep 20; \
    done

COPY . .

RUN set -eux; \
    if [ -f "/app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_irp" ]; then \
      chmod +x /app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_irp; \
    fi; \
    if [ -f "/app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_irp_omp" ]; then \
      chmod +x /app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_irp_omp; \
    fi; \
    if [ -f "/app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_ircm" ]; then \
      chmod +x /app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_ircm; \
    fi

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
