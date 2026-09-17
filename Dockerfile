# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies yang dibutuhkan untuk build binary/C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Buat virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ==========================================
# Stage 2: Runner
# ==========================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install runtime tools (curl untuk docker healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment dari stage builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Siapkan direktori data & cache
RUN mkdir -p /app/data/cache /app/data/chroma_db

# Salin source code
COPY src/ /app/src/
COPY backend/ /app/backend/
COPY script/ /app/script/
COPY data/ /app/data/

EXPOSE 8000

# Jalankan backend FastAPI dengan uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
