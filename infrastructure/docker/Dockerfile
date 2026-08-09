# infrastructure/docker/Dockerfile
#
# Recipe to containerize the app. Each instruction creates a cached layer;
# dependencies are installed BEFORE source is copied so that changing your
# code does not force a slow reinstall of all packages.

FROM python:3.11-slim

# System libraries that Python packages need:
#   libgl1-mesa-glx → OpenCV image processing
#   libpq-dev       → PostgreSQL client (psycopg2)
#   gcc             → compiling some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1 libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application source
COPY app/ ./app/
COPY ml/ ./ml/
COPY reports/ ./reports/

EXPOSE 8000

# --reload restarts the server when code changes (development convenience)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]