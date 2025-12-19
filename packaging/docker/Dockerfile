# Stone Soup Docker Image
# Multi-stage build for optimized final image

# Stage 1: Build stage
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
WORKDIR /src
COPY . .

# Install Stone Soup with dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .[dev]

# Build documentation
RUN cd docs && make html || echo "Documentation build skipped"

# Stage 2: Runtime stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy source and documentation
COPY --from=builder /src /app
COPY --from=builder /src/docs/build/html /app/docs/html 2>/dev/null || :

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create user for running application
RUN useradd -m -u 1000 stonesoup && \
    chown -R stonesoup:stonesoup /app

USER stonesoup

# Default command: Python REPL
CMD ["python"]

# Labels
LABEL maintainer="Stone Soup Developers <stonesoup@dstl.gov.uk>" \
      org.opencontainers.image.title="Stone Soup" \
      org.opencontainers.image.description="Framework for target tracking and state estimation" \
      org.opencontainers.image.url="https://github.com/dstl/Stone-Soup" \
      org.opencontainers.image.source="https://github.com/dstl/Stone-Soup" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"
