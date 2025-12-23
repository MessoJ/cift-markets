# CIFT Markets - Production Dockerfile
# Multi-stage build with Rust core compilation (Phase 5-7)

# ============================================================================
# Stage 1: Python Builder
# ============================================================================
FROM python:3.11-slim AS python-builder

# Install build dependencies (including Rust for maturin)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install maturin
RUN pip install --no-cache-dir maturin

# Set working directory FIRST
WORKDIR /build

# Copy Rust source for PyO3 binding
COPY rust_core/ /build/rust_core/

# Verify Cargo.toml exists (debug step)
RUN ls -la /build/rust_core/ && test -f /build/rust_core/Cargo.toml

# Build and install Rust Python extensions
WORKDIR /build/rust_core
RUN maturin build --release && \
    pip install target/wheels/*.whl

# Copy Python requirements
WORKDIR /build
COPY pyproject.toml /build/
COPY README.md /build/
COPY cift/__init__.py /build/cift/

# Install Python dependencies
# Explicitly install CPU-only torch and torch-geometric to prevent GPU bloat
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch==2.5.1+cpu" "torch-geometric>=2.4.0" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        "torch==2.5.1+cpu" \
        "polars>=0.20.0" \
        "pandas>=2.1.0" \
        "numpy>=1.26.0" \
        "numba>=0.58.0" \
        "transformers>=4.35.0" \
        "scikit-learn>=1.3.0" \
        "xgboost>=2.0.0" \
        "pomegranate>=1.0.0" \
        "nats-py>=2.6.0" \
        "redis>=5.0.0" \
        "websockets>=12.0" \
        "msgpack>=1.0.0" \
        "asyncpg>=0.29.0" \
        "sqlalchemy[asyncio]>=2.0.23" \
        "greenlet>=3.0.0" \
        "httpx>=0.25.0" \
        "slowapi>=0.1.9" \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir --no-deps -e .

# ============================================================================
# Stage 2: Runtime (Phase 5-7 with Rust core)
# ============================================================================
FROM python:3.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cift && \
    mkdir -p /app /app/logs && \
    chown -R cift:cift /app

# Copy virtual environment from python-builder (includes Rust modules)
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=cift:cift . .

# Ensure logs directory exists with correct permissions
RUN mkdir -p /app/logs && chown -R cift:cift /app/logs

# Switch to non-root user
USER cift

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (with Rust-powered backend!)
CMD ["uvicorn", "cift.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
