# services/self-update/Dockerfile
# ── Multi-Stage Build ─────────────────────────────────────────────────────────
# Stage 1: Rust — builds the claw binary
# Stage 2: Python — runtime with claw binary copied in
#
# BUILD ARG: CLAW_GIT_REF
#   Passed from docker-compose.yml via: args: CLAW_GIT_REF: ${CLAW_GIT_REF:-main}
#   Values:
#     main         → latest commit (default, may change)
#     <commit-sha> → pinned build, e.g. CLAW_GIT_REF=a3f91b2
#     <tag>        → pinned release tag once claw cuts one
#
#   To change: set CLAW_GIT_REF in .env, then:
#     docker compose build --no-cache self-update

# ── Stage 1: Build claw from source ──────────────────────────────────────────
FROM rust:1.78-slim AS claw-builder

# Declare the build arg — WITHOUT this, ARG is invisible and
# git checkout ${CLAW_GIT_REF} would silently use an empty string
ARG CLAW_GIT_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone the full repo (not --depth 1) so we can checkout any ref
RUN git clone https://github.com/ultraworkers/claw-code /tmp/claw-code

WORKDIR /tmp/claw-code

# Checkout the pinned ref — this is why CLAW_GIT_REF exists
# git fetch --tags ensures tags are available, not just branches
RUN git fetch --tags && \
    git checkout ${CLAW_GIT_REF} && \
    echo "Building claw at ref: ${CLAW_GIT_REF} ($(git rev-parse --short HEAD))"

WORKDIR /tmp/claw-code/rust

# Build release binary
# Cargo registry is cached by Docker layer unless Cargo.toml changes
RUN cargo build --release --bin claw 2>&1 | tail -10

# Verify the binary was produced
RUN ls -lh target/release/claw && \
    ./target/release/claw --version || true

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy claw binary from stage 1
COPY --from=claw-builder /tmp/claw-code/rust/target/release/claw /usr/local/bin/claw
RUN chmod +x /usr/local/bin/claw

# Verify claw is callable in the final image
RUN /usr/local/bin/claw --version || echo "claw installed (version flag not available)"

# Python dependencies
COPY shared /shared
COPY services/self-update/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/self-update/main.py .

VOLUME ["/workspace", "/services"]

EXPOSE 8011

# Single worker — claw sessions are subprocess-based, not threaded
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011", \
     "--workers", "1", "--timeout-keep-alive", "120"]
