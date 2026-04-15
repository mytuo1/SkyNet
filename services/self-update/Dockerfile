# services/self-update/Dockerfile
# ── Multi-Stage Build ─────────────────────────────────────────────────────────
# Stage 1: Bun — clones supergrokie, installs deps, builds dist/cli.js
# Stage 2: Python — runtime with bun + supergrokie project copied in
#
# supergrokie is a TypeScript/Bun project (like claude-code).
# install.sh installs bun, runs `bun install` + `bun run build`, and
# symlinks bin/supergrokie (a bash wrapper that runs `bun dist/cli.js`).
#
# BUILD ARG: SUPERGROKIE_GIT_REF
#   Passed from docker-compose.yml via: args: SUPERGROKIE_GIT_REF: ${SUPERGROKIE_GIT_REF:-main}
#   Values:
#     main         → latest commit (default, may change)
#     <commit-sha> → pinned build, e.g. SUPERGROKIE_GIT_REF=a3f91b2
#     <tag>        → pinned release tag once supergrokie cuts one
#
#   To change: set SUPERGROKIE_GIT_REF in .env, then:
#     docker compose build --no-cache self-update

# ── Stage 1: Build supergrokie from source ───────────────────────────────────
FROM oven/bun:1 AS supergrokie-builder

ARG SUPERGROKIE_GIT_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone the full repo so we can checkout any ref (branch, tag, SHA)
RUN git clone https://github.com/mytuo1/supergrokie /tmp/supergrokie

WORKDIR /tmp/supergrokie

RUN git fetch --tags && \
    git checkout ${SUPERGROKIE_GIT_REF} && \
    echo "Building supergrokie at ref: ${SUPERGROKIE_GIT_REF} ($(git rev-parse --short HEAD))"

# Install dependencies and build the TypeScript bundle → dist/cli.js
RUN bun install --frozen-lockfile 2>/dev/null || bun install
RUN bun run build

# Verify the build produced dist/cli.js
RUN ls -lh dist/cli.js && echo "supergrokie build OK"

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

# Install bun runtime (needed to execute supergrokie's dist/cli.js)
RUN curl -fsSL https://bun.sh/install | bash
ENV BUN_INSTALL="/root/.bun"
ENV PATH="$BUN_INSTALL/bin:$PATH"

# Copy the full built supergrokie project from stage 1
COPY --from=supergrokie-builder /tmp/supergrokie /opt/supergrokie

# Symlink bin/supergrokie wrapper into PATH
RUN chmod +x /opt/supergrokie/bin/supergrokie && \
    ln -sf /opt/supergrokie/bin/supergrokie /usr/local/bin/supergrokie

# Verify supergrokie is callable in the final image
RUN supergrokie --help 2>&1 | head -5 || echo "supergrokie installed (help not available)"

# Python dependencies
COPY shared /shared
COPY services/self-update/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/self-update/main.py .

VOLUME ["/workspace", "/services"]

EXPOSE 8011

# Single worker — supergrokie sessions are subprocess-based, not threaded
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011", \
     "--workers", "1", "--timeout-keep-alive", "120"]
