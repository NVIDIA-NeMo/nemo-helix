# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Self-hosted replacement for nvcr.io/nim/nvidia/nemoguard-jailbreak-detect.
# uv-managed. Weights download at first start (not baked).
#
# Runs on CPU by default. For GPU pods/DGX, run with `--gpus all` and
# `-e JAILBREAK_CHECK_DEVICE=cuda:0` — the Linux torch wheel bundles CUDA, so the
# same image works on both. (For a lean CPU-only image, add uv cpu/cuda extras.)
#
# Build context is this directory (so pyproject.toml + uv.lock + model/ are present).
# Requires BuildKit/buildx (this file uses a `RUN --mount=type=cache` build mount):
#   docker buildx build -t nemo/jailbreak-detect:0.1.0 --load .
ARG PYTHON_VERSION=3.11
ARG DISTROLESS_IMAGE_VERSION=v4.0.8

FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    . /etc/os-release && \
    case "${VERSION_CODENAME}" in \
        bookworm) minimum=20250419~deb12u2 ;; \
        trixie) minimum=20250419+deb13u1 ;; \
        *) echo "Unverified CA package for ${VERSION_CODENAME}" >&2; exit 1 ;; \
    esac && \
    dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge "${minimum}" && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/

ENV JAILBREAK_CHECK_DEVICE=cpu \
    HF_HOME=/opt/jailbreak-detect/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Cache mount keeps uv's ~5 GB wheel download cache out of the image layer
# (it's build-time only, not committed) while still speeding up rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

COPY model/classifier.py model/server.py ./

RUN useradd --create-home --uid 1000 jbd && \
    mkdir -p "$HF_HOME" && \
    chown -R 1000:0 /opt/jailbreak-detect && \
    chmod -R g+rwX /opt/jailbreak-detect

FROM nvcr.io/nvidia/distroless/python:${PYTHON_VERSION}-${DISTROLESS_IMAGE_VERSION} AS runtime
RUN ["python", "-c", "import shutil; shutil.rmtree('/etc/ssl/certs')"]
COPY --from=builder /etc/ssl/certs/ /etc/ssl/certs/

ENV PATH="/app/.venv/bin:$PATH"
ENV JAILBREAK_CHECK_DEVICE=cpu \
    HF_HOME=/opt/jailbreak-detect/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group
COPY --chown=1000:0 --from=builder /app/.venv /app/.venv
COPY --chown=1000:0 --from=builder /app/classifier.py /app/server.py ./
COPY --chown=1000:0 --from=builder /opt/jailbreak-detect /opt/jailbreak-detect

USER 1000
EXPOSE 8000
ENTRYPOINT ["python", "server.py", "start"]
CMD ["--port=8000"]
