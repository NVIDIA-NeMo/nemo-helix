# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72 AS codex-download
ARG CODEX_RELEASE_URL=https://github.com/openai/codex/releases/download/rust-v0.162.1
RUN set -eu; \
    arch="$(uname -m)"; \
    case "$arch" in \
        x86_64) digest=a676f5722aae0d86cbe3764332f08eab1fdd89dc0a300b3ad4ffbf48611fa3e3 ;; \
        aarch64) digest=53a9ce0b94beb27a63fdeb2f7704d95199e17b3898aa33b60461a1a93ac0d88b ;; \
        *) echo "Unsupported Codex architecture: $arch" >&2; exit 1 ;; \
    esac; \
    python -c 'import shutil, sys, urllib.request; response = urllib.request.urlopen(sys.argv[1], timeout=120); shutil.copyfileobj(response, sys.stdout.buffer)' \
        "$CODEX_RELEASE_URL/codex-package-${arch}-unknown-linux-musl.tar.gz" > /tmp/codex.tar.gz; \
    printf '%s  /tmp/codex.tar.gz\n' "$digest" | sha256sum -c -; \
    mkdir -p /opt/codex; \
    tar -xzf /tmp/codex.tar.gz -C /opt/codex

FROM python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72
WORKDIR /workspace
COPY --from=codex-download /opt/codex /opt/codex
RUN groupadd --gid 10001 harbor-agent \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash harbor-agent \
    && chown harbor-agent:harbor-agent /workspace \
    && ln -s /opt/codex/bin/codex /usr/local/bin/codex \
    && codex --version | grep -Fx 'codex-cli 0.162.1'
