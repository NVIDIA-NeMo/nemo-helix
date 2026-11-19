# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG HELIX_IMAGE=nvcr.io/nvidia/nemo-helix/nhx-api:0.10.0@sha256:5f4d6ae0485ce102e55a9d08a74925a2381f82d3abbc02d820ef1a914329913e
FROM ${HELIX_IMAGE} AS codex-download
ARG CODEX_RELEASE_URL=https://github.com/openai/codex/releases/download/rust-v0.162.1
RUN set -eu; \
    arch="$(uname -m)"; \
    case "$arch" in \
        x86_64) digest=a676f5722aae0d86cbe3764332f08eab1fdd89dc0a300b3ad4ffbf48611fa3e3 ;; \
        aarch64) digest=53a9ce0b94beb27a63fdeb2f7704d95199e17b3898aa33b60461a1a93ac0d88b ;; \
        *) echo "Unsupported Codex architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl --fail --silent --show-error --location --retry 3 --proto '=https' --proto-redir '=https' \
        "$CODEX_RELEASE_URL/codex-package-${arch}-unknown-linux-musl.tar.gz" -o /tmp/codex.tar.gz; \
    printf '%s  /tmp/codex.tar.gz\n' "$digest" | sha256sum -c -; \
    mkdir -p /opt/codex; \
    tar -xzf /tmp/codex.tar.gz -C /opt/codex; \
    test -x /opt/codex/bin/codex; \
    rm /tmp/codex.tar.gz

FROM ${HELIX_IMAGE}
USER 0
ENTRYPOINT []
CMD []
HEALTHCHECK NONE
WORKDIR /workspace
ENV HOME=/home/harbor-agent \
    NEMO_TELEMETRY_ENABLED=false \
    PATH=/app/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY --from=codex-download /opt/codex /opt/codex
COPY image_setup.py skill-removals.toml helix-config.yaml start-helix start_helix.py /opt/helix-task/
RUN set -eu; \
    groupadd --gid 10001 harbor-agent; \
    useradd --uid 10001 --gid 10001 --home-dir /home/harbor-agent --create-home --shell /bin/bash harbor-agent; \
    groupadd --gid 10002 helix-task; \
    useradd --uid 10002 --gid 10002 --home-dir /var/lib/helix-task/service \
        --no-create-home --shell /usr/sbin/nologin helix-task; \
    install -d -m 0755 /harbor/skills /var/lib/helix-task; \
    install -d -m 0700 /tests /var/lib/helix-task/root /var/lib/helix-task/evidence; \
    install -d -m 0700 -o helix-task -g helix-task /var/lib/helix-task/service; \
    install -d -m 0755 -o harbor-agent -g harbor-agent /home/harbor-agent/.agents/skills; \
    ln -s /opt/codex/bin/codex /usr/local/bin/codex; \
    chmod 0755 /opt/helix-task/start-helix; \
    chown harbor-agent:harbor-agent /workspace; \
    /app/.venv/bin/python /opt/helix-task/image_setup.py; \
    setpriv --reuid=10001 --regid=10001 --clear-groups codex --version | grep -Fx 'codex-cli 0.162.1'
