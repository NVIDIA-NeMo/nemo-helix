# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local compose image for the Phase-1 scaled-evals plugin: the platform API and
# both scaled-evals workers run from this one image.
#
# This is NOT the platform's shipping image. `docker/Dockerfile.nmp-api` builds
# that one through bake, from base images that live in internal registries, and
# it installs the `functional-services` group — which does not include this
# ephemeral plugin. Here we install the workspace scoped to the plugin instead
# (`--package`), which drops ~370 packages (guardrails, transformers, NAT) that
# a scaled-evals-only API never imports.
#
# Build from the repo root:
#   docker build -f plugins/_temporary-scaled-evals/deploy/compose/Dockerfile .

ARG PYTHON_VERSION=3.12
# Client and daemon come from the same image so buildctl cannot skew past buildkitd.
ARG BUILDKIT_IMAGE=moby/buildkit:rootless@sha256:946cf909534b789c9b84af75e6d4f8cb23c3b9cb387daec5ba0f4878a3e4647c
ARG KUBECTL_VERSION=v1.33.2

FROM ${BUILDKIT_IMAGE} AS buildkit

FROM --platform=$BUILDPLATFORM golang:1.26.7 AS jobs-launcher
ARG TARGETOS
ARG TARGETARCH
COPY services/core/jobs/jobs-launcher /app/jobs-launcher
WORKDIR /app/jobs-launcher
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -a -installsuffix cgo -ldflags="-s -w -extldflags '-static'" -trimpath -o jobs-launcher ./main.go

# =============================================================================
# Harbor runner — the thing that actually executes an evaluation.
#
# The dispatch worker spawns `$HARBOR_DIR/.venv/bin/harbor run` as a subprocess,
# and Harbor's K8sSandboxEnvironment (from sandbox-k8s) creates the Sandbox CRs.
# Neither is a dependency of the plugin, so both are installed here rather than
# through uv sync. sandbox-k8s is pinned to the version matching the cluster's
# agent-sandbox CRD; harbor comes from the curated catalog the plugin ships.
#
# =============================================================================
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS harbor

# Additional package index for the harbor / sandbox-k8s wheels, for deployments
# that mirror them privately. Empty means PyPI only, so a build has to opt in
# rather than inherit somebody else's index:
#   --build-arg HARBOR_EXTRA_INDEX_URL=https://<index>/simple
ARG HARBOR_EXTRA_INDEX_URL=""

COPY plugins/_temporary-scaled-evals/src/scaled_evals/data/harbor_runner_qualifications.json \
     /opt/harbor/qualification-catalog.json

RUN set -eu; \
    python -c 'import json,sys; c=json.load(open(sys.argv[1])); print(" ".join(v for v,r in c["releases"].items() if r["selectable"]))' \
        /opt/harbor/qualification-catalog.json > /opt/harbor/selectable-versions.txt; \
    python -c 'import json,sys; c=json.load(open(sys.argv[1])); v=c["default_version"]; assert c["releases"][v]["selectable"]; print(v)' \
        /opt/harbor/qualification-catalog.json > /opt/harbor/default-version.txt; \
    python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sandbox_k8s"]["version"])' \
        /opt/harbor/qualification-catalog.json > /opt/harbor/sandbox-k8s-version.txt; \
    test -s /opt/harbor/selectable-versions.txt; \
    python -c 'import urllib.request; urllib.request.urlretrieve("https://bootstrap.pypa.io/pip/pip.pyz", "/opt/harbor/pip.pyz")'

RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    sandbox_k8s_version="$(cat /opt/harbor/sandbox-k8s-version.txt)"; \
    for version in $(cat /opt/harbor/selectable-versions.txt); do \
        runner="/opt/harbor/${version}"; \
        uv venv "${runner}/.venv"; \
        VIRTUAL_ENV="${runner}/.venv" uv pip install \
            "harbor==${version}" "sandbox-k8s[harbor]==${sandbox_k8s_version}" \
            ${HARBOR_EXTRA_INDEX_URL:+--extra-index-url "$HARBOR_EXTRA_INDEX_URL"}; \
        "${runner}/.venv/bin/python" -c \
            'import importlib.metadata as m, sys; assert m.version("harbor") == sys.argv[1]; assert m.version("sandbox-k8s") == sys.argv[2]' \
            "${version}" "${sandbox_k8s_version}"; \
    done

# Upstream sandbox-k8s lacks emptyDir mounts for /solution, /tests, /logs and
# /installed-agent, which the oracle verifier path needs. Overlay the same
# patched adapter the standalone image uses, in place, located via importlib.
COPY plugins/_temporary-scaled-evals/harbor-patches/sandbox_k8s_harbor.py /tmp/sandbox_k8s_harbor.py
COPY plugins/_temporary-scaled-evals/harbor-patches/patch_sandbox_k8s_root.py /tmp/patch_sandbox_k8s_root.py
RUN set -eu; for version in $(cat /opt/harbor/selectable-versions.txt); do \
        runner="/opt/harbor/${version}"; \
        harbor_py="$("${runner}/.venv/bin/python" -c 'import importlib.util; print(importlib.util.find_spec("sandbox_k8s.harbor").origin)')"; \
        cp /tmp/sandbox_k8s_harbor.py "$harbor_py"; \
        "${runner}/.venv/bin/python" /tmp/patch_sandbox_k8s_root.py "$(dirname "$harbor_py")"; \
    done

# Harbor 0.20 changed both adapters. Keep their writable-volume and environment
# compatibility patches generic so older selectable runners remain buildable.
COPY plugins/_temporary-scaled-evals/harbor-patches/patch_langgraph_writable_venv.py /tmp/patch_langgraph_writable_venv.py
COPY plugins/_temporary-scaled-evals/harbor-patches/patch_pi_extra_env.py /tmp/patch_pi_extra_env.py
RUN set -eu; for version in $(cat /opt/harbor/selectable-versions.txt); do \
        runner="/opt/harbor/${version}"; \
        langgraph_py="$("${runner}/.venv/bin/python" -c 'import importlib.util; s=importlib.util.find_spec("harbor.agents.installed.langgraph"); print(s.origin if s else "")')"; \
        if [ -n "$langgraph_py" ]; then \
            "${runner}/.venv/bin/python" /tmp/patch_langgraph_writable_venv.py "$langgraph_py"; \
        fi; \
        pi_py="$("${runner}/.venv/bin/python" -c 'import importlib.util; s=importlib.util.find_spec("harbor.agents.installed.pi"); print(s.origin if s else "")')"; \
        if [ -n "$pi_py" ]; then \
            "${runner}/.venv/bin/python" /tmp/patch_pi_extra_env.py "$pi_py"; \
        fi; \
    done

# Harbor resolves jobs_dir relative to its own root, so point it at the path the
# runtime mounts writable. /opt/harbor itself stays read-only.
RUN set -eu; \
    for version in $(cat /opt/harbor/selectable-versions.txt); do \
        runner="/opt/harbor/${version}"; \
        test -x "${runner}/.venv/bin/harbor"; \
        rm -rf "${runner}/jobs"; \
        ln -s /var/lib/scaled-evals/harbor-jobs "${runner}/jobs"; \
    done; \
    ln -s "$(cat /opt/harbor/default-version.txt)" /opt/harbor/default

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

# Both the API (`GET /v1/readyz`) and the build worker shell out to buildctl.
COPY --from=buildkit /usr/bin/buildctl /usr/local/bin/buildctl
COPY --from=jobs-launcher /app/jobs-launcher/jobs-launcher /tools/jobs-launcher

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY . /app

# nmp-platform-runner is what serves the plugin (`nemo services run`); it is not
# in the plugin's dependency closure, since the runner loads plugins rather than
# the other way round. It costs 3 extra packages.
#
# nemo-platform-ext owns the implementation imported by the `nemo` SDK wrapper.
# The SDK aliases it at runtime but does not declare it as a dependency because
# release wheels vendor the package instead. Source-checkout images need it.
#
# nmp-entities is not needed by this plugin at all: the runner's startup banner
# imports `nmp.core.entities.config` unconditionally to print the database it is
# using, but does not declare nmp-entities as a dependency. A full `uv sync`
# hides that, so it only shows up in a scoped install like this one. 28 packages.
RUN --mount=type=cache,target=/root/.cache/uv \
    apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends build-essential \
    && uv sync --frozen --no-dev \
        --package nemo-scaled-evals-plugin \
        --package nmp-platform-runner \
        --package nemo-platform-ext \
        --package nmp-auth \
        --package nmp-entities \
        --package nmp-files \
        --package nmp-jobs \
        --package nmp-secrets \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"

# Dispatch tears down sandboxes, pods and networkpolicies by shelling out to
# kubectl, so it has to be on PATH even though nothing else in the image uses it.
ARG KUBECTL_VERSION
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends curl ca-certificates \
    && arch="$(dpkg --print-architecture)" \
    && curl -fsSLo /usr/local/bin/kubectl \
        "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Harbor venv (with the patched adapter) plus the task harness tree that
# SANDBOX_K8S_CONFIG_PATH and TASK_PATH point into.
COPY --from=harbor /opt/harbor /opt/harbor
COPY plugins/_temporary-scaled-evals/examples/tasks /harness/tasks
COPY plugins/_temporary-scaled-evals/deploy/k8s/sandbox.env /harness/sandbox.env
# 0777 because the runtime picks the uid; the k8s manifests mount a writable
# volume over this path anyway, so this only matters for bare `docker run`.
RUN mkdir -p /var/lib/scaled-evals/harbor-jobs \
    && chmod 0777 /var/lib/scaled-evals/harbor-jobs
ENV HARBOR_DIR=/opt/harbor/default

EXPOSE 8080

CMD ["nemo", "services", "run", "--services", "scaled-evals", "--host", "0.0.0.0", "--port", "8080"]
