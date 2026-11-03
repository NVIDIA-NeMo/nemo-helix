# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The image for a Hermes-harness Fabric agent, owned by its author rather than rendered.
#
# It is a normal Fabric image plus the war-game requirements, plus TWO Hermes-only lines that do
# what a `telemetry:` block does for the DeepAgents victim — because Hermes reaches Relay through
# an opt-in plugin, not through the adapter's telemetry wiring.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

# iproute2 and a `sandbox` user: the OpenShell sandbox needs both.  [war-game]
RUN apt-get update \
 && apt-get install -y --no-install-recommends iproute2 git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system sandbox \
 && useradd --system --gid sandbox --create-home --home-dir /home/sandbox --shell /bin/bash sandbox

WORKDIR /workspace

# The main runtime uses Relay 0.11; Fabric 0.6's Hermes adapter needs Relay 0.10
# and Python 3.14 in a separate environment, just as in generated agent images.
RUN uv venv --python 3.12 /workspace/.venv \
 && uv pip install --python /workspace/.venv/bin/python \
      "nemo-helix[nemo-agents-plugin]" \
      "mcp[cli]"
RUN uv venv --python 3.14 /opt/hermes-venv \
 && git init --quiet /opt/hermes-agent \
 && git -C /opt/hermes-agent remote add origin https://github.com/NousResearch/hermes-agent.git \
 && git -C /opt/hermes-agent fetch --depth 1 origin bf0203a794a07aac3ff764972ab9087baa4e4f4e \
 && git -C /opt/hermes-agent checkout --quiet --detach FETCH_HEAD \
 && /workspace/.venv/bin/python -m nemo_agents_plugin.container.install_hermes \
      --checkout /opt/hermes-agent --python /opt/hermes-venv/bin/python \
 && chmod -R a+rX /opt/uv /opt/hermes-agent /opt/hermes-venv
ENV ADAPTER_PYTHON=/opt/hermes-venv/bin/python

COPY agent.yaml /workspace/agent.yaml
COPY ledger_mcp.py /app/ledger_mcp.py

# Relay reads system-layer plugin config from here, and Agent Hardener uploads each round's guardrails
# into it before the victim starts.  [war-game]
RUN mkdir -p /etc/nemo-relay /home/sandbox/.agent-hardener/relay \
 && chown -R sandbox:sandbox /etc/nemo-relay /home/sandbox/.agent-hardener /workspace /app

# --- The two Hermes-only lines. [hermes] ---
#
# Hermes reads Relay's plugin config ONLY from this env var — not from /etc/nemo-relay, which Relay
# discovers on its own for every other integration. Point it at the file Agent Hardener uploads into.
ENV HERMES_NEMO_RELAY_PLUGINS_TOML=/etc/nemo-relay/plugins.toml
# Relay is an opt-in Hermes plugin. Without enabling it, its middleware never registers and the env
# var above is read by nobody — the victim runs, emits nothing, and the run's preflight fails it.
#
# Enable it as `sandbox`, not root: Hermes keeps plugin state per-user (~/.hermes), so enabling as
# root leaves it "not enabled" for the user the victim actually runs as.
USER sandbox
RUN /opt/hermes-venv/bin/hermes plugins enable observability/nemo_relay
USER root

ENV AGENT_CONFIG_PATH=/workspace/agent.yaml
ENV PORT=8000
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/workspace/.venv

USER sandbox
EXPOSE 8000

# The Fabric contract: serve the agent on $PORT. Run explicitly rather than via ENTRYPOINT metadata,
# because `openshell sandbox exec` does not propagate image ENV (neither PATH nor AGENT_CONFIG_PATH).
ENTRYPOINT ["sh", "-c", "exec python -m nemo_agents_plugin.fabric.server --agent-config \"$AGENT_CONFIG_PATH\" --host 0.0.0.0 --port \"$PORT\""]
