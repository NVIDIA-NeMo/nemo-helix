# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The image for a Hermes-harness Fabric agent, owned by its author rather than rendered.
#
# Hermes reads the guardrails uploaded by Agent Hardener through its Relay config override.
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

# Helix uses the base image's Python and Relay 0.11. Hermes requires Python 3.14
# and Relay 0.10 in a separate environment.
RUN uv venv --python /usr/local/bin/python /workspace/.venv \
 && uv pip install --python /workspace/.venv/bin/python \
      "nemo-helix[nemo-agents-plugin]>=0.10,<0.11" "mcp[cli]"
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

# Point Hermes' built-in Relay integration at the config Agent Hardener uploads.
ENV HERMES_NEMO_RELAY_PLUGINS_TOML=/etc/nemo-relay/plugins.toml

ENV AGENT_CONFIG_PATH=/workspace/agent.yaml
ENV PORT=8000
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/workspace/.venv

USER sandbox
EXPOSE 8000

# The Fabric contract: serve the agent on $PORT. Run explicitly rather than via ENTRYPOINT metadata,
# because `openshell sandbox exec` does not propagate image ENV (neither PATH nor AGENT_CONFIG_PATH).
ENTRYPOINT ["sh", "-c", "exec python -m nemo_agents_plugin.fabric.server --agent-config \"$AGENT_CONFIG_PATH\" --host 0.0.0.0 --port \"$PORT\""]
