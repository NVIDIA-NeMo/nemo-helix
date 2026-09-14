# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The image for a Fabric agent, owned by its author rather than rendered.
#
# `nemo agents package` can render one, but a rendered Dockerfile pins the packaging machine's own
# nemo-platform version — from a git checkout that is something like `0.4.0.post96.dev0+789b8466e`,
# which no index serves — and a fixed nemo-relay an agent cannot ask to change. Registering this file
# beside agent.yaml uploads it with the bundle, and Agent Hardener reads it back in preference.
#
# It is a normal Fabric image plus three things the war-game needs, each marked below.
FROM python:3.12-slim

# iproute2 and a `sandbox` user: the OpenShell sandbox needs both.  [war-game]
RUN apt-get update \
 && apt-get install -y --no-install-recommends iproute2 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system sandbox \
 && useradd --system --gid sandbox --create-home --home-dir /home/sandbox --shell /bin/bash sandbox

WORKDIR /workspace

# The Fabric entrypoint, its DeepAgents adapter, and the MCP SDK the tool server uses.
#
# nemo-relay is pinned <0.8 because the Fabric adapters cap it there. That is also why the uploaded
# plugins.toml uses observability schema 3: 0.7.3 rejects 4 outright, and a rejected schema fails the
# whole plugin initialize() — taking the guardrail component in the same file down with it.
#
# No `RUN --mount=type=cache`: OpenShell builds through the Docker Engine API's classic builder,
# which rejects BuildKit-only syntax. Agent Hardener strips these flags anyway; not writing them is one
# less surprise.  [war-game]
# nemo-fabric and the harness adapter are named explicitly: the published
# nemo-platform[nemo-agents-plugin] ships nemo_agents_plugin.fabric.server but does not pull the
# Fabric runtime it imports, so the entrypoint dies at startup with
# `ModuleNotFoundError: No module named 'nemo_fabric'`.
RUN python -m venv /workspace/.venv \
 && /workspace/.venv/bin/pip install --no-cache-dir \
      "nemo-platform[nemo-agents-plugin]" \
      "nemo-fabric" \
      "nemo-fabric-adapters-deepagents" \
      "nemo-relay>=0.7.2,<0.8" \
      "mcp[cli]"

COPY agent.yaml /workspace/agent.yaml
COPY ledger_mcp.py /workspace/ledger_mcp.py

# Relay reads system-layer plugin config from here, and Agent Hardener uploads each round's guardrails
# into it before the victim starts. Missing directory → the run fails at deploy with
# `mkdir: cannot create directory '/etc/nemo-relay': Permission denied`, surfaced as a tar-extract
# error. `/etc` is the one layer NeMo Fabric accepts plugin config from.  [war-game]
RUN mkdir -p /etc/nemo-relay /home/sandbox/.agent-hardener/relay \
 && chown -R sandbox:sandbox /etc/nemo-relay /home/sandbox/.agent-hardener /workspace

ENV AGENT_CONFIG_PATH=/workspace/agent.yaml
ENV PORT=8000
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/workspace/.venv

USER sandbox
EXPOSE 8000

# The Fabric contract: serve the agent on $PORT. Agent Hardener runs this command explicitly rather than
# relying on the ENTRYPOINT, because `openshell sandbox exec` does not propagate image ENV — neither
# PATH nor AGENT_CONFIG_PATH is visible to it.
ENTRYPOINT ["sh", "-c", "exec python -m nemo_agents_plugin.fabric.server --agent-config \"$AGENT_CONFIG_PATH\" --host 0.0.0.0 --port \"$PORT\""]
