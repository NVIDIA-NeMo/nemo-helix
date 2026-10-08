# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Packages the LangGraph agent and its HTTP server into one image.
#
#   docker build -t langgraph-victim:v1 examples/langgraph-victim
#   docker run --rm -p 8000:8000 -e INFERENCE_API_KEY=... langgraph-victim:v1
FROM python:3.12-slim

# iproute2 and a `sandbox` user: what the OpenShell sandbox needs if this image is war-gamed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends iproute2 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system sandbox \
 && useradd --system --gid sandbox --create-home --home-dir /home/sandbox --shell /bin/bash sandbox

WORKDIR /app

# The agent's own dependencies — no agent-hardener. Pinned to the ranges the repo resolves.
# nemo-relay carries the [langgraph] extra: that is what pulls the callback-handler integration.
RUN pip install --no-cache-dir \
      "langgraph>=1.2,<2" \
      "langchain-core>=1.4,<2" \
      "langchain-openai>=1.1.9,<2.0" \
      "nemo-relay[langgraph]>=0.8,<1.0" \
      "fastapi>=0.115,<1.0" \
      "uvicorn>=0.30,<1.0"

COPY agent.py server.py /app/

# Where each round's guardrails are uploaded, and where the ATOF stream is written. Both must exist
# and be writable by the runtime user before the victim starts — a missing /etc/nemo-relay fails the
# run at deploy with `mkdir: cannot create directory '/etc/nemo-relay': Permission denied`.
RUN mkdir -p /etc/nemo-relay /home/sandbox/.agent-hardener/relay \
 && chown -R sandbox:sandbox /etc/nemo-relay /home/sandbox/.agent-hardener /app

ENV PORT=8000
# Unbuffered so tool output and tracebacks reach `docker logs` as they happen.
ENV PYTHONUNBUFFERED=1

USER sandbox
EXPOSE 8000

CMD ["python", "/app/server.py"]
