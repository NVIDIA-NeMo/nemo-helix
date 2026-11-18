#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Register, package, and deploy the Studio Assistant on a local platform.
#
# Re-running replaces the existing registration and deployment, so prompt and
# skill edits take effect. Requires a running platform, Docker, and
# NEMO_DEFAULT_MODEL set to a model entity id from `nemo models list`.
set -euo pipefail

agent_name="${STUDIO_ASSISTANT_NAME:-nemo-studio-assistant}"
deployment_name="${agent_name}-fabric-deployment"
image_tag="${agent_name}:fabric-local"
export NHX_BASE_URL="${NHX_BASE_URL:-http://127.0.0.1:8080}"

agent_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${agent_dir}/../.." && pwd)"
cd "${repo_root}"

nemo() { uv run --frozen nemo "$@"; }
step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

step "Checking prerequisites"
curl -sf "${NHX_BASE_URL}/health/ready" >/dev/null \
  || fail "no platform is ready at ${NHX_BASE_URL}; start it with 'nemo services run' (see SETUP.md)"
docker info >/dev/null 2>&1 || fail "Docker is not running; the assistant deploys as a Docker container"
[[ -n "${NEMO_DEFAULT_MODEL:-}" ]] \
  || fail "set NEMO_DEFAULT_MODEL to a model entity id from 'nemo models list', for example default/<entity-id>"
echo "Platform ${NHX_BASE_URL}, model ${NEMO_DEFAULT_MODEL}"

step "Snapshotting docs for search_docs"
"${agent_dir}/scripts/snapshot-docs.sh"

step "Removing any previous ${agent_name} deployment and registration"
nemo agents undeploy "${deployment_name}" --yes >/dev/null 2>&1 || echo "No previous deployment"
nemo agents delete "${agent_name}" --yes >/dev/null 2>&1 || echo "No previous registration"

step "Registering ${agent_name}"
nemo agents register --name "${agent_name}" \
  --agent-config agents/nemo-studio-assistant-ethos/agent.yaml

step "Packaging ${image_tag}"
nemo agents package \
  --agent agents/nemo-studio-assistant/agent.yaml \
  --pyproject agents/nemo-studio-assistant/pyproject.toml \
  --tag "${image_tag}"

step "Deploying ${deployment_name}"
nemo agents deploy --agent "${agent_name}" \
  --name "${deployment_name}" \
  --mode docker --image "${image_tag}"

step "Smoke test"
nemo agents invoke --agent-deployment "${deployment_name}" \
  --input "In one sentence, what can you help me with in NeMo Studio?"

cat <<EOF

${agent_name} is running. Open ${NHX_BASE_URL}/studio and use the assistant in the top bar.
The Studio topic gate uses STUDIO_ASSISTANT_TOPIC_GATE_MODEL, or NEMO_DEFAULT_MODEL when unset,
from the environment of the process running 'nemo services run'.
Tear down with:
  nemo agents undeploy ${deployment_name} --yes && nemo agents delete ${agent_name} --yes
EOF
