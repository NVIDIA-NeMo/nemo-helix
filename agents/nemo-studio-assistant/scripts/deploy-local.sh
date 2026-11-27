#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Register, package, and deploy the Studio Assistant on a local platform.
#
# Usage:
#   deploy-local.sh             Deploy the Studio assistant (writes pause for Studio's approval card).
#   deploy-local.sh --terminal  Deploy a terminal assistant for 'nemo agents chat' (writes need a
#                               confirmation token, and an off-topic gate wraps the model).
#
# Re-running replaces the Studio registration and deployment, so prompt and
# skill edits take effect. The terminal topic gate is created once per
# workspace: if its backend model or policy changes, remove the existing
# gate first (the error message says how). The terminal agent reads its model
# from NEMO_TERMINAL_MODEL, which this script sets to the gate, so registering
# agent.terminal.yaml by hand without the gate fails instead of running
# ungated. Requires a running platform, Docker, the repo's pinned uv (run
# inside `flox activate`), and NEMO_DEFAULT_MODEL set to a model entity id
# from `nemo models list`. Packaging uses a nemo-helix wheel built from this
# checkout unless NEMO_AGENTS_WHEEL is already set.
set -euo pipefail

terminal=false
case "${1:-}" in
  "") ;;
  --terminal) terminal=true ;;
  *) echo "usage: deploy-local.sh [--terminal]" >&2; exit 2 ;;
esac

export NHX_BASE_URL="${NHX_BASE_URL:-http://127.0.0.1:8080}"

agent_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${agent_dir}/../.." && pwd)"
cd "${repo_root}"

nemo() { uv run --frozen nemo "$@"; }
step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [[ "${terminal}" == true ]]; then
  agent_name="${TERMINAL_ASSISTANT_NAME:-nemo-studio-assistant-terminal}"
  package_config="agents/nemo-studio-assistant/agent.terminal.yaml"
  register_config="agents/nemo-studio-assistant/agent.terminal.yaml"
  image_suffix="terminal"
else
  agent_name="${STUDIO_ASSISTANT_NAME:-nemo-studio-assistant}"
  package_config="agents/nemo-studio-assistant/agent.yaml"
  register_config="agents/nemo-studio-assistant-ethos/agent.yaml"
  image_suffix="local"
fi
deployment_name="${agent_name}-fabric-deployment"
image_tag="${agent_name}:fabric-${image_suffix}"

step "Checking prerequisites"
curl -sf "${NHX_BASE_URL}/health/ready" >/dev/null \
  || fail "no platform is ready at ${NHX_BASE_URL}; start it with 'nemo services run' (see SETUP.md)"
docker info >/dev/null 2>&1 || fail "Docker is not running; the assistant deploys as a Docker container"
[[ -n "${NEMO_DEFAULT_MODEL:-}" ]] \
  || fail "set NEMO_DEFAULT_MODEL to a model entity id from 'nemo models list', for example default/<entity-id>"
if [[ "${terminal}" == true && "${NEMO_DEFAULT_MODEL}" != */* ]]; then
  fail "NEMO_DEFAULT_MODEL must be '<workspace>/<model>' for the terminal topic gate"
fi
echo "Platform ${NHX_BASE_URL}, model ${NEMO_DEFAULT_MODEL}"

if [[ "${terminal}" == true ]]; then
  step "Creating the terminal topic gate over ${NEMO_DEFAULT_MODEL}"
  gate_workspace="${NEMO_DEFAULT_MODEL%%/*}"
  gate_model="$(uv run --frozen --project agents/nemo-studio-assistant \
    python -m nemo_studio_assistant.topic_gate \
    --workspace "${gate_workspace}" --backend-model "${NEMO_DEFAULT_MODEL}")" \
    || fail "the topic gate was not created; see the output above"
  export NEMO_TERMINAL_MODEL="${gate_model}"
  echo "Terminal agent model ${NEMO_TERMINAL_MODEL}"
fi

step "Snapshotting docs for search_docs"
"${agent_dir}/scripts/snapshot-docs.sh"

if [[ -z "${NEMO_AGENTS_WHEEL:-}" ]]; then
  step "Building the nemo-helix wheel from this checkout"
  uv build --package nemo-helix --wheel --out-dir dist
  export NEMO_AGENTS_WHEEL=LATEST
fi

step "Packaging ${image_tag}"
nemo agents package \
  --agent "${package_config}" \
  --pyproject agents/nemo-studio-assistant/pyproject.toml \
  --tag "${image_tag}"

step "Removing any previous ${agent_name} deployment and registration"
nemo agents undeploy "${deployment_name}" --yes >/dev/null 2>&1 || echo "No previous deployment"
for _ in $(seq 1 60); do
  nemo agents deployments get "${deployment_name}" >/dev/null 2>&1 || break
  sleep 2
done
nemo agents deployments get "${deployment_name}" >/dev/null 2>&1 \
  && fail "previous deployment ${deployment_name} is still being removed; re-run in a minute"
nemo agents delete "${agent_name}" --yes >/dev/null 2>&1 || echo "No previous registration"

step "Registering ${agent_name}"
nemo agents register --name "${agent_name}" \
  --agent-config "${register_config}"

step "Deploying ${deployment_name}"
nemo agents deploy --agent "${agent_name}" \
  --name "${deployment_name}" \
  --mode docker --image "${image_tag}"

step "Smoke test"
nemo agents invoke --agent-deployment "${deployment_name}" \
  --input "In one sentence, what can you help me with in NeMo Studio?"

if [[ "${terminal}" == true ]]; then
  cat <<EOF

${agent_name} is running in the terminal. Chat with it:
  nemo agents chat --agent-deployment ${deployment_name}
Writes ask for confirmation in the terminal. Off-topic requests are refused by the topic gate
${NEMO_DEFAULT_MODEL}, which the agent calls in place of the backend model.
Tear down with:
  nemo agents undeploy ${deployment_name} --yes && nemo agents delete ${agent_name} --yes
EOF
else
  cat <<EOF

${agent_name} is running. Open ${NHX_BASE_URL}/studio and use the assistant in the top bar.
The Studio topic gate uses STUDIO_ASSISTANT_TOPIC_GATE_MODEL, or NEMO_DEFAULT_MODEL when unset,
from the environment of the process running 'nemo services run'.
Tear down with:
  nemo agents undeploy ${deployment_name} --yes && nemo agents delete ${agent_name} --yes
EOF
fi
