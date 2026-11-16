#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run test_taskset_run_controller.py the way CI does: host an auth-off platform at NHX_AGENT_BASE_URL and an
# auth-on one at NHX_AGENT_AUTH_BASE_URL with launch_test_platform.py, wait for both, then run only that file
# against them. Only that file, because the shared conftest fixtures read the same variables as the port to start
# their own platform on. Every test has to run here, so a skip fails the run.
#
# From the repository root: bash .github/scripts/run-taskset-run-controller-integration.sh
# Logs, platform data and the JUnit report go to $RUNNER_TEMP/controller-platforms, or a new temp dir locally.
set -euo pipefail
: "${NHX_AGENT_BASE_URL:=http://localhost:18090}" "${NHX_AGENT_AUTH_BASE_URL:=http://localhost:18092}"
export NHX_AGENT_BASE_URL NHX_AGENT_AUTH_BASE_URL
logs="${RUNNER_TEMP:-$(mktemp -d)}/controller-platforms"
mkdir -p "${logs}"
echo "Platform logs, data and the JUnit report: ${logs}"

launcher=plugins/nemo-evals/tests/integration/launch_test_platform.py
auth_off_port="${NHX_AGENT_BASE_URL##*:}"
auth_on_port="${NHX_AGENT_AUTH_BASE_URL##*:}"
uv run python "${launcher}" --port "${auth_off_port}" --work-dir "${logs}/auth-off" > "${logs}/auth-off.log" 2>&1 &
auth_off=$!
uv run python "${launcher}" --port "${auth_on_port}" --auth --work-dir "${logs}/auth-on" > "${logs}/auth-on.log" 2>&1 &
auth_on=$!
# Stop both platforms however the run ends, a readiness failure included.
trap 'kill "${auth_off}" "${auth_on}" 2>/dev/null || true' EXIT

deadline=$((SECONDS + 300))
for url in "${NHX_AGENT_BASE_URL}" "${NHX_AGENT_AUTH_BASE_URL}"; do
  # --max-time: a platform that takes the connection but never answers must not outlast the deadline.
  until curl -sf --max-time 10 "${url}/health/ready" > /dev/null; do
    # A platform that exited never becomes ready: fail now, not at the deadline.
    if ! { kill -0 "${auth_off}" && kill -0 "${auth_on}"; } 2>/dev/null || [ "${SECONDS}" -ge "${deadline}" ]; then
      echo "::error::platform at ${url} is not ready: a platform exited, or 300s passed" >&2
      tail -n 200 "${logs}"/*.log
      exit 1
    fi
    sleep 2
  done
done

status=0
uv run pytest plugins/nemo-evals/tests/integration/test_taskset_run_controller.py \
  -m integration -n 0 -v -rs --junitxml="${logs}/junit.xml" || status=$?
if [ "${status}" -eq 0 ] && ! grep -q ' skipped="0"' "${logs}/junit.xml"; then
  echo "::error::taskset-run controller tests skipped instead of running against both platforms" >&2
  status=1
fi
if [ "${status}" -ne 0 ]; then tail -n 200 "${logs}"/*.log; fi
exit "${status}"
