#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test that the `nemo` CLI on PATH actually works.
#
# Install-method agnostic: whatever put `nemo` on PATH (`uv tool install` from
# a wheel, `uv sync` in the dev workspace, `pipx`, a distro package, ...) is
# the caller's concern. This script only exercises the CLI and the platform
# it boots.
#
# Used by .github/workflows/ci.yaml's wheel-test job; runs locally too.
#
# Env vars:
#   WHEEL_TEST_TIMEOUT   seconds to wait for platform health (default: 60)
set -euo pipefail
# Job control: backgrounded jobs get their own process group, so
# `kill -- -$!` later takes down the whole tree (parent + worker
# subprocesses spawned by `nemo services run`) without depending on
# `setsid` (which isn't standard on macOS).
set -m

# Default to non-TTY rendering so Typer / Rich don't try to draw progress bars
# or wrap based on a fake terminal width. Caller can override.
export _TYPER_FORCE_DISABLE_TERMINAL="${_TYPER_FORCE_DISABLE_TERMINAL:-1}"

LOG="${RUNNER_TEMP:-/tmp}/nemo-services.log"
TIMEOUT_SECS="${WHEEL_TEST_TIMEOUT:-60}"

cleanup() {
  set +e
  if [[ -n "${SERVICES_PID:-}" ]]; then
    kill -TERM -- "-${SERVICES_PID}" 2>/dev/null
    for _ in 1 2 3 4 5; do
      kill -0 "${SERVICES_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVICES_PID}" 2>/dev/null
  fi
  if [[ -f "${LOG}" ]]; then
    echo "----- nemo services log -----"
    cat "${LOG}"
  fi
}
trap cleanup EXIT

# Required runtime tools. jq is preinstalled on standard GH runners; surface
# a clear error for local users running this script with an incomplete env.
for cmd in nemo jq; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "::error::${cmd} is not on PATH; install it before running this script" >&2
    exit 1
  fi
done

echo "----- nemo on PATH -----"
command -v nemo
version_output="$(nemo --version)"
echo "${version_output}"
# A bare `0.0.0` is the CLI's fallback when its importlib.metadata lookup
# misses the distribution it was installed from, which is invisible unless the
# output is asserted. Match it exactly: nightly wheels are stamped with a
# sentinel `0.0.0.dev<timestamp>` that is a legitimate version string. The
# pattern is the subset of PEP 440 this project stamps — release, optional
# pre-release, any number of .post/.dev segments, optional local version.
version="${version_output#nemo version }"
if [[ "${version}" == "0.0.0" ]] ||
  [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?(\.(post|dev)[0-9]+)*(\+[0-9a-z.]+)?$ ]]; then
  echo "::error::nemo --version did not report an installed version: ${version_output}" >&2
  exit 1
fi

# Import-time checks for commands that ship in the bundled wrapper. We don't
# call `nemo services --help` here because `nemo services run` below is a
# stronger functional check of that group.
echo "----- import-time CLI checks -----"
nemo --help >/dev/null
nemo agents --help >/dev/null
nemo skills list >/dev/null

echo "----- boot platform and wait for default+system workspaces -----"
# No --services / --controllers: the CLI's default is "every service the
# install knows about." For a wheel-test job that's exactly what we want —
# any bundled service module that fails to import, run alembic, or answer
# health probes is a real wheel-installability bug. The caller must have
# installed `nemo-helix[services]` (the umbrella extra) for this to
# resolve a meaningful set; with the base install you'd get just the SDK.
nemo services run >"${LOG}" 2>&1 &
SERVICES_PID=$!

# The entities service seeds `default` and `system` back-to-back in
# services/core/entities/.../initialize.py — requiring both catches the
# case where seeding partially succeeded then crashed. JSON output is
# used so the assertion survives Rich table rendering changes.
deadline=$(($(date +%s) + TIMEOUT_SECS))
until output="$(nemo workspaces list -f json 2>/dev/null)" \
  && jq -e '
       any(.data[]?; .name == "default")
       and any(.data[]?; .name == "system")
     ' >/dev/null <<<"${output}"; do
  if (($(date +%s) > deadline)); then
    echo "::error::platform did not become healthy with default+system workspaces in ${TIMEOUT_SECS}s"
    echo "----- last \`nemo workspaces list -f json\` output (stdout+stderr) -----"
    nemo workspaces list -f json 2>&1 || true
    # Full services log is dumped by the EXIT trap.
    exit 1
  fi
  if ! kill -0 "${SERVICES_PID}" 2>/dev/null; then
    # Reap so the error message carries the actual exit code.
    wait "${SERVICES_PID}" 2>/dev/null
    services_exit=$?
    echo "::error::nemo services run exited (code ${services_exit}) before becoming healthy"
    SERVICES_PID=""  # cleanup shouldn't re-kill
    exit 1
  fi
  sleep 2
done

echo "platform healthy; default+system workspaces present"
