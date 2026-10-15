#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Dev-only: install the `nemo-relay` GATEWAY BINARY, the one Fabric eval dependency that cannot come
# from a wheel. It is required for live ATIF trajectory capture on out-of-process harnesses (codex).
#
# Everything else is in the lock —
# `uv sync --frozen --package nhx-evals-sdk --extra fabric --inexact` installs the nemo-fabric
# SDK, the codex/claude/hermes adapters, and the nemo-relay Python bindings without removing the
# existing workspace environment. The pip `nemo-relay` package is bindings-only (its wheel declares
# no console script and contains no executable), so the daemon is published solely as a GitHub
# release asset.
#
# The version defaults to the `nemo-relay` bindings installed in the venv, so the daemon and the
# bindings cannot drift apart when the lock moves.
#
# To run against an unreleased Fabric instead of the locked wheels, install the checkout directly:
#   uv pip install --python .venv/bin/python "/path/to/NeMo-Fabric[codex,relay,runtime]"
# and use the package-scoped command above to restore the locked project dependencies without
# pruning unrelated installed packages. (That needs cargo — Fabric builds a Rust/pyo3 extension
# from source.)
#
# A live codex run additionally needs the `codex` CLI + `codex login` auth.
# See skills/nemo-evals-plugin/references/agent-evaluation.md and
# packages/nhx_evals_sdk/tests/agent_eval/test_fabric_integration.py.
#
# Usage:
#   script/dev-install-fabric.sh                                # version matching the installed bindings
#   NEMO_RELAY_VERSION=0.9.3 script/dev-install-fabric.sh        # pin a specific gateway release
set -euo pipefail

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Project venv not found at $VENV_PY. Run 'make bootstrap-python' (see SETUP.md) first." >&2
  exit 1
fi

# NeMo-Relay tags releases with the version PyPI publishes, but PyPI normalizes prereleases
# (for example, 0.7.0rc1) while the git tag is semver (0.7.0-rc.1), so convert.
if [ -z "${NEMO_RELAY_VERSION:-}" ]; then
  bindings_version="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("nemo-relay"))' 2>/dev/null || true)"
  if [ -z "$bindings_version" ]; then
    echo "nemo-relay is not installed in $VENV_PY, so the gateway version cannot be derived." >&2
    echo "Run 'uv sync --frozen --package nhx-evals-sdk --extra fabric --inexact' first," >&2
    echo "or pass NEMO_RELAY_VERSION=<release> explicitly." >&2
    exit 1
  fi
  NEMO_RELAY_VERSION="$(printf '%s' "$bindings_version" | sed -E 's/([0-9])(a|b|rc)\.?([0-9]+)$/\1-\2.\3/')"
  echo "Using nemo-relay gateway ${NEMO_RELAY_VERSION} to match the installed bindings (${bindings_version})."
fi

# Skip only when an existing nemo-relay already matches, so an explicit version request is honored
# rather than silently short-circuited by any PATH match.
if command -v nemo-relay >/dev/null 2>&1; then
  # `|| true`: a broken nemo-relay on PATH exits non-zero, and under `set -e -o pipefail` that would
  # abort here — in the very branch that exists to replace it. An empty version just means "reinstall".
  installed_relay_ver="$(nemo-relay --version 2>/dev/null | awk '{print $NF}' || true)"
  if [ "$installed_relay_ver" = "$NEMO_RELAY_VERSION" ]; then
    echo "nemo-relay gateway already on PATH at ${installed_relay_ver}: $(command -v nemo-relay)"
    exit 0
  fi
  echo "nemo-relay ${installed_relay_ver:-?} on PATH differs from requested ${NEMO_RELAY_VERSION}; (re)installing ..."
fi

# Host platform -> NeMo-Relay release target triple. Every platform in the workspace's
# [tool.uv] environments has a published asset.
case "$(uname -s):$(uname -m)" in
  Darwin:arm64) relay_target="aarch64-apple-darwin" ;;
  Linux:x86_64) relay_target="x86_64-unknown-linux-musl" ;;
  Linux:aarch64 | Linux:arm64) relay_target="aarch64-unknown-linux-musl" ;;
  *)
    echo "No published nemo-relay gateway for $(uname -s):$(uname -m)." >&2
    echo "See https://github.com/NVIDIA/NeMo-Relay/releases for available targets." >&2
    exit 1
    ;;
esac

RELAY_BIN_DIR="${CARGO_HOME:-$HOME/.cargo}/bin"
asset="nemo-relay-cli-${relay_target}-${NEMO_RELAY_VERSION}"
base="https://github.com/NVIDIA/NeMo-Relay/releases/download/${NEMO_RELAY_VERSION}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading nemo-relay gateway ${NEMO_RELAY_VERSION} (${relay_target}) from GitHub releases ..."
curl -fsSL -o "${tmp}/nemo-relay" "${base}/${asset}"
curl -fsSL -o "${tmp}/nemo-relay.sha256" "${base}/${asset}.sha256"
want="$(awk '{print $1}' "${tmp}/nemo-relay.sha256")"
if command -v sha256sum >/dev/null 2>&1; then
  got="$(sha256sum "${tmp}/nemo-relay" | awk '{print $1}')"
else
  got="$(shasum -a 256 "${tmp}/nemo-relay" | awk '{print $1}')"
fi
if [ "$want" != "$got" ]; then
  echo "Checksum mismatch for ${asset}: expected ${want}, got ${got}" >&2
  exit 1
fi

mkdir -p "$RELAY_BIN_DIR"
install -m 0755 "${tmp}/nemo-relay" "${RELAY_BIN_DIR}/nemo-relay"
echo "nemo-relay gateway installed: ${RELAY_BIN_DIR}/nemo-relay ($("${RELAY_BIN_DIR}/nemo-relay" --version 2>/dev/null || echo '?'))"

# Warn if the install dir isn't on PATH — live tests resolve the gateway via shutil.which().
case ":${PATH}:" in
  *":${RELAY_BIN_DIR}:"*) : ;;
  *) echo "NOTE: ${RELAY_BIN_DIR} is not on PATH — add it so 'nemo-relay' is found: export PATH=\"${RELAY_BIN_DIR}:\$PATH\"" >&2 ;;
esac
