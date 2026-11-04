# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedded Rego policy evaluation with thread-local regopy interpreters."""

import threading
from typing import Any, cast

from nhx.core.auth.app.embedded_pdp.rego_engine import ENTRYPOINTS, PolicyEngineError, RegoPolicy
from nhx.core.auth.app.scope_families import with_scope_families

ENTRYPOINT_MAP = {name: index for index, name in enumerate(ENTRYPOINTS)}
# Preserve the public class name for callers that only use set_data/evaluate.
# Construction no longer accepts WASM paths or runtime resource-limit arguments.
OPAPolicy = RegoPolicy


class _PolicyRuntimeManager:
    """Own policy data snapshots and thread-local Rego interpreters."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._data_loaded = False
        self._data_version = 0
        self._generation = 0

    def _clear_thread_policy(self) -> None:
        for attr in ("policy", "policy_generation", "policy_data_version"):
            if hasattr(self._local, attr):
                delattr(self._local, attr)

    def _create_policy(self) -> RegoPolicy:
        return RegoPolicy()

    def _data_snapshot(self) -> tuple[dict[str, Any], int, bool]:
        with self._lock:
            return self._data, self._data_version, self._data_loaded

    def _generation_snapshot(self) -> int:
        with self._lock:
            return self._generation

    def _get_thread_policy(self) -> RegoPolicy | None:
        return cast(RegoPolicy | None, getattr(self._local, "policy", None))

    def _sync_data_if_needed(self, policy: RegoPolicy) -> None:
        while True:
            local_version = cast(int, getattr(self._local, "policy_data_version", -1))
            data, data_version, data_loaded = self._data_snapshot()
            if local_version == data_version:
                return

            if data_loaded:
                policy.set_data(data)

            _, current_data_version, _ = self._data_snapshot()
            if data_version == current_data_version:
                self._local.policy_data_version = data_version
                return

    def get_policy(self) -> RegoPolicy:
        """Get or create the current thread's policy runtime."""
        while True:
            generation = self._generation_snapshot()
            policy = self._get_thread_policy()
            policy_generation = cast(int, getattr(self._local, "policy_generation", -1))
            if policy is None or policy_generation != generation:
                policy = self._create_policy()
                self._local.policy = policy
                self._local.policy_generation = generation
                self._local.policy_data_version = -1

            self._sync_data_if_needed(policy)
            if generation == self._generation_snapshot():
                return policy

    def set_data(self, data: dict[str, Any]) -> None:
        """Publish a replacement policy-data snapshot."""
        with self._lock:
            self._data = data
            self._data_loaded = True
            self._data_version += 1

        policy = self._get_thread_policy()
        if policy is not None and getattr(self._local, "policy_generation", -1) == self._generation_snapshot():
            self._sync_data_if_needed(policy)

    def reload(self) -> None:
        """Force each thread to reload Rego source on next access."""
        with self._lock:
            self._generation += 1
        self._clear_thread_policy()
        self.get_policy()

    def reset_for_testing(self) -> None:
        """Reset policy runtime state between tests."""
        with self._lock:
            self._data = {}
            self._data_loaded = False
            self._data_version += 1
            self._generation += 1
        self._clear_thread_policy()


_policy_runtime = _PolicyRuntimeManager()


def get_policy() -> RegoPolicy:
    """Get or create the current thread's policy runtime."""
    return _policy_runtime.get_policy()


def set_policy_data(data: dict[str, Any]) -> None:
    """Replace authorization data, deriving scope families for scopes.rego."""
    _policy_runtime.set_data(with_scope_families(data))


def reload_policy() -> None:
    """Force each thread to reload its policy runtime on next access."""
    _policy_runtime.reload()


def _reset_policy_state_for_testing() -> None:
    """Reset module policy state between tests."""
    _policy_runtime.reset_for_testing()


def evaluate(entrypoint: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a policy entrypoint, failing closed on interpreter errors."""
    if entrypoint not in ENTRYPOINT_MAP:
        raise PolicyEngineError(f"Invalid entrypoint: {entrypoint}. Valid: {list(ENTRYPOINT_MAP)}")

    try:
        result = get_policy().evaluate(input_data, ENTRYPOINT_MAP[entrypoint])
    except Exception as exc:
        raise PolicyEngineError(f"Rego execution error: {exc}") from exc

    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict) and "result" in result:
        result = result["result"]

    if not result:
        return {
            "allow": {"allowed": False, "headers": {"X-NHX-Authorized": "false"}},
            "has_permissions": {"allowed": False},
            "has_role": {"has_role": False},
        }[entrypoint]

    if not isinstance(result, dict):
        raise PolicyEngineError(f"Unexpected Rego result for '{entrypoint}'")
    return result


def validate_entrypoint(entrypoint: str) -> bool:
    return entrypoint in ENTRYPOINT_MAP


def get_valid_entrypoints() -> list[str]:
    return list(ENTRYPOINT_MAP)
