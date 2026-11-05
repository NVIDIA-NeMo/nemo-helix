# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-owned native OPA runtime shared by request threads."""

import json
import os
import threading
from typing import Any

from nhx.common.config import get_service_config
from nhx.core.auth.app.embedded_pdp.rego_engine import (
    ENTRYPOINTS,
    PolicyEngineError,
    RegoPolicy,
)
from nhx.core.auth.app.embedded_pdp.rego_engine import (
    check_policy as _check_policy,
)
from nhx.core.auth.app.scope_families import with_scope_families
from nhx.core.auth.config import AuthServiceConfig

ENTRYPOINT_MAP = {name: index for index, name in enumerate(ENTRYPOINTS)}
OPAPolicy = RegoPolicy


class _PolicyRuntimeManager:
    """Own one child per service process, never one child per request thread."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner_pid = os.getpid()
        self._policy: RegoPolicy | None = None
        self._data: dict[str, Any] = {}
        self._data_json: str | None = None
        self._data_loaded = False
        self._stopped = False

    def _after_fork(self) -> None:
        if self._owner_pid != os.getpid():
            # Inherited clients, locks and data belong to the parent. Runtime
            # finalizers also check ownership before touching its child/socket.
            self._lock = threading.RLock()
            self._owner_pid = os.getpid()
            self._policy = None
            self._data = {}
            self._data_json = None
            self._data_loaded = False
            self._stopped = False

    def _create_policy(self) -> RegoPolicy:
        config = get_service_config(AuthServiceConfig)
        return RegoPolicy(
            binary_path=config.opa_binary_path,
            startup_timeout_seconds=config.opa_startup_timeout_seconds,
            request_timeout_seconds=config.opa_request_timeout_seconds,
            shutdown_timeout_seconds=config.opa_shutdown_timeout_seconds,
        )

    def get_policy(self) -> RegoPolicy:
        self._after_fork()
        with self._lock:
            if self._stopped:
                raise PolicyEngineError("Managed OPA has been stopped")
            if self._policy is None:
                self._policy = self._create_policy()
            return self._policy

    def start(self) -> RegoPolicy:
        self._after_fork()
        with self._lock:
            self._stopped = False
            if self._policy is not None and not self._policy.is_running():
                self._policy.close()
                self._policy = None
                self._data = {}
                self._data_json = None
                self._data_loaded = False
            return self.get_policy()

    def set_data(self, data: dict[str, Any]) -> None:
        self._after_fork()
        with self._lock:
            try:
                content = json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":"))
                document = json.loads(content)
            except Exception:
                self.invalidate_data()
                raise
            policy = self.get_policy()
            if self._data_loaded and self._data_json == content and policy.is_ready():
                return
            self._data_loaded = False
            policy.invalidate_data()
            try:
                policy.set_data(document)
            except Exception:
                self._data = {}
                self._data_json = None
                raise
            self._data = document
            self._data_json = content
            self._data_loaded = True

    def invalidate_data(self) -> None:
        self._after_fork()
        with self._lock:
            self._data_loaded = False
            self._data = {}
            self._data_json = None
            if self._policy is not None:
                self._policy.invalidate_data()

    def is_ready(self) -> bool:
        self._after_fork()
        with self._lock:
            return not self._stopped and self._data_loaded and self._policy is not None and self._policy.is_ready()

    def reload(self) -> None:
        self._after_fork()
        with self._lock:
            if self._stopped:
                raise PolicyEngineError("Managed OPA has been stopped")
            data, data_loaded = self._data, self._data_loaded
            self._data_loaded = False
            if self._policy is not None:
                self._policy.invalidate_data()
                self._policy.close()
                self._policy = None
            candidate = None
            try:
                candidate = self._create_policy()
                if data_loaded:
                    candidate.set_data(data)
            except Exception:
                self._data = {}
                self._data_json = None
                if candidate is not None:
                    # Retain ownership if bounded cleanup itself needs a retry.
                    self._policy = candidate
                    candidate.close()
                    self._policy = None
                raise
            self._policy = candidate
            self._data_loaded = data_loaded

    def close(self) -> None:
        self._after_fork()
        with self._lock:
            self._stopped = True
            self._data_loaded = False
            self._data = {}
            self._data_json = None
            if self._policy is not None:
                self._policy.close()
            self._policy = None

    def reset_for_testing(self) -> None:
        self.close()
        with self._lock:
            self._stopped = False


_policy_runtime = _PolicyRuntimeManager()


def get_policy() -> RegoPolicy:
    """Return the service process's shared managed OPA instance."""
    return _policy_runtime.get_policy()


def start_policy() -> RegoPolicy:
    """Start native OPA during the owning auth service's startup."""
    return _policy_runtime.start()


def close_policy() -> None:
    """Stop the child and prohibit implicit restart after shutdown."""
    _policy_runtime.close()


def is_policy_ready() -> bool:
    return _policy_runtime.is_ready()


def invalidate_policy_data() -> None:
    _policy_runtime.invalidate_data()


def check_policy() -> None:
    """One-shot preflight, without leaking a daemon outside service lifespan."""
    config = get_service_config(AuthServiceConfig)
    _check_policy(config.opa_binary_path, config.opa_startup_timeout_seconds)


def set_policy_data(data: dict[str, Any]) -> None:
    """Replace authorization data, deriving the same shared scope families."""
    try:
        document = with_scope_families(data)
    except Exception:
        invalidate_policy_data()
        raise
    _policy_runtime.set_data(document)


def reload_policy() -> None:
    _policy_runtime.reload()


def _reset_policy_state_for_testing() -> None:
    _policy_runtime.reset_for_testing()


def evaluate(entrypoint: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate only public entrypoints, failing closed on runtime errors."""
    if entrypoint not in ENTRYPOINT_MAP:
        raise PolicyEngineError(f"Invalid entrypoint: {entrypoint}. Valid: {list(ENTRYPOINT_MAP)}")
    try:
        result = get_policy().evaluate(input_data, ENTRYPOINT_MAP[entrypoint])
    except Exception as exc:
        raise PolicyEngineError(f"OPA execution error: {exc}") from exc
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    if not isinstance(result, dict):
        raise PolicyEngineError(f"Unexpected OPA result for '{entrypoint}'")
    verdict = "has_role" if entrypoint == "has_role" else "allowed"
    if type(result.get(verdict)) is not bool:
        raise PolicyEngineError(f"Unexpected OPA '{verdict}' decision for '{entrypoint}'")
    return result


def validate_entrypoint(entrypoint: str) -> bool:
    return entrypoint in ENTRYPOINT_MAP


def get_valid_entrypoints() -> list[str]:
    return list(ENTRYPOINT_MAP)
