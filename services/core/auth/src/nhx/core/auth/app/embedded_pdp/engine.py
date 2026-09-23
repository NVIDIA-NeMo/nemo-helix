# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OPA WASM Policy Engine using wasmtime."""

import json
import logging
import threading
from functools import cache
from typing import Any, Dict, List, Optional, cast

from nhx.core.auth.app.embedded_pdp.policy_wasm import ensure_embedded_policy_wasm
from wasmtime import Config, Engine, Func, FuncType, Instance, Limits, Memory, MemoryType, Module, Store, ValType

logger = logging.getLogger(__name__)

# Entrypoint IDs (order of -e flags in: opa build -e authz/allow -e authz/has_permissions -e authz/has_role)
ENTRYPOINT_MAP = {"allow": 0, "has_permissions": 1, "has_role": 2}

DATA_LOADING_FUEL = 10_000_000_000


class PolicyEngineError(Exception):
    """Error during policy evaluation."""


@cache
def _get_engine() -> Engine:
    """Return the process-wide wasmtime Engine.

    Engine setup is the heavyweight step in wasmtime — JIT code generation and process-wide
    trap/signal-handling registration — and wasmtime's own guidance is one Engine per process with
    many cheap Stores created from it. Creating a fresh Engine per thread-local OPAPolicy would
    churn through many Engines over an xdist worker's lifetime; that churn, not any single test, is
    what was crashing workers with no traceback ("node down: Not properly terminated").
    """
    config = Config()
    config.consume_fuel = True
    return Engine(config)


class OPAPolicy:
    """Wrapper for OPA WASM policy evaluation."""

    def __init__(self, wasm_path: str, *, fuel_limit: int = 200_000_000, memory_limit_mb: int = 32):
        self._owner_thread_id = threading.get_ident()
        engine = _get_engine()

        self.fuel_limit = fuel_limit
        self.store = Store(engine)
        self.store.set_fuel(DATA_LOADING_FUEL)
        self.store.set_limits(memory_size=memory_limit_mb * 1024 * 1024)
        module = Module.from_file(engine, wasm_path)

        # OPA requires these imports (in order they appear in the module)
        # env::opa_builtin0..4, env::opa_abort, env::memory
        self.memory = Memory(self.store, MemoryType(Limits(16, memory_limit_mb * 16)))

        imports = [
            Func(self.store, FuncType([ValType.i32(), ValType.i32()], [ValType.i32()]), lambda a, b: 0),  # builtin0
            Func(
                self.store, FuncType([ValType.i32(), ValType.i32(), ValType.i32()], [ValType.i32()]), lambda a, b, c: 0
            ),  # builtin1
            Func(
                self.store,
                FuncType([ValType.i32(), ValType.i32(), ValType.i32(), ValType.i32()], [ValType.i32()]),
                lambda a, b, c, d: 0,
            ),  # builtin2
            Func(
                self.store,
                FuncType([ValType.i32(), ValType.i32(), ValType.i32(), ValType.i32(), ValType.i32()], [ValType.i32()]),
                lambda a, b, c, d, e: 0,
            ),  # builtin3
            Func(
                self.store,
                FuncType(
                    [ValType.i32(), ValType.i32(), ValType.i32(), ValType.i32(), ValType.i32(), ValType.i32()],
                    [ValType.i32()],
                ),
                lambda a, b, c, d, e, f: 0,
            ),  # builtin4
            Func(self.store, FuncType([ValType.i32()], []), lambda addr: None),  # opa_abort
            self.memory,
        ]

        self.instance = Instance(self.store, module, imports)
        self.exports = self.instance.exports(self.store)
        self._base_heap = self._export_func("opa_heap_ptr_get")(self.store)
        self._data_heap = self._base_heap
        self._data_addr: Optional[int] = None

    def _assert_owner_thread(self) -> None:
        current_thread_id = threading.get_ident()
        if current_thread_id != self._owner_thread_id:
            raise RuntimeError(
                f"OPAPolicy used from a different thread (owner={self._owner_thread_id}, current={current_thread_id})"
            )

    def _export_func(self, name: str) -> Func:
        self._assert_owner_thread()
        return cast(Func, self.exports[name])

    def _write_json(self, data: Any) -> int:
        """Write JSON to WASM memory, return OPA value address."""
        self._assert_owner_thread()
        json_bytes = json.dumps(data).encode("utf-8")
        addr = self._export_func("opa_malloc")(self.store, len(json_bytes))
        self.memory.write(self.store, json_bytes, addr)
        return self._export_func("opa_json_parse")(self.store, addr, len(json_bytes))

    def _read_json(self, addr: int) -> Any:
        """Read OPA value as JSON from WASM memory."""
        self._assert_owner_thread()
        json_addr = self._export_func("opa_json_dump")(self.store, addr)
        mem = self.memory.data_ptr(self.store)
        end = json_addr
        while mem[end] != 0:
            end += 1
        return json.loads(bytes(mem[json_addr:end]).decode("utf-8"))

    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the base data document."""
        self._assert_owner_thread()
        self.store.set_fuel(DATA_LOADING_FUEL)
        self._export_func("opa_heap_ptr_set")(self.store, self._base_heap)
        self._data_addr = self._write_json(data)
        self._data_heap = self._export_func("opa_heap_ptr_get")(self.store)

    def evaluate(self, input_data: Dict[str, Any], entrypoint: int = 0) -> Any:
        """Evaluate policy with given input."""
        self._assert_owner_thread()
        if self._data_addr is None:
            raise PolicyEngineError("Policy data not loaded — refusing to evaluate (fail-closed)")

        self.store.set_fuel(self.fuel_limit)

        heap_base = getattr(self, "_data_heap", self._base_heap)
        self._export_func("opa_heap_ptr_set")(self.store, heap_base)

        ctx = self._export_func("opa_eval_ctx_new")(self.store)
        self._export_func("opa_eval_ctx_set_input")(self.store, ctx, self._write_json(input_data))
        self._export_func("opa_eval_ctx_set_data")(self.store, ctx, self._data_addr)
        self._export_func("opa_eval_ctx_set_entrypoint")(self.store, ctx, entrypoint)

        self._export_func("eval")(self.store, ctx)
        return self._read_json(self._export_func("opa_eval_ctx_get_result")(self.store, ctx))


class _PolicyRuntimeManager:
    """Owns policy data snapshots and thread-local WASM policy runtimes."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._data_loaded = False
        self._data_version = 0
        self._generation = 0

    def _clear_thread_policy(self) -> None:
        for attr in ("policy", "policy_generation", "policy_data_version"):
            if hasattr(self._local, attr):
                delattr(self._local, attr)

    def _create_policy(self) -> OPAPolicy:
        from nhx.common.config import get_service_config
        from nhx.core.auth.config import AuthServiceConfig

        cfg = get_service_config(AuthServiceConfig)
        path = ensure_embedded_policy_wasm(auto_build=cfg.embedded_pdp_auto_build_wasm)
        return OPAPolicy(
            str(path),
            fuel_limit=cfg.embedded_pdp_cpu_limit * 1_000_000,
            memory_limit_mb=cfg.embedded_pdp_memory_limit_mb,
        )

    def _data_snapshot(self) -> tuple[Dict[str, Any], int, bool]:
        with self._lock:
            return self._data, self._data_version, self._data_loaded

    def _generation_snapshot(self) -> int:
        with self._lock:
            return self._generation

    def _get_thread_policy(self) -> Optional[OPAPolicy]:
        return cast(Optional[OPAPolicy], getattr(self._local, "policy", None))

    def _sync_data_if_needed(self, policy: OPAPolicy) -> None:
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

    def get_policy(self) -> OPAPolicy:
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

    def set_data(self, data: Dict[str, Any]) -> None:
        """Set policy data (principals, roles, etc.)."""
        with self._lock:
            self._data = data
            self._data_loaded = True
            self._data_version += 1

        policy = self._get_thread_policy()
        if policy is not None and getattr(self._local, "policy_generation", -1) == self._generation_snapshot():
            self._sync_data_if_needed(policy)

    def reload(self) -> None:
        """Force each thread to rebuild its policy runtime on next access."""
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


def get_policy() -> OPAPolicy:
    """Get or create the current thread's policy runtime."""
    return _policy_runtime.get_policy()


def set_policy_data(data: Dict[str, Any]) -> None:
    """Set policy data (principals, roles, etc.)."""
    _policy_runtime.set_data(data)


def reload_policy() -> None:
    """Force each thread to rebuild its policy runtime on next access."""
    _policy_runtime.reload()


def _reset_policy_state_for_testing() -> None:
    """Reset module policy state between tests."""
    _policy_runtime.reset_for_testing()


def evaluate(entrypoint: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a policy entrypoint."""
    if entrypoint not in ENTRYPOINT_MAP:
        raise PolicyEngineError(f"Invalid entrypoint: {entrypoint}. Valid: {list(ENTRYPOINT_MAP.keys())}")

    try:
        result = get_policy().evaluate(input_data, ENTRYPOINT_MAP[entrypoint])
    except Exception as exc:
        msg = str(exc)
        if "all fuel consumed" in msg:
            raise PolicyEngineError(f"Policy evaluation exceeded fuel limit for entrypoint '{entrypoint}'") from exc
        if "memory" in msg.lower():
            raise PolicyEngineError(f"Policy evaluation exceeded memory limit for entrypoint '{entrypoint}'") from exc
        raise PolicyEngineError(f"WASM execution error: {msg}") from exc

    # OPA returns [[{result: ...}]] - unwrap it
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict) and "result" in result:
        result = result["result"]

    if not result:
        return {
            "allow": {"allowed": False, "headers": {"X-NHX-Authorized": "false"}},
            "has_permissions": {"allowed": False},
            "has_role": {"has_role": False},
        }.get(entrypoint, {})

    return result


def validate_entrypoint(entrypoint: str) -> bool:
    return entrypoint in ENTRYPOINT_MAP


def get_valid_entrypoints() -> List[str]:
    return list(ENTRYPOINT_MAP.keys())
