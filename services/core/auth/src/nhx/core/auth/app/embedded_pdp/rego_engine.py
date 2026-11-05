# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpret shared Rego sources with the native Regorus Python binding."""

import json
import threading
from pathlib import Path
from typing import Any

# Upstream re-exports its PyO3 extension dynamically and ships no Engine stub.
from regorus import Engine  # ty: ignore[unresolved-import]

POLICY_DIRECTORY = Path(__file__).parent.parent / "policies"
ENTRYPOINTS = ("allow", "has_permissions", "has_role")


class PolicyEngineError(Exception):
    """Error loading or evaluating an embedded policy."""


class RegoPolicy:
    """Thread-confined native interpreter with replaceable authorization data.

    Policies are parsed/prepared in memory, not compiled to a repository artifact.
    External OPA evaluates the same source and data. This uses Regorus's Engine
    API, not its separate RVM bytecode API or an embedded WASM runtime.
    """

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._modules = {path.name: path.read_text() for path in sorted(POLICY_DIRECTORY.glob("*.rego"))}
        if not self._modules:
            raise PolicyEngineError(f"No Rego policies found in {POLICY_DIRECTORY}")
        self._data_loaded = False
        self._engine = self._compile({})

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("RegoPolicy used from a different thread")

    def _compile(self, data: dict[str, Any]) -> Engine:
        engine = Engine()
        for name, source in self._modules.items():
            engine.add_policy(name, source)
        # JSON avoids Python binding numeric coercions and preserves complete
        # strings/object keys, including embedded NUL and non-ASCII characters.
        engine.add_data_json(json.dumps(data, allow_nan=False))
        engine.set_input_json('{"method":"GET","path":"/health/live"}')
        # Regorus prepares rules lazily. Validate the source during initialization
        # or refresh, before publishing a runtime with an unprepared policy.
        engine.eval_rule_as_json("data.authz.allow")
        return engine

    def set_data(self, data: dict[str, Any]) -> None:
        """Replace the runtime so removed grants cannot survive a data refresh."""
        self._assert_owner_thread()
        self._data_loaded = False
        try:
            engine = self._compile(data)
        except Exception as exc:
            raise PolicyEngineError(f"Unable to load Rego policy data: {exc}") from exc
        self._engine = engine
        self._data_loaded = True

    def evaluate(self, input_data: dict[str, Any], entrypoint: int = 0) -> list[dict[str, Any]]:
        """Evaluate an entrypoint, retaining the historical PDP result envelope."""
        self._assert_owner_thread()
        if not self._data_loaded:
            raise PolicyEngineError("Policy data not loaded — refusing to evaluate (fail-closed)")
        if entrypoint not in range(len(ENTRYPOINTS)):
            raise PolicyEngineError(f"Invalid entrypoint: {entrypoint}")
        try:
            self._engine.set_input_json(json.dumps(input_data, allow_nan=False))
            result = json.loads(self._engine.eval_rule_as_json(f"data.authz.{ENTRYPOINTS[entrypoint]}"))
        except Exception as exc:
            raise PolicyEngineError(f"Rego evaluation failed: {exc}") from exc
        if not isinstance(result, dict):
            raise PolicyEngineError("Rego entrypoint returned a non-object decision")
        verdict = "has_role" if ENTRYPOINTS[entrypoint] == "has_role" else "allowed"
        if type(result.get(verdict)) is not bool:
            raise PolicyEngineError(f"Rego entrypoint returned an invalid '{verdict}' decision")
        return [{"result": result}]
