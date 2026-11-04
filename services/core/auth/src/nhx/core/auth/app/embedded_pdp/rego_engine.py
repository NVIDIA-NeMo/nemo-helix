# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpret the shared Rego source without an OPA executable or WASM artifact."""

import json
import threading
from pathlib import Path
from typing import Any

from regopy import Bundle, Interpreter
from regopy.rego_shared import rego_get_error

POLICY_DIRECTORY = Path(__file__).parent.parent / "policies"
ENTRYPOINTS = ("allow", "has_permissions", "has_role")


class PolicyEngineError(Exception):
    """Error loading or evaluating an embedded policy."""


def _native_error(interpreter: Interpreter) -> str:
    """Read diagnostics without formatting a potentially null native Output.

    regopy 1.5.2 checks pointers against 0 rather than None in several wrappers.
    A compiler failure can therefore yield an invalid Bundle/Output instead of
    raising RegoError. Formatting such an Output dereferences NULL in native code.
    Keep this version-specific diagnostic access confined to this helper.
    """
    return rego_get_error(interpreter._impl) or "native Rego evaluation failed"


class RegoPolicy:
    """Thread-confined Rego interpreter with compiled, replaceable data snapshots.

    Compilation happens in memory at initialization and data refresh, never as a
    repository build step. Source and data are shared with the external OPA PDP.
    There are no WASM fuel or linear-memory limits in this backend.
    """

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._modules = {path.name: path.read_text() for path in sorted(POLICY_DIRECTORY.glob("*.rego"))}
        if not self._modules:
            raise PolicyEngineError(f"No Rego policies found in {POLICY_DIRECTORY}")
        self._data_loaded = False
        # Validate the shipped source at startup, but refuse authorization until
        # real policy data has been explicitly loaded (including an empty document).
        self._interpreter, self._bundle = self._compile({})

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("RegoPolicy used from a different thread")

    def _compile(self, data: dict[str, Any]) -> tuple[Interpreter, Bundle]:
        interpreter = Interpreter()
        for name, source in self._modules.items():
            interpreter.add_module(name, source)
        interpreter.add_data(data)
        bundle = interpreter.build(None, [f"authz/{name}" for name in ENTRYPOINTS])
        if not bundle.ok():
            raise PolicyEngineError(f"Rego compilation failed: {_native_error(interpreter)}")
        return interpreter, bundle

    def set_data(self, data: dict[str, Any]) -> None:
        """Replace data, rather than merging revoked grants into the previous data."""
        self._assert_owner_thread()
        # A failed refresh must not leave this runtime authorizing with stale grants.
        self._data_loaded = False
        try:
            interpreter, bundle = self._compile(data)
        except Exception as exc:
            raise PolicyEngineError(f"Unable to load Rego policy data: {exc}") from exc
        self._interpreter, self._bundle = interpreter, bundle
        self._data_loaded = True

    def evaluate(self, input_data: dict[str, Any], entrypoint: int = 0) -> list[dict[str, Any]]:
        """Evaluate an entrypoint, retaining the historical PDP result envelope."""
        self._assert_owner_thread()
        if not self._data_loaded:
            raise PolicyEngineError("Policy data not loaded — refusing to evaluate (fail-closed)")
        if entrypoint not in range(len(ENTRYPOINTS)):
            raise PolicyEngineError(f"Invalid entrypoint: {entrypoint}")
        # Input(dict) passes individual strings through a NUL-terminated C ABI,
        # truncating identities, scopes, and object keys containing U+0000.
        # Escaped JSON terms preserve the complete request at that boundary.
        self._interpreter.set_input_term(json.dumps(input_data, allow_nan=False))
        output = self._interpreter.query_bundle_entrypoint(self._bundle, f"authz/{ENTRYPOINTS[entrypoint]}")
        if not output.ok():
            raise PolicyEngineError(f"Rego evaluation failed: {_native_error(self._interpreter)}")
        # Output.results already owns decoded Python values. Avoid retaining native
        # Node objects across queries, or dereferencing an unsuccessful Output.
        if len(output) == 1 and not output[0].expressions:
            return []
        if len(output) != 1 or len(output[0].expressions) != 1:
            raise PolicyEngineError("Unexpected Rego result shape")
        result = output[0].expressions[0]
        if not isinstance(result, dict):
            raise PolicyEngineError("Rego entrypoint returned a non-object decision")
        return [{"result": result}]
