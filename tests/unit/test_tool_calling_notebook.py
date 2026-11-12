# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the tutorial's route-readiness logic without a platform or GPU."""

import ast
import json
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from unittest.mock import Mock

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "docs/example-applications/tool-calling.ipynb"


class RouteNotFound(Exception):
    pass


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _inference_cell() -> ast.Module:
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "def wait_for_model_gateway(" in source:
            return ast.parse(source)
    raise AssertionError("Tutorial route-readiness cell is missing")


def _waiter(gateway: Mock, clock: Clock) -> Callable[..., None]:
    definition = next(
        node
        for node in _inference_cell().body
        if isinstance(node, ast.FunctionDef) and node.name == "wait_for_model_gateway"
    )
    namespace = {"igw": gateway, "time": clock, "NotFoundError": RouteNotFound, "WORKSPACE": "default"}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    function = namespace["wait_for_model_gateway"]
    assert isinstance(function, FunctionType)
    return function


@pytest.mark.parametrize("name", ["base", "base&adapters/default/adapter"])
def test_waits_for_model_discovery_before_returning(name: str) -> None:
    gateway = Mock()
    gateway.model_get.side_effect = [RouteNotFound("not discovered"), Mock()]
    clock = Clock()

    _waiter(gateway, clock)(name)

    assert clock.now == 2
    assert gateway.model_get.call_count == 2
    gateway.model_get.assert_called_with(workspace="default", name=name, trailing_uri="v1/models")


def test_route_readiness_times_out_with_model_reference() -> None:
    gateway = Mock()
    gateway.model_get.side_effect = RouteNotFound("not discovered")
    clock = Clock()

    with pytest.raises(TimeoutError, match="default/base"):
        _waiter(gateway, clock)("base", timeout_seconds=4)

    assert clock.now == 4
    assert gateway.model_get.call_count == 2


def test_does_not_retry_non_discovery_errors() -> None:
    gateway = Mock()
    gateway.model_get.side_effect = PermissionError("denied")

    with pytest.raises(PermissionError, match="denied"):
        _waiter(gateway, Clock())("base")

    assert gateway.model_get.call_count == 1


def test_waits_for_each_route_before_its_inference_request() -> None:
    calls: list[ast.stmt] = [
        node
        for node in _inference_cell().body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in {"wait_for_model_gateway", "test_tool_calling"}
    ]
    events = []
    namespace = {
        "WORKSPACE": "default",
        "MODEL_NAME": "base",
        "ADAPTER_NAME": "adapter",
        "BASE_MODEL_REF": "default/base",
        "ADAPTER_MODEL_REF": "default/base&adapters/default/adapter",
        "wait_for_model_gateway": lambda name: events.append(("wait", name)),
        "test_tool_calling": lambda name, label: events.append(("infer", name)),
    }

    exec(compile(ast.Module(body=calls, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)

    assert events == [
        ("wait", "base"),
        ("infer", "default/base"),
        ("wait", "base&adapters/default/adapter"),
        ("infer", "default/base&adapters/default/adapter"),
    ]
