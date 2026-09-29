# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from ipaddress import ip_address
from urllib.parse import urlparse

import pytest

pytest.importorskip("harbor")

from nemo_evaluator.examples.harbor_test_agent import AGENT_DIR, WrappedAgent


def test_packaged_wrapper_resolves_example_assets() -> None:
    assert WrappedAgent.name() == "hello-harbor-agent"
    assert AGENT_DIR.name == "agent"
    for name in ("agent.py", "main.py", "tracing.py"):
        assert (AGENT_DIR / name).is_file()


def test_notebook_rejects_cleartext_secret_uploads() -> None:
    notebook_path = AGENT_DIR.parent / "harbor_taskset_e2e.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "".join(notebook["cells"][1]["source"])
    function_node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_secret_transport"
    )
    namespace = {"ip_address": ip_address, "urlparse": urlparse}
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), notebook_path, "exec"), namespace)
    validate_secret_transport = namespace["validate_secret_transport"]

    for base_url in (
        "http://localhost:8080",
        "http://dev.localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        "https://platform.example.com",
    ):
        validate_secret_transport(base_url)

    for base_url in ("http://platform.example.com", "ftp://platform.example.com", "not-a-url"):
        with pytest.raises(ValueError):
            validate_secret_transport(base_url)
