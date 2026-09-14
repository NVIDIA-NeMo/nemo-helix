# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for capturing OTLP inside the Fabric sandbox.

The receiver itself is covered by the shared suite; what is specific here is running it the way
the sandbox does -- as a seeded script in a separate process -- and the command that drives it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import time
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.runtimes.fabric import container_runtime, otlp_receiver
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_receiver import EXPORT_SUFFIX, READY_FILENAME
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import fold_exports, otlp_trace_path
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask

from packages.nemo_evaluator_sdk.tests.agent_eval._otlp_testkit import export, post, span_names

RECEIVER = Path(otlp_receiver.__file__)
_TASK = AgentEvalTask(id="t", intent="i", inputs={"instruction": "do it"})


def _runtime() -> container_runtime.FabricContainerRuntime:
    return container_runtime.FabricContainerRuntime(
        config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}},
        provider=object(),
        image="img",
    )


def test_the_receiver_runs_as_a_seeded_script_and_its_exports_fold(tmp_path: Path, unused_tcp_port) -> None:
    # Run the way the sandbox runs it: a separate interpreter, no import of this package.
    traces = tmp_path / "traces"
    process = subprocess.Popen([sys.executable, str(RECEIVER), str(traces), str(unused_tcp_port)])
    try:
        deadline = time.monotonic() + 15
        while not (traces / READY_FILENAME).is_file():
            assert time.monotonic() < deadline and process.poll() is None, "receiver never signalled ready"
            time.sleep(0.05)

        assert post(f"http://127.0.0.1:{unused_tcp_port}/v1/traces", export("codex-turn")) == 200
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert fold_exports(traces) == 1
    assert span_names(otlp_trace_path(tmp_path)) == ["codex-turn"]


def test_the_seeded_receiver_needs_nothing_the_sandbox_image_lacks() -> None:
    # The image ships Fabric and its harnesses, not this package or its dependencies, so a single
    # non-stdlib import would make the receiver unrunnable there.
    seeded = _runtime()._seed_files(_TASK, None)[0][container_runtime._RECEIVER_PATH]
    assert seeded == RECEIVER.read_text(encoding="utf-8")

    imported = set()
    for node in ast.walk(ast.parse(seeded)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


def test_the_sandbox_command_waits_for_the_receiver_and_keeps_fabric_s_exit_status() -> None:
    # Fabric must not start before the receiver can be reached, and stopping the receiver must not
    # mask a failed run.
    command = _runtime()._fabric_command()

    assert READY_FILENAME in command
    # Anchored on the seeded paths, so renaming either one cannot leave the ordering unchecked.
    assert command.index(container_runtime._RECEIVER_PATH) < command.index(container_runtime._DRIVER_PATH)
    assert "STATUS=$?" in command
    assert command.rstrip().endswith("exit $STATUS")


def test_the_receiver_writes_where_the_sandbox_download_will_find_it() -> None:
    # /out is what the runtime downloads; a capture written anywhere else never leaves the sandbox.
    command = _runtime()._fabric_command()

    assert container_runtime._TRACES_DIR.startswith(f"{container_runtime._OUT_DIR}/")
    assert container_runtime._TRACES_DIR in command


def test_a_stored_export_that_will_not_decode_costs_the_flush_not_the_trial(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / f"000001{EXPORT_SUFFIX}").write_bytes(b"\xff\xfe not protobuf")
    (traces / f"000002{EXPORT_SUFFIX}").write_bytes(export("survivor"))

    assert fold_exports(traces) == 1
    assert span_names(otlp_trace_path(tmp_path)) == ["survivor"]
