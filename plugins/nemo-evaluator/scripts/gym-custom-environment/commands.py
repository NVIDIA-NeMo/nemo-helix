# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal subprocess wrapper shared by prerequisite and workflow modules.

The wrapper executes argument vectors without a shell, merges only explicit
environment overrides, converts command failures into readable exceptions, and
optionally records stdout as evidence. Developers do not invoke this module.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


class CommandError(RuntimeError):
    """Raised when an external command exits unsuccessfully."""


class CommandRunner:
    """Run the few external tools needed during local preparation.

    Platform resources are managed through the Python SDK; this wrapper is
    limited to workstation tools such as ``uv`` and ``kubectl``.
    """

    def __init__(self, *, working_directory: Path) -> None:
        """Configure the working directory shared by child processes."""
        self.working_directory = working_directory

    def run(
        self,
        arguments: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        input_text: str | None = None,
        output_path: Path | None = None,
    ) -> str:
        """Run to completion and optionally preserve stdout as workflow evidence."""
        try:
            result = subprocess.run(
                list(arguments),
                cwd=self.working_directory,
                env={**os.environ, **(environment or {})},
                input=input_text,
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or f"exit code {error.returncode}"
            command = " ".join(arguments)
            raise CommandError(f"{command} failed: {detail}") from error

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.stdout, encoding="utf-8")
        return result.stdout

    def run_json(
        self,
        arguments: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        input_text: str | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Run a command and require its stdout to contain a JSON object."""
        output = self.run(
            arguments,
            environment=environment,
            input_text=input_text,
            output_path=output_path,
        )
        value = json.loads(output)
        if not isinstance(value, dict):
            command = " ".join(arguments)
            raise CommandError(f"{command} did not return a JSON object")
        return value

    def start(
        self,
        arguments: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        stdout: TextIO | int | None = None,
    ) -> subprocess.Popen[str]:
        """Start a background helper, such as a temporary ``kubectl`` port-forward."""
        return subprocess.Popen(
            list(arguments),
            cwd=self.working_directory,
            env={**os.environ, **(environment or {})},
            text=True,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
