# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared agentic-base image build helper.

Both the Harbor runner (Claude Code AUT) and the NAT runner (NeMo Agent
Toolkit AUT) inherit from the same ``nhx-agentic-base:latest`` Docker image.
This module owns the image name and the build helper so both runners go
through one path; previously the helper lived on the Harbor runner and the
NAT runner silently depended on Harbor having been run first to populate
the image.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console

console = Console()

AGENTIC_BASE_IMAGE = "nhx-agentic-base:latest"


async def build_agentic_base_image(
    project_root: Path,
    max_retries: int = 2,
    dockerfile_name: str = "Dockerfile.agentic-base",
) -> None:
    """Build the shared ``nhx-agentic-base:latest`` image used by both Harbor and NAT runners.

    Retries transient build failures up to ``max_retries`` times. Raises
    ``RuntimeError`` if the build fails on every attempt.
    """
    for attempt in range(max_retries + 1):
        if attempt > 0:
            console.print(f"[yellow]Retrying Docker build (attempt {attempt + 1}/{max_retries + 1})...[/yellow]")
        else:
            console.print("[bold]Building Docker image...[/bold]")
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "--network=host",
            "-f",
            str(project_root / dockerfile_name),
            "-t",
            AGENTIC_BASE_IMAGE,
            str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            console.print("[green]Docker image built successfully.[/green]")
            return
        console.print(f"[red]Docker build failed (exit={proc.returncode})[/red]")
        if stderr:
            console.print(stderr.decode()[-1000:])
    raise RuntimeError("Docker build failed after retries")
