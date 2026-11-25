# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent CLI an OpenSandbox agent bundle installs into the Gym host.

The host provider names the installed executable in ``AGENT_EXECUTABLE_ENV_KEY``. The Gym host runtime
checks it with ``--version`` and puts its directory first on ``PATH`` only after the container-engine setup
has dropped privileges: on the Docker-enabled image the host starts as root with ``SYS_ADMIN``, and the
check would otherwise run the agent CLI with both, in the job's writable work directory.
"""

import os
import subprocess
from collections.abc import Callable, MutableMapping

AGENT_EXECUTABLE_ENV_KEY = "NHX_GYM_AGENT_EXECUTABLE"
#: How long ``--version`` may take; a CLI that hangs fails the host rather than holding it until the
#: caller's readiness timeout.
VERSION_TIMEOUT_S = 120


def activate_agent_executable(
    executable: str | None,
    environ: MutableMapping[str, str] = os.environ,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Fail unless ``executable --version`` succeeds, then put its directory first on ``PATH``.

    A no-op without an executable. Raises on a missing, failing or hanging agent so the host reports a
    bootstrap failure instead of letting Gym install a different CLI at its first rollout.
    """
    if not executable:
        return
    path = environ.get("PATH", "")
    agent_path = f"{os.path.dirname(executable)}{os.pathsep}{path}" if path else os.path.dirname(executable)
    try:
        result = run(
            [executable, "--version"],
            # With its directory already first on PATH, as the launcher this replaces ran it: a bundle's CLI can
            # need a sibling, such as a `#!/usr/bin/env node` script shipped beside its own `node`.
            env={**environ, "PATH": agent_path},
            stdin=subprocess.DEVNULL,
            # To the host's stderr, as the shell launcher this replaces did.
            stdout=2,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"the agent bundle's executable {executable} could not run: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"the agent bundle's executable {executable} --version exited {result.returncode}")
    environ["PATH"] = agent_path
