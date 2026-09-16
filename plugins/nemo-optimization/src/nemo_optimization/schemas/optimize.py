# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path helper shared by the optimize bundle preflight.

The optimize study spec itself lives in
``nemo_agent_optimization_plugin.schemas.optimize`` (``AgentOptimizeSpec`` /
``OptimizeSpec`` / ``OptimizeSubmitSpec``). This module used to carry its own,
now-superseded copy of that spec; it was dropped, leaving only the one helper
that :mod:`nemo_optimization.bundle` still imports.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath


def is_fileset_relative(config_path: str) -> bool:
    """True when *config_path* stays inside a fileset root once joined to it.

    Checked in both host flavours — the submitting client may be on Windows while
    the task host is Linux, so a POSIX-only check would let ``C:\\bundle\\optimize.yaml``
    through, and a POSIX-only ``..`` scan would miss ``..\\escape.yaml``.  ``~`` is rejected
    too: it is not absolute to ``PurePath``, but it expands to a client home directory
    that does not exist on the task host.  A bare drive letter (``D:optimize.yml``) is
    rejected too: ``PureWindowsPath`` treats it as drive-relative rather than absolute, but
    it is still anchored to a drive's current directory on the client host, not the fileset
    root.
    """
    if config_path.startswith("~"):
        return False
    flavours = (PurePosixPath(config_path), PureWindowsPath(config_path))
    return not any(path.is_absolute() or ".." in path.parts or path.drive for path in flavours)
