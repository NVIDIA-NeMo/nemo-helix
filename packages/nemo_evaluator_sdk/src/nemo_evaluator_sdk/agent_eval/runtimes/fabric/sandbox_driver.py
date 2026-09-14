# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Fabric run, executed inside the sandbox and printed as JSON on stdout.

``FabricContainerRuntime`` seeds this file into the sandbox and runs it as a script: it reads this
module's source rather than importing it, so the sandbox's Fabric is the one that has to satisfy
these imports.

Fabric's own CLI is not an option. It became a presets/examples experimentation tool in Fabric 0.2
(binary ``nemo-fabric``, selectors ``--preset``/``--example``), so there is no config-driven CLI
entrypoint to exec; the Python API is the supported way to run an arbitrary agent config.

Errors are not caught: a traceback on a non-zero exit is what the runtime already grades a failed
trial from, and swallowing it here would produce a well-formed result payload for a run that did
not happen.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]


def main(argv: list[str]) -> int:
    """Run ``argv[1]`` (agent config) over ``argv[2]`` (input text) and print the ``RunResult``."""
    config_path, input_path = Path(argv[1]), Path(argv[2])
    config = FabricConfig.from_mapping(json.loads(config_path.read_text(encoding="utf-8")))
    text = input_path.read_text(encoding="utf-8")
    # Relative paths in the config resolve against the config's own directory, as they would for a
    # caller that loaded it from there.
    result = asyncio.run(Fabric().run(config, input=text, base_dir=str(config_path.parent)))
    print(json.dumps(result.to_mapping(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
