# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check startup imports using the installed wheel's isolated interpreter."""

import importlib
import importlib.metadata
import sys
from pathlib import Path


def main() -> None:
    modules = [importlib.import_module("nhx_sandbox.harbor_opensandbox_services")]
    distribution = importlib.metadata.distribution("nemo-helix")
    evals = next(ep for ep in distribution.entry_points if ep.group == "nemo.services" and ep.name == "evals")
    evals.load()
    modules.append(importlib.import_module(evals.module))

    for module in modules:
        assert module.__file__ is not None, module.__name__
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), module.__file__
    print("built-wheel evals service and sandbox imports OK")


if __name__ == "__main__":
    main()
