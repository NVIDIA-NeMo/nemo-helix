# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build`` -- the entry point of the builder's processes.

``broker`` is the credential broker. It runs as its own Deployment, under its own ServiceAccount,
and is the only process that holds the registry credential and the signing key
(``run/broker.py``). This dispatcher has no way to pick an identity: the one a process runs under
is its Deployment's, or its job step's.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable


def _load(name: str, rest: list[str]) -> Callable[[], int]:
    """Imported lazily, so a process only pays for what it runs."""
    if name == "broker":
        from nemo_builder_plugin.run.broker import main as broker_main

        return lambda: broker_main(rest)
    raise KeyError(name)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print("usage: nhx-build {broker}", file=sys.stderr)
        return 2
    try:
        command = _load(args[0], args[1:])
    except KeyError:
        print(f"unknown command {args[0]!r}; expected broker", file=sys.stderr)
        return 2
    return command()


if __name__ == "__main__":
    raise SystemExit(main())
