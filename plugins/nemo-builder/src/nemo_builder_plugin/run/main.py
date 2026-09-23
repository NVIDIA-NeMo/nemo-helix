# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build`` -- the entry point the three job steps run as.

One binary, three subcommands, three identities. Which one runs is decided by the compiler, and
the identity it runs under is decided by the execution profile the compiler named -- so this
dispatcher deliberately has no way to pick an identity or widen what a step may do.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

COMMANDS: dict[str, Callable[[], int]] = {}


def _load(name: str) -> Callable[[], int]:
    """Imported lazily, so a step only pays for what it runs.

    `supervise` needs the Kubernetes client and `fetch` needs a platform client; importing both
    into every step would make each one depend on libraries it has no business holding.
    """
    if name == "fetch":
        from nemo_builder_plugin.run.fetch import main

        return main
    if name == "supervise":
        from nemo_builder_plugin.run.supervise import main

        return main
    if name == "push":
        from nemo_builder_plugin.run.push import main

        return main
    raise KeyError(name)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print("usage: nhx-build {fetch|supervise|push}", file=sys.stderr)
        return 2
    try:
        command = _load(args[0])
    except KeyError:
        print(f"unknown command {args[0]!r}; expected fetch, supervise or push", file=sys.stderr)
        return 2
    return command()


if __name__ == "__main__":
    raise SystemExit(main())
