# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build``: the entry point of the builder's processes."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable


def _load(name: str, rest: list[str]) -> Callable[[], int]:
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
        command = _load(args[0], args[1:])
    except KeyError:
        print(f"unknown command {args[0]!r}; expected fetch, supervise or push", file=sys.stderr)
        return 2
    return command()


if __name__ == "__main__":
    raise SystemExit(main())
