# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Child entrypoint for SDK-started local services daemons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nemo_helix_ext.local.services import ServiceRunConfig, run_services


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m nemo_helix_ext.local._service_child <run-request.json>\n")
        return 2
    request_path = Path(args[0])
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    finally:
        request_path.unlink(missing_ok=True)
    run_services(ServiceRunConfig(**payload), _mode="daemon")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
