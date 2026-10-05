# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Mapping
from pathlib import Path


def xdist_worker_xdg_cache_home(environ: Mapping[str, str]) -> str | None:
    """Return an isolated XDG cache path for each pytest-xdist worker.

    garakapi copies its packaged plugin cache into ``$XDG_CACHE_HOME/garak`` at
    import time, then immediately reads the copied JSON file. During xdist
    collection, multiple workers can import garakapi at the same time and race on
    the shared cache file, which can surface as intermittent JSONDecodeError
    during test collection. Isolating XDG_CACHE_HOME per worker keeps that
    test-only behavior out of garakapi's production code.
    """
    worker_id = environ.get("PYTEST_XDIST_WORKER")
    if not worker_id or environ.get("NHX_PYTEST_XDIST_CACHE_HOME_ISOLATED"):
        return None

    base_cache_home = Path(environ.get("XDG_CACHE_HOME") or environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    run_uid = environ.get("PYTEST_XDIST_TESTRUNUID", "default")
    return str(base_cache_home / "pytest-xdist" / run_uid / worker_id)
