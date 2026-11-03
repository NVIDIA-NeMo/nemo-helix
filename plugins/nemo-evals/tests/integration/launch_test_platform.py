# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a ``nemo services run`` platform with the evaluator integration harness's config, in the foreground.

For suites that test against a platform someone else runs, such as ``test_taskset_run_controller.py``: start one
per auth mode, wait for ``/health/ready``, then point the suite's base-URL variable at it. The config is the one
``conftest.py`` writes for its own platforms (subprocess jobs, IGW mock providers, fast PDP refresh), so these
platforms serve the same profiles and settings the tests assume. Stop it with SIGTERM or Ctrl+C.

    uv run python plugins/nemo-evals/tests/integration/launch_test_platform.py --port 18092 --auth
"""

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path
from types import ModuleType


def _harness() -> ModuleType:
    """The integration ``conftest.py``, loaded by path: it is a pytest plugin, not an importable module."""
    spec = importlib.util.spec_from_file_location("evals_integration_harness", Path(__file__).with_name("conftest.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help="port to serve on, on localhost")
    parser.add_argument("--auth", action="store_true", help="enable auth (the PDP); off by default")
    parser.add_argument("--work-dir", type=Path, help="config, data and job storage (default: a new temp dir)")
    args = parser.parse_args()

    harness = _harness()
    base_url = f"http://localhost:{args.port}"
    work_root: Path = args.work_dir or Path(tempfile.mkdtemp(prefix="evals-test-platform-"))
    work_root.mkdir(parents=True, exist_ok=True)
    config_path = harness._materialize_subprocess_config(work_root, base_url=base_url, auth_enabled=args.auth)
    env = {
        **os.environ,
        "NHX_BASE_URL": base_url,
        "NHX_CONFIG_FILE_PATH": str(config_path),
        harness.DATA_DIR_ENVVAR: str(work_root / "data"),
        harness.MOCK_PROVIDER_PREFIX_ENVVAR: harness.MOCK_PROVIDER_PREFIX,
    }
    command = [
        "uv",
        "run",
        "nemo",
        "services",
        "run",
        "--service-group",
        "all",
        "--controllers",
        ",".join(harness.REQUIRED_CONTROLLERS),
        "--port",
        str(args.port),
    ]
    print(f"Serving {'auth-on' if args.auth else 'auth-off'} platform at {base_url} from {work_root}", flush=True)
    os.chdir(harness.REPO_ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
