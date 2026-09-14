# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the bundled harness CLIs in this image are actually runnable.

`nemo-agents-plugin` pulls `nemo-fabric[claude,codex]`, whose wheels vendor
native executables: `claude` bundles a Node runtime and `codex` is a Rust
binary. Importing the packages and stat-ing the binaries is not enough -- a
wheel resolved for the wrong architecture, or one whose dynamic loader is
missing from a distroless runtime, is present and executable but dies on
launch. Run each CLI so that failure surfaces at build time instead of in a
task.

`--version` is non-mutating and needs neither network nor credentials.
"""

import pathlib
import subprocess
import sys

import claude_agent_sdk
from codex_cli_bin import bundled_codex_path

TIMEOUT_SECONDS = 120


def bundled_claude_path() -> pathlib.Path:
    """Resolve the `claude` binary vendored inside the claude-agent-sdk wheel.

    This mirrors `SubprocessCLITransport._find_bundled_cli`, which is a private
    instance method that ignores `self`. Recomputing the path here keeps the
    check off a private API; if the SDK ever relocates the binary, this check
    fails loudly, which is what it is for.
    """
    return pathlib.Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"


def main() -> int:
    paths = {
        "claude": bundled_claude_path(),
        "codex": bundled_codex_path(),
    }

    failures: list[str] = []
    for name, path in paths.items():
        if not path:
            failures.append(f"{name}: no bundled binary resolved")
            continue
        try:
            completed = subprocess.run(  # noqa: S603
                [str(path), "--version"],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except OSError as exc:
            failures.append(f"{name}: {path} failed to launch: {exc!r}")
            continue
        except subprocess.TimeoutExpired:
            failures.append(f"{name}: {path} timed out after {TIMEOUT_SECONDS}s")
            continue

        if completed.returncode != 0:
            failures.append(f"{name}: {path} exited {completed.returncode}: {completed.stderr.strip()[:200]!r}")
            continue

        print(f"{name}: {path} -> {completed.stdout.strip()}")

    if failures:
        print("unusable bundled harness CLIs:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
