# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the pinned Hermes source with its upstream dependency overrides."""

import argparse
import runpy
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path


def install_hermes(checkout: Path, python: Path) -> None:
    # Hermes deliberately overrides conflicting transitive requirements (for
    # example browser-harness's Pillow pin). Use the pinned source's own lock
    # and checker rather than maintaining a second set of exceptions in Helix.
    checker = runpy.run_path(str(checkout / "pm" / "pip_check.py"))
    lock = checkout / "uv.lock"
    with tempfile.TemporaryDirectory(prefix="nemo-hermes-install-") as directory:
        overrides = Path(directory) / "overrides.txt"
        overrides.write_text("\n".join(checker["override_requirements"](lock)) + "\n")
        subprocess.run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--prerelease=allow",
                "--python",
                str(python),
                "--overrides",
                str(overrides),
                f"nemo-fabric=={version('nemo-fabric')}",
                # Fabric deliberately gives Hermes its own Relay 0.10 range.
                # The umbrella Fabric relay extra selects 0.11 and conflicts.
                f"nemo-fabric-adapters-hermes[relay]=={version('nemo-fabric')}",
                "nemo-relay==0.10.0",
                "--editable",
                str(checkout),
            ],
            check=True,
        )
    result = subprocess.run(
        ["uv", "--no-config", "pip", "check", "--python", str(python)],
        capture_output=True,
        text=True,
    )
    report = result.stdout + result.stderr
    print(report, end="")
    if result.returncode and checker["unexpected_incompatibilities"](report, lock) != []:
        raise subprocess.CalledProcessError(result.returncode, result.args, output=report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    # Resolving the interpreter symlink would select the base Python instead
    # of the isolated venv.
    install_hermes(args.checkout.resolve(), args.python.absolute())


if __name__ == "__main__":
    main()
