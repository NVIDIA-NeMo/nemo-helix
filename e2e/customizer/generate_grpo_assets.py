# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the GRPO assets for the customizer E2E tests.

Writes these formats under ``<generation.output_dir>/``:

- ``grpo_math_env/``: Gym's ``math_with_judge`` server packaged as a ``wheels-v1`` environment.
  Its config sets ``should_use_judge: false``, so the reward is ``math-verify`` alone.
- ``grpo_math_env_native/``: the same server as a ``native-v1`` tree, installed from an index.
- ``grpo_math_env_native_ref/``: ``native-v1`` reference-only, resolved to the image's built-in.
- ``grpo_ascii_tree_env/`` and ``grpo_ascii_tree_smoke/``: Prime Intellect ``ascii-tree``
  converted by ``pi-to-gym-conversion`` into an ``adapter-wheels-v1`` package and its dataset.
- ``grpo_math_smoke/``: DAPO-Math-17k Gym rows, a few train and holdout prompts.
- ``grpo_math_uplift/``: the same, with enough prompts for a measurable reward increase.

The wheels formats vendor a closure pinned to the ``nemo-gym``, ``ray``, and ``openai`` that
the training image runs. All three come from NeMo-RL at the ``NEMO_RL_REF`` that
``docker-bake.hcl`` pins: ``nemo-gym`` from its Gym submodule, ``ray`` and ``openai`` from its
``uv.lock``.

    PYTHONPATH=. uv run --frozen python e2e/customizer/generate_grpo_assets.py

Needs internet: it clones NeMo-RL at the pin, resolves the wheel closures, fetches the hub
environment, and downloads the DAPO-Math datasets from Hugging Face. ``--rl-dir`` reuses a
NeMo-RL checkout already at the pin.

The environments are not published to S3: CI builds them with ``--env-only`` from the checkout
it tests, so a pin bump cannot leave them stale. The DAPO-Math datasets are published.
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from e2e.customizer.assets_manifest import generation_config, repo_root

logger = logging.getLogger(__name__)

ENV_FORMAT = "grpo_math_env"
NATIVE_ENV_FORMAT = "grpo_math_env_native"
NATIVE_REF_ENV_FORMAT = "grpo_math_env_native_ref"
ADAPTER_ENV_FORMAT = "grpo_ascii_tree_env"
ADAPTER_SMOKE_FORMAT = "grpo_ascii_tree_smoke"
SMOKE_FORMAT = "grpo_math_smoke"
UPLIFT_FORMAT = "grpo_math_uplift"
GYM_SERVER = "resources_servers/math_with_judge"
GYM_SUBMODULE = "3rdparty/Gym-workspace/Gym"
ADAPTER_HUB_ID = "primeintellect/ascii-tree"
# Pinned: a later release can narrow Requires-Python past the training image's.
ADAPTER_HUB_VERSION = "0.1.5"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rl-dir", type=Path, default=None, help="NeMo-RL checkout at the pin (default: clone it).")
    parser.add_argument("--arch", choices=("x86_64", "aarch64"), default="x86_64", help="Training node architecture.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <generation.output_dir>.")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the manifest generation seed.")
    parser.add_argument("--smoke-train-size", type=int, default=64)
    parser.add_argument("--smoke-validation-size", type=int, default=16)
    parser.add_argument("--uplift-train-size", type=int, default=2048)
    parser.add_argument("--uplift-validation-size", type=int, default=200)
    parser.add_argument("--adapter-dataset-size", type=int, default=32)
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Build only the environment packages (and the adapter's dataset); skip the DAPO-Math datasets.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    cfg = generation_config()
    seed = cfg["seed"] if args.seed is None else args.seed
    output_root = (args.output_dir or cfg["output_dir"]).resolve()
    platform_dir = repo_root()

    with tempfile.TemporaryDirectory(prefix="nemo-rl-") as tmp:
        rl_dir = args.rl_dir.resolve() if args.rl_dir else _clone_rl(platform_dir, Path(tmp) / "RL")
        gym_root = rl_dir / GYM_SUBMODULE
        # The packagers pip-install into the interpreter that runs them, so they get their own
        # environment instead of the platform venv the tests run in. It carries the conversion
        # extra (verifiers) as well.
        tools = Path(tmp) / "conversion-venv" / "bin"
        _run(["uv", "sync", "--frozen", "--package", "nhx-rl", "--extra", "conversion"], cwd=platform_dir,
             env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(tools.parent)})  # fmt: skip
        env_dirs = [_fresh_dir(output_root / name) for name in (ENV_FORMAT, NATIVE_ENV_FORMAT, NATIVE_REF_ENV_FORMAT)]
        package = [str(tools / "python"), "scripts/grpo-examples/gym_to_env_package.py", "--gym-root", str(gym_root),
                   "--server", GYM_SERVER]  # fmt: skip
        # The packager reads the image's nemo-gym, ray, and openai pins from this checkout.
        _run([*package, "--format", "wheels-v1", "--arch", args.arch, "--nemo-rl-root", str(rl_dir),
              "--out-dir", str(env_dirs[0])], cwd=platform_dir)  # fmt: skip
        _run([*package, "--format", "native-v1", "--out-dir", str(env_dirs[1])], cwd=platform_dir)
        _run([*package, "--format", "native-v1", "--reference-only", "--out-dir", str(env_dirs[2])], cwd=platform_dir)

        adapter_dir = _fresh_dir(output_root / ADAPTER_ENV_FORMAT)
        adapter_data_dir = _fresh_dir(output_root / ADAPTER_SMOKE_FORMAT)
        adapter_dir.mkdir()
        adapter_data_dir.mkdir()
        _run(
            [
                str(tools / "pi-to-gym-conversion"),
                "--hub-id", ADAPTER_HUB_ID,
                "--hub-version", ADAPTER_HUB_VERSION,
                "--out-dir", str(adapter_dir),
                "--dataset-dir", str(adapter_data_dir),
                "--dataset-size", str(args.adapter_dataset_size),
                "--validation-fraction", "0.1",
                "--nemo-rl-root", str(rl_dir),
                "--gym-root", str(gym_root),
            ],
            cwd=platform_dir,
        )  # fmt: skip

        for env_dir in (*env_dirs, adapter_dir):
            _run([str(tools / "pi-to-gym-conversion"), "--validate-only", str(env_dir)], cwd=platform_dir)
    if args.env_only:
        logger.info("Wrote the environment packages under %s", output_root)
        return 0

    for format_name, train_size, validation_size in (
        (SMOKE_FORMAT, args.smoke_train_size, args.smoke_validation_size),
        (UPLIFT_FORMAT, args.uplift_train_size, args.uplift_validation_size),
    ):
        _run(
            [
                "uv", "run", "--with", "datasets", "scripts/grpo-examples/prepare_math_with_judge.py",
                "--out-dir", str(_fresh_dir(output_root / format_name)),
                "--train-size", str(train_size),
                "--validation-size", str(validation_size),
                "--seed", str(seed),
            ],
            cwd=platform_dir,
        )  # fmt: skip

    logger.info("Wrote the environment packages and datasets under %s", output_root)
    return 0


def _clone_rl(platform_dir: Path, dest: Path) -> Path:
    """Shallow-clone NeMo-RL at the ``docker-bake.hcl`` pin, with only the Gym submodule."""
    bake = (platform_dir / "docker-bake.hcl").read_text(encoding="utf-8")
    repo, ref = _bake_default(bake, "NEMO_RL_REPO"), _bake_default(bake, "NEMO_RL_REF")
    logger.info("Cloning %s at %s", repo, ref)
    dest.mkdir(parents=True)
    _run(["git", "init", "--quiet"], cwd=dest)
    _run(["git", "remote", "add", "origin", repo], cwd=dest)
    _run(["git", "fetch", "--quiet", "--depth", "1", "origin", ref], cwd=dest)
    _run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=dest)
    _run(["git", "submodule", "update", "--init", "--quiet", GYM_SUBMODULE], cwd=dest)
    return dest


def _bake_default(bake: str, variable: str) -> str:
    match = re.search(rf'variable\s+"{variable}"\s*\{{\s*default\s*=\s*"([^"]+)"', bake)
    if match is None:
        raise SystemExit(f"{variable} has no default in docker-bake.hcl")
    return match.group(1)


def _fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    logger.info("$ %s", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
