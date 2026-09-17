# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: convert Prime Intellect hub envs to adapter-wheels-v1 (host with internet)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from nmp.rl.tasks.environment.convert import ConvertEnvironmentSpec, convert_prime_environment
from nmp.rl.tasks.environment.validate import (
    EnvironmentPackageValidationError,
    load_manifest,
    validate_package_layout,
)

logger = logging.getLogger(__name__)


def _validation_fraction(raw: str) -> float:
    """Fraction in [0, 1). At 1.0 (or above) the split would leave no training rows."""
    value = float(raw)
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1), got {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Prime Intellect hub environment to adapter-wheels-v1 + Gym JSONL. "
            "Run on a machine with internet; training clusters consume the output offline. "
            "Invoke via: uv run --package nmp-rl pi-to-gym-conversion ..."
        ),
    )
    # Not required: --validate-only checks a package that already exists on disk and never
    # touches the hub, so demanding a hub slug there would be a lie. main() enforces both
    # for the conversion path instead.
    parser.add_argument("--hub-id", help="Hub slug, e.g. primeintellect/ascii-tree")
    parser.add_argument(
        "--hub-version",
        default=None,
        help=(
            "Exact hub package version to vendor, e.g. 0.1.5. Unset resolves to whatever the "
            "index offers, which can change under you: a new release that narrows "
            "Requires-Python will fail the download, and one that still installs may not run "
            "on the training image's Python. Pin this to make a conversion reproducible."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for environment package (nemo-environment.yaml + wheels + configs)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Output directory for training.jsonl (default: sibling of out-dir)",
    )
    parser.add_argument("--vf-env-id", default=None, help="Override verifiers load id")
    parser.add_argument("--vf-env-args", default="{}", help="JSON object of vf_env_args")
    parser.add_argument("--dataset-size", type=int, default=-1, help="Max dataset rows (-1 = all)")
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument(
        "--validation-fraction",
        type=_validation_fraction,
        default=0.0,
        help="Fraction of rows to write to validation.jsonl, in [0, 1) (0 = train only)",
    )
    parser.add_argument(
        "--wheels-dir",
        type=Path,
        default=None,
        help=(
            "Use pre-vendored *.whl files instead of pip download "
            "(must be non-empty; adapter-wheels-v1 always requires wheels)"
        ),
    )
    parser.add_argument(
        "--nemo-rl-root",
        type=Path,
        default=None,
        help=(
            "NeMo-RL checkout, submodules included, at the commit NEMO_RL_REF pins in "
            "docker-bake.hcl. The whole agent closure is derived from it: nemo-gym from the "
            "Gym submodule, ray and openai from its uv.lock. Without it the package needs "
            "sandbox egress to start."
        ),
    )
    parser.add_argument(
        "--gym-root",
        type=Path,
        default=None,
        help=(
            "NeMo-Gym checkout, if not using --nemo-rl-root. nemo-gym is not on an index, so "
            "its dependencies are read by building it from source."
        ),
    )
    parser.add_argument(
        "--nemo-gym-version",
        default=None,
        help="Fail unless the Gym checkout builds this exact version. Only needed when the "
        "checkout is not the one NEMO_RL_REF pins.",
    )
    parser.add_argument(
        "--ray-version",
        default=None,
        help="Override the ray pin from --nemo-rl-root's uv.lock. Gym appends "
        "'ray[default]==<this>' to every per-server venv install.",
    )
    parser.add_argument(
        "--openai-version",
        default=None,
        help="Override the openai pin from --nemo-rl-root's uv.lock. Appended the same way.",
    )
    parser.add_argument(
        "--validate-only",
        type=Path,
        default=None,
        metavar="ENV_ROOT",
        help="Validate an existing environment package and exit",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload env + dataset FileSets to the platform Files API (requires NMP_BASE_URL)",
    )
    parser.add_argument("--workspace", default="default", help="Workspace for --upload")
    parser.add_argument(
        "--environment-name",
        default=None,
        help="Environment FileSet name for --upload (default: derived from hub id)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset FileSet name for --upload (default: <environment-name>-dataset)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.validate_only is not None:
        root = args.validate_only.resolve()
        try:
            manifest = load_manifest(root)
            validate_package_layout(root, manifest)
        except EnvironmentPackageValidationError as exc:
            logger.error("%s", exc)
            return 1
        print(json.dumps({"valid": True, "format": manifest.format, "name": manifest.metadata.name}))
        return 0

    missing = [name for name, value in (("--hub-id", args.hub_id), ("--out-dir", args.out_dir)) if value is None]
    if missing:
        parser.error(f"the following arguments are required for conversion: {', '.join(missing)}")

    try:
        vf_env_args = json.loads(args.vf_env_args)
    except json.JSONDecodeError as exc:
        parser.error(f"--vf-env-args must be valid JSON: {exc}")
    if not isinstance(vf_env_args, dict):
        parser.error(f"--vf-env-args must be a JSON object, got {type(vf_env_args).__name__}")
    spec = ConvertEnvironmentSpec(
        hub_id=args.hub_id,
        hub_version=args.hub_version,
        out_dir=args.out_dir.resolve(),
        dataset_dir=args.dataset_dir.resolve() if args.dataset_dir else None,
        vf_env_id=args.vf_env_id,
        vf_env_args=vf_env_args,
        dataset_size=args.dataset_size,
        dataset_seed=args.dataset_seed,
        validation_fraction=args.validation_fraction,
        wheels_dir=args.wheels_dir.resolve() if args.wheels_dir else None,
        nemo_rl_root=args.nemo_rl_root.resolve() if args.nemo_rl_root else None,
        gym_root=args.gym_root.resolve() if args.gym_root else None,
        nemo_gym_version=args.nemo_gym_version,
        ray_version=args.ray_version,
        openai_version=args.openai_version,
    )
    try:
        result = convert_prime_environment(spec)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    payload: dict = {
        "environment_root": str(result.environment_root),
        "dataset_dir": str(result.dataset_dir),
        "format": result.manifest.format,
        "training_jsonl": str(result.training_jsonl),
        "validation_jsonl": str(result.validation_jsonl) if result.validation_jsonl else None,
    }

    if args.upload:
        from nmp.rl.tasks.environment.upload import upload_converted_packages

        base_url = os.environ.get("NMP_BASE_URL") or os.environ.get("NMP_FILES_URL")
        if not base_url:
            logger.error("--upload requires NMP_BASE_URL or NMP_FILES_URL")
            return 2
        slug = args.hub_id.split("/")[-1].replace("_", "-")
        env_name = args.environment_name or f"{slug}-env"
        ds_name = args.dataset_name or f"{env_name}-dataset"
        try:
            refs = upload_converted_packages(
                environment_root=result.environment_root,
                dataset_dir=result.dataset_dir,
                workspace=args.workspace,
                environment_name=env_name,
                dataset_name=ds_name,
                base_url=base_url,
                api_key=os.environ.get("NMP_API_KEY"),
            )
        except Exception as exc:
            logger.error("Upload failed: %s", exc)
            return 1
        payload["uploaded"] = {"environment": refs.environment, "dataset": refs.dataset}

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
