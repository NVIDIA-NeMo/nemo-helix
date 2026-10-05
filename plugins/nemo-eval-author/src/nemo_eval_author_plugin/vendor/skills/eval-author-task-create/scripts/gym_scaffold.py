#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a Gym-native draft using Gym's own scaffolder in its runtime environment."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import shutil
from pathlib import Path


def scaffold(output: Path, name: str, instruction: Path, description: str, author: str) -> dict:
    # Optional provider imports belong to the selected Gym interpreter, not Eval Author's env.
    native = importlib.import_module("nemo_gym.environment.scaffold")
    manifest_api = importlib.import_module("nemo_gym.environment.manifest")
    version = importlib.metadata.version("nemo-gym")
    text = instruction.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("instruction is empty")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = native.scaffold_environment(kind="environment", name=name, root=output)
        manifest_path = result.asset_dir / "manifest.yaml"
        manifest = manifest_api.load_manifest(manifest_path)
        manifest.description = description
        manifest.authors = [author]
        manifest_path.write_text(manifest_api.dump_manifest(manifest))
        dataset = result.asset_dir / "data/example.jsonl"
        # The native example verifier is only a template; never label this a working eval.
        dataset.write_text(
            json.dumps(
                {
                    "responses_create_params": {"input": [{"role": "user", "content": text}]},
                    "expected_answer": "TODO: implement criteria from the approved proposal",
                }
            )
            + "\n"
        )
        (output / "instruction.md").write_text(text)
        receipt = {
            "schema": "nemo.eval_author.gym_task_draft.v1",
            "provider": "gym",
            "gym_version": version,
            "draft": str(output),
            "environment": name,
            "manifest": str(manifest_path),
            "scaffolder": "nemo_gym.environment.scaffold.scaffold_environment",
            "written": True,
            "runnable": False,
            "validation": "not_run",
            "remaining": [
                "implement scenario tools/state and verifier",
                "replace example verifier fixtures",
                "validate manifest and datasets",
                "exercise positive and negative controls",
                "run the real agent repeatedly and measure retained traces",
            ],
        }
        (output / "draft.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--author", required=True)
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = scaffold(
                args.out.resolve(), args.name, args.instruction_file.resolve(), args.description, args.author
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "written": False,
                    "error_type": type(exc).__name__,
                    "error": "Gym scaffolding failed; check runtime, input, and fresh destination",
                }
            )
        )
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
