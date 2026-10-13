# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write adapter-wheels-v1 environment packages and Gym JSONL datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nhx.rl.schemas.environment import (
    AdapterRef,
    AdapterWheelsV1Manifest,
    EnvironmentManifestMetadata,
    VerifiersAgentInstanceConfig,
)
from nhx.rl.tasks.environment.allowlist import DEFAULT_ADAPTER_AGENT, IMAGE_ADAPTER_ALLOWLIST
from nhx.rl.tasks.environment.validate import validate_package_layout


@dataclass(frozen=True)
class ConvertedPackage:
    environment_root: Path
    dataset_dir: Path
    manifest: AdapterWheelsV1Manifest
    training_jsonl: Path
    validation_jsonl: Path | None = None


def hub_id_to_vf_env_id(hub_id: str) -> str:
    """Map hub slug to verifiers load id (primeintellect/foo -> foo)."""
    if "/" in hub_id:
        return hub_id.split("/", 1)[1]
    return hub_id


def hub_id_to_package_name(hub_id: str) -> str:
    """PyPI package name from hub slug (ascii-tree stays ascii-tree)."""
    return hub_id_to_vf_env_id(hub_id).replace("-", "_").replace(".", "_")


def build_verifiers_agent_yaml(vf_env_id: str, vf_env_args: dict[str, Any]) -> dict[str, Any]:
    cfg = VerifiersAgentInstanceConfig(
        vf_env_id=vf_env_id,
        vf_env_args=vf_env_args,
        description=f"Prime Intellect {vf_env_id} via Gym verifiers_agent",
    )
    inner = cfg.model_dump(exclude_none=True)
    return {
        "verifiers_agent": {
            "responses_api_agents": {
                DEFAULT_ADAPTER_AGENT: inner,
            }
        }
    }


def build_policy_model_yaml() -> dict[str, Any]:
    """Define the ``responses_api_models/policy_model`` server the agent references.

    ``verifiers_agent.model_server`` points at ``responses_api_models/policy_model`` (matching
    NeMo-Gym's own shipped agent configs), but nothing defines that server unless a second
    config supplies it. Without it Gym fails the merged-config check with
    ``ServerRefNotFoundError: ... Available responses_api_models: (none)``.

    Mirrors NeMo-Gym's ``responses_api_models/vllm_model/configs/vllm_model_for_training.yaml``.
    The three interpolations resolve against the global config NeMo-RL injects at spin-up
    (``policy_base_url`` / ``policy_api_key`` / ``policy_model_name`` — see
    ``build_sandbox_global_config`` in nemo_rl.environments.sandbox.nemo_gym_actor), so this
    points Gym at the vLLM engine NeMo-RL is already running rather than starting its own.

    Shipped inside the package rather than referenced from the Gym source tree: the sandbox
    mounts the package, so a self-contained config does not depend on the runtime image's
    directory layout.
    """
    return {
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "entrypoint": "app.py",
                    "base_url": "${policy_base_url}",
                    "api_key": "${policy_api_key}",
                    "model": "${policy_model_name}",
                    "return_token_id_information": True,
                    "uses_reasoning_parser": True,
                }
            }
        }
    }


def write_adapter_wheels_package(
    *,
    out_dir: Path,
    hub_id: str,
    vf_env_id: str | None = None,
    vf_env_args: dict[str, Any] | None = None,
    adapter_agent: str = DEFAULT_ADAPTER_AGENT,
    description: str | None = None,
    wheels_src: Path | None = None,
) -> AdapterWheelsV1Manifest:
    """Materialize adapter-wheels-v1 tree under out_dir."""
    vf_env_id = vf_env_id or hub_id_to_vf_env_id(hub_id)
    vf_env_args = vf_env_args or {}
    image_root = IMAGE_ADAPTER_ALLOWLIST.get(adapter_agent)
    if image_root is None:
        raise ValueError(f"Unknown adapter agent: {adapter_agent!r}")

    manifest = AdapterWheelsV1Manifest(
        adapter=AdapterRef(
            agent=adapter_agent,
            image_config_root=image_root,
        ),
        # policy_model first: it defines the responses_api_models server that the agent's
        # model_server field references. Gym merges all config_paths before validating refs,
        # so order is not strictly required, but declaring the server before its consumer
        # keeps the package readable.
        config_paths=["configs/policy_model.yaml", "configs/verifiers_agent.yaml"],
        metadata=EnvironmentManifestMetadata(
            name=vf_env_id.replace("_", "-"),
            description=description or f"Prime Intellect {hub_id} via Gym {adapter_agent}",
            hub_id=hub_id,
            vf_env_id=vf_env_id,
            adapter_agent=adapter_agent,
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True)
    wheels_dir = out_dir / "wheels"
    wheels_dir.mkdir(exist_ok=True)

    if wheels_src is not None:
        # Clear first. Copying overwrites by FILENAME, so a wheel left from an earlier run
        # of the same out_dir survives whenever the new closure resolved that project to a
        # different version -- and the package then vendors both. Nothing rejects that
        # locally, so it surfaces on the cluster as `you require uvicorn==0.52.1 and
        # uvicorn==0.52.3, we can conclude that your requirements are unsatisfiable`.
        for stale in wheels_dir.glob("*.whl"):
            stale.unlink()
        for whl in wheels_src.glob("*.whl"):
            (wheels_dir / whl.name).write_bytes(whl.read_bytes())

    agent_yaml = build_verifiers_agent_yaml(vf_env_id, vf_env_args)
    (out_dir / "configs" / "verifiers_agent.yaml").write_text(
        yaml.safe_dump(agent_yaml, sort_keys=False),
        encoding="utf-8",
    )
    (out_dir / "configs" / "policy_model.yaml").write_text(
        yaml.safe_dump(build_policy_model_yaml(), sort_keys=False),
        encoding="utf-8",
    )
    (out_dir / "nemo-environment.yaml").write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    validate_package_layout(out_dir, manifest)
    return manifest


def write_dataset_jsonl(
    *,
    dataset_dir: Path,
    rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]] | None = None,
    train_name: str = "training.jsonl",
    val_name: str = "validation.jsonl",
) -> tuple[Path, Path | None]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / train_name
    with train_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    val_path: Path | None = None
    if validation_rows:
        val_path = dataset_dir / val_name
        with val_path.open("w", encoding="utf-8") as f:
            for row in validation_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_path, val_path


def dataset_row_from_verifiers(
    *,
    idx: int,
    prompt: list[dict],
    vf_env_id: str,
    example_id: int | str,
    answer: str = "",
    task: str = "",
    info: dict | None = None,
    adapter_agent: str = DEFAULT_ADAPTER_AGENT,
) -> dict[str, Any]:
    question = ""
    if prompt and isinstance(prompt[-1], dict):
        question = str(prompt[-1].get("content") or "")
    return {
        "task_idx": idx,
        "vf_env_id": vf_env_id,
        "responses_create_params": {"input": prompt},
        "agent_ref": {"type": "responses_api_agents", "name": adapter_agent},
        "question": question,
        "answer": answer,
        "task": task or vf_env_id,
        "example_id": example_id,
        "info": info or {},
    }
