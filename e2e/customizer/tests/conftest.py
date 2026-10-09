# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for customizer GPU e2e tests (S3 asset staging + uplift eval)."""

import logging
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import stage_assets

logger = logging.getLogger(__name__)

REQUIRE_UPLIFT = os.environ.get("E2E_REQUIRE_UPLIFT") == "1"
GPU_TEST_TIMEOUT = int(os.environ.get("E2E_GPU_TEST_TIMEOUT", "5400"))


@dataclass(frozen=True)
class CustomizerAssetCache:
    chat_format: Path
    dpo: Path
    qwen3_model: Path
    unsloth_model: Path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply a generous default timeout to customizer uplift tests."""
    for item in items:
        if "/e2e/customizer/tests/" not in str(item.fspath):
            continue
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(GPU_TEST_TIMEOUT))


@pytest.fixture(scope="session")
def require_uplift() -> bool:
    return REQUIRE_UPLIFT


@pytest.fixture(scope="module")
def platform_base_url(client: NemoClient) -> str:
    return os.environ.get("NHX_BASE_URL") or client.base_url


@pytest.fixture(scope="module")
def customizer_workspace(client: NemoClient) -> Iterator[str]:
    workspaces = WorkspacesClient.from_client(client)
    name = f"e2e-cust-{uuid.uuid4().hex[:8]}"
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name))
    yield name
    try:
        workspaces.delete_workspace(name=name)
    except Exception:
        logger.warning("Failed to delete customizer workspace %s (leaked)", name, exc_info=True)


@pytest.fixture(scope="session")
def customizer_asset_cache() -> CustomizerAssetCache:
    return CustomizerAssetCache(
        chat_format=stage_assets.sync_dataset_format("chat_format"),
        dpo=stage_assets.sync_dataset_format("dpo"),
        qwen3_model=stage_assets.sync_model("qwen3-0.6b"),
        unsloth_model=stage_assets.sync_model("qwen2.5-0.5b-instruct-unsloth"),
    )


@pytest.fixture(scope="session")
def chat_format_cache(customizer_asset_cache: CustomizerAssetCache) -> Path:
    return customizer_asset_cache.chat_format


@pytest.fixture(scope="session")
def dpo_cache(customizer_asset_cache: CustomizerAssetCache) -> Path:
    return customizer_asset_cache.dpo


@pytest.fixture(scope="session")
def qwen3_model_cache(customizer_asset_cache: CustomizerAssetCache) -> Path:
    return customizer_asset_cache.qwen3_model


@pytest.fixture(scope="session")
def unsloth_model_cache(customizer_asset_cache: CustomizerAssetCache) -> Path:
    return customizer_asset_cache.unsloth_model


@pytest.fixture(scope="module")
def squad_fileset(client: NemoClient, customizer_workspace: str, chat_format_cache: Path) -> str:
    """SQuAD CHAT dataset fileset staged from S3 (train + validation in one fileset)."""
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=chat_format_cache,
        name_prefix="squad",
        files={"train.jsonl": "training.jsonl", "validation.jsonl": "validation.jsonl"},
    )


@pytest.fixture(scope="module")
def squad_val_rows(chat_format_cache: Path) -> list[dict]:
    return ceval.load_chat_jsonl(chat_format_cache / "validation.jsonl")


@pytest.fixture(scope="module")
def unsloth_train_fileset(client: NemoClient, customizer_workspace: str, chat_format_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=chat_format_cache,
        name_prefix="squad-train",
        files={"train.jsonl": "training.jsonl"},
    )


@pytest.fixture(scope="module")
def unsloth_val_fileset(client: NemoClient, customizer_workspace: str, chat_format_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=chat_format_cache,
        name_prefix="squad-val",
        files={"validation.jsonl": "validation.jsonl"},
    )


@pytest.fixture(scope="module")
def helpsteer_dpo_fileset(client: NemoClient, customizer_workspace: str, dpo_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=dpo_cache,
        name_prefix="helpsteer3-dpo",
        files={"training.jsonl": "training.jsonl", "validation.jsonl": "validation.jsonl"},
    )


@pytest.fixture(scope="module")
def dpo_eval_rows(dpo_cache: Path) -> list[dict]:
    path = dpo_cache / "eval.jsonl"
    if not path.is_file():
        pytest.fail(f"DPO eval rows missing after S3 sync: {path}")
    return ceval.load_chat_jsonl(path)


@pytest.fixture(scope="module")
def automodel_base_entity(
    client: NemoClient,
    customizer_workspace: str,
    qwen3_model_cache: Path,
) -> str:
    return stage_assets.stage_model_entity(
        client,
        customizer_workspace,
        entity_name="qwen3-0-6b-automodel",
        snapshot_dir=qwen3_model_cache,
        hf_repo="Qwen/Qwen3-0.6B",
    )


@pytest.fixture(scope="module")
def unsloth_base_entity(
    client: NemoClient,
    customizer_workspace: str,
    unsloth_model_cache: Path,
) -> str:
    return stage_assets.stage_model_entity(
        client,
        customizer_workspace,
        entity_name="qwen25-05b-unsloth",
        snapshot_dir=unsloth_model_cache,
        hf_repo="unsloth/Qwen2.5-0.5B-Instruct",
    )


@pytest.fixture(scope="module")
def rl_base_entity(
    client: NemoClient,
    customizer_workspace: str,
    qwen3_model_cache: Path,
) -> str:
    return stage_assets.stage_model_entity(
        client,
        customizer_workspace,
        entity_name="qwen3-0-6b-rl",
        snapshot_dir=qwen3_model_cache,
        hf_repo="Qwen/Qwen3-0.6B",
    )
