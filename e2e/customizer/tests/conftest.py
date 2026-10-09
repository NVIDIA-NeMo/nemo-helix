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
from nemo_helix_plugin.files.types import FilesetPurpose
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer import stage_assets

logger = logging.getLogger(__name__)

REQUIRE_UPLIFT = os.environ.get("E2E_REQUIRE_UPLIFT") == "1"
GPU_TEST_TIMEOUT = int(os.environ.get("E2E_GPU_TEST_TIMEOUT", "5400"))
# Smoke trains a few steps, so it stages only the first rows of each shared JSONL file.
SMOKE_DATASET_ROWS = 64

EMBED_HF_REPO = "nvidia/Nemotron-3-Embed-1B-BF16"
NEMOTRON_LIGHTNING_HF_REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


@dataclass(frozen=True)
class CustomizerAssetCache:
    chat_format: Path
    dpo: Path
    qwen3_model: Path


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
    return os.environ.get("NHX_BASE_URL") or str(client.base_url)


@pytest.fixture(scope="module")
def customizer_workspace(client: NemoClient) -> Iterator[str]:
    name = f"e2e-cust-{uuid.uuid4().hex[:8]}"
    workspaces = WorkspacesClient.from_client(client)
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name))
    yield name
    try:
        workspaces.delete_workspace(name=name)
    except Exception:
        logger.warning("Failed to delete customizer workspace %s (leaked)", name, exc_info=True)


# Session fixtures that sync from S3. Prefetched when the session starts: CI's S3
# credentials expire an hour after they are issued, and a test late in a long session
# would otherwise make its first sync after that.
S3_ASSET_FIXTURES = (
    "customizer_asset_cache",
    "embedding_nvdocs_cache",
    "embedding_nvdocs_smoke_cache",
    "grpo_math_env_cache",
    "grpo_math_env_native_cache",
    "grpo_math_env_native_ref_cache",
    "grpo_ascii_tree_env_cache",
    "grpo_ascii_tree_smoke_cache",
    "grpo_math_smoke_cache",
    "grpo_math_uplift_cache",
    "embed_model_cache",
    "nemotron_lightning_model_cache",
)


@pytest.fixture(scope="session", autouse=True)
def _prefetch_s3_assets(request: pytest.FixtureRequest) -> None:
    for name in sorted(stage_assets.fixtures_to_prefetch(request.session.items, S3_ASSET_FIXTURES)):
        request.getfixturevalue(name)


@pytest.fixture(scope="session")
def customizer_asset_cache() -> CustomizerAssetCache:
    return CustomizerAssetCache(
        chat_format=stage_assets.sync_dataset_format("chat_format"),
        dpo=stage_assets.sync_dataset_format("dpo"),
        qwen3_model=stage_assets.sync_model("qwen3-0.6b"),
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
def embed_model_cache() -> Path:
    return stage_assets.sync_model("nemotron-3-embed-1b-bf16")


@pytest.fixture(scope="session")
def nemotron_lightning_model_cache() -> Path:
    """~66 GB of weights; synced only by the tests that use it."""
    return stage_assets.sync_model("nemotron-3.5-lightning-30b-a3b-bf16")


@pytest.fixture(scope="session")
def embedding_nvdocs_cache() -> Path:
    """Mined NVDocs retrieval data: ``training.jsonl`` plus the frozen ``eval_beir/`` split."""
    return stage_assets.sync_dataset_format("embedding_nvdocs")


@pytest.fixture(scope="session")
def embedding_nvdocs_smoke_cache() -> Path:
    """A small sample of the mined NVDocs ``training.jsonl``, for smoke runs."""
    return stage_assets.sync_dataset_format("embedding_nvdocs_smoke")


@pytest.fixture(scope="session")
def grpo_math_env_cache() -> Path:
    """Gym ``math_with_judge`` server packaged as a ``wheels-v1`` environment."""
    return stage_assets.sync_dataset_format("grpo_math_env")


@pytest.fixture(scope="session")
def grpo_math_env_native_cache() -> Path:
    """The same server as a ``native-v1`` tree; the sandbox installs it from an index."""
    return stage_assets.sync_dataset_format("grpo_math_env_native")


@pytest.fixture(scope="session")
def grpo_math_env_native_ref_cache() -> Path:
    """``native-v1`` reference-only: configs that resolve to the image's built-in server."""
    return stage_assets.sync_dataset_format("grpo_math_env_native_ref")


@pytest.fixture(scope="session")
def grpo_ascii_tree_env_cache() -> Path:
    """Prime Intellect ``ascii-tree`` converted to ``adapter-wheels-v1``."""
    return stage_assets.sync_dataset_format("grpo_ascii_tree_env")


@pytest.fixture(scope="session")
def grpo_ascii_tree_smoke_cache() -> Path:
    """The converter's dataset for ``ascii-tree``, with its vendored Hugging Face cache."""
    return stage_assets.sync_dataset_format("grpo_ascii_tree_smoke")


@pytest.fixture(scope="session")
def grpo_math_smoke_cache() -> Path:
    return stage_assets.sync_dataset_format("grpo_math_smoke")


@pytest.fixture(scope="session")
def grpo_math_uplift_cache() -> Path:
    return stage_assets.sync_dataset_format("grpo_math_uplift")


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
def nvdocs_embedding_fileset(client: NemoClient, customizer_workspace: str, embedding_nvdocs_cache: Path) -> str:
    """One fileset serves as both automodel ``dataset.training`` and the retrieve-eval ``dataset``."""
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=embedding_nvdocs_cache,
        name_prefix="nvdocs-embed",
        files={"training.jsonl": "training.jsonl", "eval_beir": "eval_beir"},
    )


@pytest.fixture(scope="module")
def squad_smoke_fileset(client: NemoClient, customizer_workspace: str, chat_format_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=chat_format_cache,
        name_prefix="squad-smoke",
        files={"train.jsonl": "training.jsonl", "validation.jsonl": "validation.jsonl"},
        max_rows=SMOKE_DATASET_ROWS,
    )


@pytest.fixture(scope="module")
def unsloth_smoke_train_fileset(client: NemoClient, customizer_workspace: str, chat_format_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=chat_format_cache,
        name_prefix="squad-smoke-train",
        files={"train.jsonl": "training.jsonl"},
        max_rows=SMOKE_DATASET_ROWS,
    )


@pytest.fixture(scope="module")
def unsloth_smoke_val_fileset(client: NemoClient, customizer_workspace: str, chat_format_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=chat_format_cache,
        name_prefix="squad-smoke-val",
        files={"validation.jsonl": "validation.jsonl"},
        max_rows=SMOKE_DATASET_ROWS,
    )


@pytest.fixture(scope="module")
def helpsteer_dpo_smoke_fileset(client: NemoClient, customizer_workspace: str, dpo_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=dpo_cache,
        name_prefix="helpsteer3-dpo-smoke",
        files={"training.jsonl": "training.jsonl", "validation.jsonl": "validation.jsonl"},
        max_rows=SMOKE_DATASET_ROWS,
    )


@pytest.fixture(scope="module")
def nvdocs_embedding_smoke_fileset(
    client: NemoClient, customizer_workspace: str, embedding_nvdocs_smoke_cache: Path
) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=embedding_nvdocs_smoke_cache,
        name_prefix="nvdocs-embed-smoke",
        files={"training.jsonl": "training.jsonl"},
    )


def _stage_tree(client: NemoClient, workspace: str, local_dir: Path, name_prefix: str, purpose: FilesetPurpose) -> str:
    """Upload every top-level entry of ``local_dir``, hidden ones included, to the fileset root."""
    return stage_assets.stage_dataset_fileset(
        client,
        workspace,
        local_dir=local_dir,
        name_prefix=name_prefix,
        files={path.name: path.name for path in sorted(local_dir.iterdir())},
        purpose=purpose,
    )


@pytest.fixture(scope="module")
def grpo_math_env_fileset(client: NemoClient, customizer_workspace: str, grpo_math_env_cache: Path) -> str:
    """Environment fileset with the package tree (``nemo-environment.yaml``, ``wheels/``, ...) at its root."""
    return _stage_tree(
        client, customizer_workspace, grpo_math_env_cache, "math-with-judge-env", FilesetPurpose.ENVIRONMENT
    )


@pytest.fixture(scope="module")
def grpo_math_env_native_fileset(
    client: NemoClient, customizer_workspace: str, grpo_math_env_native_cache: Path
) -> str:
    return _stage_tree(
        client, customizer_workspace, grpo_math_env_native_cache, "math-with-judge-native", FilesetPurpose.ENVIRONMENT
    )


@pytest.fixture(scope="module")
def grpo_math_env_native_ref_fileset(
    client: NemoClient, customizer_workspace: str, grpo_math_env_native_ref_cache: Path
) -> str:
    return _stage_tree(
        client,
        customizer_workspace,
        grpo_math_env_native_ref_cache,
        "math-with-judge-native-ref",
        FilesetPurpose.ENVIRONMENT,
    )


@pytest.fixture(scope="module")
def grpo_ascii_tree_env_fileset(client: NemoClient, customizer_workspace: str, grpo_ascii_tree_env_cache: Path) -> str:
    return _stage_tree(
        client, customizer_workspace, grpo_ascii_tree_env_cache, "ascii-tree-env", FilesetPurpose.ENVIRONMENT
    )


@pytest.fixture(scope="module")
def grpo_ascii_tree_smoke_fileset(
    client: NemoClient, customizer_workspace: str, grpo_ascii_tree_smoke_cache: Path
) -> str:
    """The whole dataset tree: the sandbox reads the vendored ``.huggingface`` cache offline."""
    return _stage_tree(
        client, customizer_workspace, grpo_ascii_tree_smoke_cache, "ascii-tree-smoke", FilesetPurpose.DATASET
    )


@pytest.fixture(scope="module")
def grpo_math_smoke_fileset(client: NemoClient, customizer_workspace: str, grpo_math_smoke_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=grpo_math_smoke_cache,
        name_prefix="math-with-judge-smoke",
        files={"training.jsonl": "training.jsonl", "validation.jsonl": "validation.jsonl"},
    )


@pytest.fixture(scope="module")
def grpo_math_uplift_fileset(client: NemoClient, customizer_workspace: str, grpo_math_uplift_cache: Path) -> str:
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=grpo_math_uplift_cache,
        name_prefix="math-with-judge-uplift",
        files={"training.jsonl": "training.jsonl", "validation.jsonl": "validation.jsonl"},
    )


@pytest.fixture(scope="module")
def vllm_lora_template(client: NemoClient, customizer_workspace: str) -> Iterator[str]:
    """Unbound LoRA-enabled vLLM config that smoke jobs auto-deploy LLM outputs with."""
    name = jobs.create_vllm_deployment_template(client, customizer_workspace, lora_enabled=True)
    yield name
    _delete_deployment_config(client, customizer_workspace, name)


@pytest.fixture(scope="module")
def vllm_template(client: NemoClient, customizer_workspace: str) -> Iterator[str]:
    """Unbound vLLM config without LoRA, for the embedding model."""
    name = jobs.create_vllm_deployment_template(client, customizer_workspace, lora_enabled=False)
    yield name
    _delete_deployment_config(client, customizer_workspace, name)


def _delete_deployment_config(client: NemoClient, workspace: str, name: str) -> None:
    try:
        ModelsClient.from_client(client).delete_deployment_config(name=name, workspace=workspace)
    except Exception:
        logger.warning("Failed to delete deployment config %s/%s", workspace, name, exc_info=True)


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
    qwen3_model_cache: Path,
) -> str:
    return stage_assets.stage_model_entity(
        client,
        customizer_workspace,
        entity_name="qwen3-0-6b-unsloth",
        snapshot_dir=qwen3_model_cache,
        hf_repo="Qwen/Qwen3-0.6B",
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


@pytest.fixture(scope="module")
def embed_base_entity(client: NemoClient, customizer_workspace: str, embed_model_cache: Path) -> str:
    return stage_assets.stage_model_entity(
        client,
        customizer_workspace,
        entity_name="nemotron-3-embed-1b",
        snapshot_dir=embed_model_cache,
        hf_repo=EMBED_HF_REPO,
    )


@pytest.fixture(scope="module")
def nemotron_lightning_base_entity(
    client: NemoClient, customizer_workspace: str, nemotron_lightning_model_cache: Path
) -> str:
    return stage_assets.stage_model_entity(
        client,
        customizer_workspace,
        entity_name="nemotron-3-5-lightning-30b-a3b",
        snapshot_dir=nemotron_lightning_model_cache,
        hf_repo=NEMOTRON_LIGHTNING_HF_REPO,
    )
