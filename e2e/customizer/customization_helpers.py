# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for plugin-based customization E2E tests."""

import json
import logging
import os
import re
import uuid

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest
from nemo_helix_plugin.schema import SecretRef
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.testing.e2e.customizer import wait_for_model_spec
from pydantic import SecretStr

logger = logging.getLogger(__name__)

_NAME_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def unique_name(prefix: str) -> str:
    """Return a short, API-safe test resource name."""
    clean_prefix = _NAME_SAFE_RE.sub("-", prefix.lower()).strip("-") or "e2e"
    clean_prefix = clean_prefix[:40].rstrip("-")
    return f"{clean_prefix}-{uuid.uuid4().hex[:8]}"


def create_hf_token_secret(
    client: NemoClient,
    workspace: str,
    *,
    env_var: str = "HF_TOKEN",
    name_prefix: str = "e2e-hf-token",
) -> str | None:
    """Create an HF token secret when the configured environment variable is set."""
    token = os.environ.get(env_var)
    if not token:
        logger.info("%s is not set; creating HuggingFace filesets without a token", env_var)
        return None

    secret_name = unique_name(name_prefix)
    logger.info("Creating HuggingFace credential resource in workspace %s", workspace)
    SecretsClient.from_client(client).create_secret(
        workspace=workspace,
        body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(token)),
    )
    return secret_name


def delete_secret_if_present(client: NemoClient, workspace: str, secret_name: str | None) -> None:
    """Best-effort delete for optional E2E secrets."""
    if secret_name is None:
        return
    try:
        SecretsClient.from_client(client).delete_secret(name=secret_name, workspace=workspace)
        logger.info("Deleted credential resource in workspace %s", workspace)
    except NotFoundError:
        logger.debug("credential resource in workspace %s already deleted", workspace)
    except Exception:
        logger.warning("Failed to delete credential resource in workspace %s", workspace, exc_info=True)


def create_hf_model_entity(
    client: NemoClient,
    workspace: str,
    *,
    entity_name: str,
    hf_repo: str,
    token_secret: str | None = None,
    wait_timeout: int = 600,
) -> str:
    """Register a model entity backed by a Hugging Face fileset (weights pulled at runtime)."""
    fileset_name = f"{entity_name}-weights"
    storage = HuggingfaceStorageConfig(
        repo_id=hf_repo,
        repo_type="model",
        token_secret=SecretRef(token_secret) if token_secret else None,
    )

    logger.info("Creating HuggingFace fileset %s/%s for %s", workspace, fileset_name, hf_repo)
    FilesClient.from_client(client).create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=fileset_name,
            description=f"E2E model source: {hf_repo}",
            storage=storage,
            purpose=FilesetPurpose.MODEL,
        ),
        exist_ok=True,
    )
    ModelsClient.from_client(client).create_model(
        workspace=workspace,
        body=CreateModelEntityRequest(
            name=entity_name,
            fileset=f"{workspace}/{fileset_name}",
            description=f"E2E model entity for {hf_repo}",
            custom_fields={"hf_model_id": hf_repo},
        ),
        exist_ok=True,
    )
    wait_for_model_spec(client, workspace, entity_name, timeout=wait_timeout)
    return entity_name


def assert_fileset_has_files(client: NemoClient, workspace: str, fileset_name: str) -> None:
    """Assert that a fileset exists and has at least one file."""
    files = FilesClient.from_client(client).list_files(name=fileset_name, workspace=workspace).data().data
    logger.info("Fileset %s/%s has %d file(s)", workspace, fileset_name, len(files))
    for file_info in files:
        logger.info("  %s", file_info.path)
    assert files, f"Expected fileset {workspace}/{fileset_name} to contain files"


def assert_output_registered(
    client: NemoClient,
    workspace: str,
    *,
    base_model_name: str,
    output_name: str,
) -> None:
    """Assert a completed backend output exists as a model entity or adapter."""
    models = ModelsClient.from_client(client)
    try:
        models.get_model(name=output_name, workspace=workspace)
        logger.info("Output registered as model entity %s/%s", workspace, output_name)
        return
    except NotFoundError:
        pass

    base_model = models.get_model(name=base_model_name, workspace=workspace).data()
    adapter_names = [adapter.name for adapter in (base_model.adapters or [])]
    if output_name in adapter_names:
        logger.info("Output registered as adapter %s on base model %s", output_name, base_model_name)
        return

    try:
        models.get_adapter(name=output_name, workspace=workspace)
        logger.info("Output registered as top-level adapter %s/%s", workspace, output_name)
        return
    except NotFoundError:
        pass

    pytest.fail(
        f"Output {output_name} was not registered as a model entity or adapter. "
        f"Adapters on {base_model_name}: {adapter_names}"
    )


def _format_status(status: object) -> str:
    dump_json = getattr(status, "model_dump_json", None)
    if callable(dump_json):
        return str(dump_json(indent=2))

    dump = getattr(status, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), indent=2, default=str)

    return str(status)
