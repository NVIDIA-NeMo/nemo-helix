# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform ``fileset`` workspace-seed handler for agent-eval.

The SDK's workspace-seed machinery ships only ``inline``/``path`` kinds and knows nothing about
filesets. This module registers a ``fileset`` :class:`~nemo_evaluator_sdk.agent_eval.workspace_seeds.SeedHandler`
as an import side effect, so that when the evaluator plugin runs an agent-eval job, tasks may stage a
file from a stored fileset. Resolution happens at seed time against the Files service, using the
running job's task SDK — the SDK layer never gains a files dependency or awareness of this kind.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from typing import Any, Literal

from nemo_evaluator.filesets import FilesetRef, download_dataset_sync
from nemo_evaluator_sdk.agent_eval.workspace_seeds import WorkspaceSeedError, register_seed_handler
from nemo_helix_plugin.client_provider import get_task_nemo_client
from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Service identity used to build the task SDK (matches ``tasks/agent_evaluate.py``).
_EVALUATOR_SERVICE = "evaluator"


class FilesetSeed(BaseModel):
    """A reference to a stored fileset, resolved via the Files service at seed time."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["fileset"] = "fileset"
    ref: str = Field(
        description="Fileset reference: 'workspace/name', or 'workspace/name#path' for a single file.",
    )

    @field_validator("ref")
    @classmethod
    def _validate_ref_shape(cls, value: str) -> str:
        # Validate the 'workspace/name[#fragment]' shape only — no Files-service call.
        base = value.split("#", 1)[0]
        parts = [part for part in base.split("/") if part]
        if len(parts) != 2:
            raise ValueError(f"fileset ref must be 'workspace/name' or 'workspace/name#path', got {value!r}")
        return value


class FilesetSeedHandler:
    """Resolves a :class:`FilesetSeed` to bytes by downloading the referenced file via the Files service."""

    kind = "fileset"

    def parse(self, value: Mapping[str, Any]) -> BaseModel:
        return FilesetSeed.model_validate(value)

    def resolve(self, seed: BaseModel) -> bytes:
        assert isinstance(seed, FilesetSeed)
        # Acquire the client at resolve time from the running job's ambient identity.
        client = get_task_nemo_client(_EVALUATOR_SERVICE)
        with tempfile.TemporaryDirectory() as staging:
            try:
                downloaded = download_dataset_sync(client, FilesetRef(root=seed.ref), staging)
            except Exception as exc:  # noqa: BLE001 - surface any resolution failure as a seed error
                raise WorkspaceSeedError(f"fileset seed {seed.ref!r} could not be resolved: {exc}") from exc
            files = [downloaded] if downloaded.is_file() else sorted(p for p in downloaded.rglob("*") if p.is_file())
            if len(files) != 1:
                raise WorkspaceSeedError(
                    f"fileset seed {seed.ref!r} resolved to {len(files)} files; a seed maps one path to one "
                    "file — reference a single file with a '#path' fragment."
                )
            return files[0].read_bytes()


register_seed_handler(FilesetSeedHandler())
