# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation tests for UnslothJobInput / UnslothJobOutput."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_unsloth_plugin.schema import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputRequest,
    ScheduleSpec,
    TrainingSpec,
    UnslothJobInput,
    UnslothJobOutput,
)
from nemo_unsloth_plugin.transform import transform_input_to_output
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients
from pydantic import ValidationError

BASE_URL = "http://test"


def _model_spec_json(*, is_embedding: bool, head_type: str | None) -> dict[str, object] | None:
    if not is_embedding and head_type is None:
        return None
    return {
        "context_size": 2048,
        "head_type": head_type or "unknown",
        "is_embedding_model": is_embedding,
        "checkpoint_model_name": "Qwen2.5-0.5B-Instruct",
        "family": "qwen",
        "num_layers": 1,
        "hidden_size": 1,
        "num_attention_heads": 1,
        "num_kv_heads": 1,
        "ffn_hidden_size": 1,
        "vocab_size": 1,
        "tied_embeddings": True,
        "gated_mlp": True,
        "base_num_parameters": 1,
        "precision": "bf16",
    }


def _model_json(*, is_embedding: bool = False, head_type: str | None = None) -> dict[str, object]:
    return {
        "id": "model-m",
        "name": "m",
        "workspace": "default",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
        "spec": _model_spec_json(is_embedding=is_embedding, head_type=head_type),
        "fileset": "default/m",
        "trust_remote_code": False,
    }


def _fileset_json(workspace: str, name: str) -> dict[str, object]:
    return {
        "id": f"{workspace}-{name}",
        "name": name,
        "workspace": workspace,
        "description": "",
        "purpose": "generic",
        "storage": {"type": "local", "path": "/tmp/files"},
        "metadata": {},
        "custom_fields": {},
        "project": "",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }


async def _run_transform_async(
    spec: UnslothJobInput,
    *,
    is_embedding: bool = False,
    head_type: str | None = None,
) -> UnslothJobOutput:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        parts = path.split("/")
        if path.startswith("/apis/models/v2/workspaces/"):
            return httpx.Response(
                200, request=request, json=_model_json(is_embedding=is_embedding, head_type=head_type)
            )
        if path.startswith("/apis/files/v2/workspaces/"):
            return httpx.Response(200, request=request, json=_fileset_json(parts[5], parts[7]))
        return httpx.Response(404, request=request, json={"detail": "unexpected request"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncNemoClient(base_url=BASE_URL, workspace="default", http_client=http_client)
        platform = AsyncCustomizationPlatformClients(
            files=AsyncFilesClient.from_client(client),
            models=AsyncModelsClient.from_client(client),
        )
        return await transform_input_to_output(spec, "default", platform)


def _run_transform(spec: UnslothJobInput) -> UnslothJobOutput:
    return asyncio.run(_run_transform_async(spec))


class TestCanonicalReexport:
    """Pin that the canonical types come from the service package."""

    def test_unsloth_job_output_lives_in_service(self) -> None:
        # Re-exported from the plugin for caller convenience, but the
        # source of truth is the service. Keeps the dependency direction
        # plugin → service.
        assert UnslothJobOutput.__module__ == "nmp.unsloth.schemas"


# Raw Pydantic payload under mutation in validation tests.
def _minimal_payload() -> dict[str, Any]:
    return {
        "model": {"name": "unsloth/Qwen2.5-0.5B-Instruct", "max_seq_length": 2048},
        "dataset": {"path": "/data/sample.jsonl"},
        "schedule": {"max_steps": 60},
    }


class TestMinimalShape:
    def test_minimal_payload_validates(self) -> None:
        spec = UnslothJobInput.model_validate(_minimal_payload())
        # Defaults applied
        assert spec.training.finetuning_type == "lora"
        assert spec.training.lora is not None
        assert spec.training.lora.rank == 16
        # Unsloth's recommended 7-module set
        assert spec.training.lora.target_modules == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        assert spec.optimizer.optim == "adamw_8bit"
        assert spec.hardware.precision == "bf16"

    def test_fixture_minimal_unsloth_sft_loads(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "minimal_unsloth_sft.json"
        UnslothJobInput.model_validate(json.loads(fixture.read_text()))


class TestRequiredFields:
    def test_dataset_path_required(self) -> None:
        payload = _minimal_payload()
        del payload["dataset"]
        with pytest.raises(ValidationError):
            UnslothJobInput.model_validate(payload)

    def test_model_required(self) -> None:
        payload = _minimal_payload()
        del payload["model"]
        with pytest.raises(ValidationError):
            UnslothJobInput.model_validate(payload)


class TestSchedule:
    def test_empty_schedule_defaults_to_one_epoch(self) -> None:
        # Consistent with Automodel: epochs defaults to 1 (no epochs/max_steps mutex).
        payload = _minimal_payload()
        payload["schedule"] = {}
        spec = UnslothJobInput.model_validate(payload)
        assert spec.schedule.epochs == 1
        assert spec.schedule.max_steps is None

    def test_epochs_and_max_steps_both_allowed(self) -> None:
        # max_steps caps/overrides epochs at train time; the two are not mutually exclusive.
        payload = _minimal_payload()
        payload["schedule"] = {"epochs": 3, "max_steps": 60}
        spec = UnslothJobInput.model_validate(payload)
        assert spec.schedule.epochs == 3
        assert spec.schedule.max_steps == 60

    def test_either_one_is_fine(self) -> None:
        for sched in ({"epochs": 3}, {"max_steps": 60}):
            payload = _minimal_payload()
            payload["schedule"] = sched
            UnslothJobInput.model_validate(payload)


class TestQuantizationMutex:
    def test_4bit_and_8bit_rejected(self) -> None:
        payload = _minimal_payload()
        payload["model"] = {
            "name": "x",
            "max_seq_length": 1024,
            "load_in_4bit": True,
            "load_in_8bit": True,
        }
        with pytest.raises(ValidationError, match="load_in_4bit and model.load_in_8bit"):
            UnslothJobInput.model_validate(payload)


class TestAllWeightsFinetuneRules:
    def test_all_weights_ft_rejects_4bit(self) -> None:
        payload = _minimal_payload()
        payload["training"] = {"finetuning_type": "all_weights"}
        # default load_in_4bit=True
        with pytest.raises(ValidationError, match="incompatible with 4-bit/8-bit"):
            UnslothJobInput.model_validate(payload)

    def test_all_weights_ft_rejects_lora_block(self) -> None:
        payload = _minimal_payload()
        payload["model"]["load_in_4bit"] = False
        payload["training"] = {"finetuning_type": "all_weights", "lora": {"rank": 8}}
        with pytest.raises(ValidationError, match="training.lora must be unset"):
            UnslothJobInput.model_validate(payload)

    def test_all_weights_ft_clean(self) -> None:
        payload = _minimal_payload()
        payload["model"]["load_in_4bit"] = False
        payload["training"] = {"finetuning_type": "all_weights"}
        spec = UnslothJobInput.model_validate(payload)
        assert spec.training.lora is None


class TestWarmupMutex:
    def test_warmup_steps_and_ratio_rejected(self) -> None:
        payload = _minimal_payload()
        payload["schedule"] = {"max_steps": 60, "warmup_steps": 10, "warmup_ratio": 0.1}
        with pytest.raises(ValidationError, match="warmup_steps and schedule.warmup_ratio"):
            UnslothJobInput.model_validate(payload)


class TestSaveMethodCompatibility:
    def test_merged_save_with_lora_ok(self) -> None:
        payload = _minimal_payload()
        payload["output"] = {"save_method": "merged_16bit"}
        UnslothJobInput.model_validate(payload)

    def test_merged_save_with_all_weights_rejected(self) -> None:
        payload = _minimal_payload()
        payload["model"]["load_in_4bit"] = False
        payload["training"] = {"finetuning_type": "all_weights"}
        payload["output"] = {"save_method": "merged_16bit"}
        with pytest.raises(ValidationError, match="only valid for training.finetuning_type='lora'"):
            UnslothJobInput.model_validate(payload)


class TestExtraForbidden:
    def test_unknown_top_level_rejected(self) -> None:
        payload = _minimal_payload()
        payload["mystery_field"] = "boom"
        with pytest.raises(ValidationError):
            UnslothJobInput.model_validate(payload)


class TestTransformOutput:
    def test_auto_name_when_output_omitted(self) -> None:
        spec = UnslothJobInput.model_validate(_minimal_payload())
        out = _run_transform(spec)
        # Auto-name draws from the model basename + dataset basename.
        # "Qwen2.5-0.5B-Instruct" → "Qwen2-5-0-5B-Instruct" (dots → hyphens).
        assert out.output.name.startswith("Qwen2-5-0-5B-Instruct-sample-")
        assert out.output.type == "adapter"
        assert out.output.save_method == "lora"
        # Fileset defaults to the entity name (mirrors automodel).
        assert out.output.fileset == out.output.name

    def test_explicit_name_preserved(self) -> None:
        payload = _minimal_payload()
        payload["output"] = {"name": "my-run", "save_method": "lora"}
        out = _run_transform(UnslothJobInput.model_validate(payload))
        assert out.output.name == "my-run"
        assert out.output.fileset == "my-run"

    def test_merged_inferred_as_model_type(self) -> None:
        payload = _minimal_payload()
        payload["output"] = {"save_method": "merged_4bit"}
        out = _run_transform(UnslothJobInput.model_validate(payload))
        assert out.output.type == "model"
        assert out.output.save_method == "merged_4bit"

    @pytest.mark.parametrize(
        ("is_embedding", "head_type"),
        [(True, None), (False, "cross_encoder")],
        ids=["legacy-embedding", "cross-encoder"],
    )
    def test_encoder_model_rejected(self, is_embedding: bool, head_type: str | None) -> None:
        spec = UnslothJobInput.model_validate(_minimal_payload())
        with pytest.raises(ValueError, match="Encoder-model SFT"):
            asyncio.run(_run_transform_async(spec, is_embedding=is_embedding, head_type=head_type))


class TestSubSpecExtras:
    def test_dataset_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetSpec.model_validate({"path": "/x", "junk": 1})

    def test_lora_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoRAParams.model_validate({"rank": 8, "junk": 1})

    def test_model_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLoadSpec.model_validate({"name": "x", "junk": 1})

    def test_schedule_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScheduleSpec.model_validate({"max_steps": 1, "junk": 1})

    def test_training_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSpec.model_validate({"junk": 1})

    def test_output_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OutputRequest.model_validate({"junk": 1})
