# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A LoRA adapter is registered in the configured output workspace, not the base model's."""

from types import SimpleNamespace
from typing import Any

import pytest
from nemo_helix_plugin.client.errors import ConflictError
from nhx.customization_common.schemas.model_entity import ModelEntityCreationError, ModelEntityTaskConfig
from nhx.customization_common.tasks.model_entity.run import ModelEntityRunner

OUTPUT_WORKSPACE = "ws-output"
BASE_WORKSPACE = "ws-base"


def _config() -> ModelEntityTaskConfig:
    return ModelEntityTaskConfig.model_validate(
        {
            "name": "my-adapter",
            "workspace": OUTPUT_WORKSPACE,
            "description": "adapter",
            "fileset": {"workspace": OUTPUT_WORKSPACE, "name": "my-adapter", "path": None},
            "model_entity": f"{BASE_WORKSPACE}/base-model",
            "base_model": f"{BASE_WORKSPACE}/base-model",
            "peft": {"type": "lora", "rank": 16, "alpha": 32},
            "trust_remote_code": False,
            "deployment_config": None,
        }
    )


class _Conflict(ConflictError):
    def __init__(self) -> None:
        Exception.__init__(self, "already exists")


class _Result:
    def __init__(self, value: Any) -> None:
        self._value = value

    def data(self) -> Any:
        return self._value


class _Adapter:
    def model_dump(self) -> dict:
        return {"name": "my-adapter"}


class _RecordingModels:
    """Records adapter calls. Nested (base-model-workspace) adapter endpoints are not provided on purpose."""

    def __init__(self, *, conflict: bool = False, fail_update: bool = False) -> None:
        self.conflict = conflict
        self.fail_update = fail_update
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_adapter(self, **kwargs: Any) -> _Result:
        self.calls.append(("create_adapter", kwargs))
        if self.conflict:
            raise _Conflict()
        return _Result(_Adapter())

    def update_adapter(self, **kwargs: Any) -> _Result:
        self.calls.append(("update_adapter", kwargs))
        if self.fail_update:
            raise RuntimeError("boom")
        return _Result(_Adapter())


def _run(models: _RecordingModels) -> tuple[dict, Any]:
    base_me = SimpleNamespace(name="base-model", workspace=BASE_WORKSPACE)
    runner = ModelEntityRunner(models, None, None)  # type: ignore[arg-type]
    return runner._create_or_update_adapter(_config(), base_me, f"{OUTPUT_WORKSPACE}/my-adapter")  # type: ignore[arg-type]


def test_create_registers_adapter_in_output_workspace_with_qualified_base_model():
    models = _RecordingModels()

    result, base = _run(models)

    assert [name for name, _ in models.calls] == ["create_adapter"]
    kwargs = models.calls[0][1]
    assert kwargs["workspace"] == OUTPUT_WORKSPACE
    assert kwargs["body"].model == f"{BASE_WORKSPACE}/base-model"
    assert kwargs["body"].fileset == f"{OUTPUT_WORKSPACE}/my-adapter"
    assert result == {"name": "my-adapter"}
    assert base.workspace == BASE_WORKSPACE


def test_conflict_updates_adapter_in_output_workspace():
    models = _RecordingModels(conflict=True)

    _run(models)

    assert [name for name, _ in models.calls] == ["create_adapter", "update_adapter"]
    update = models.calls[1][1]
    assert update["workspace"] == OUTPUT_WORKSPACE
    assert update["name"] == "my-adapter"
    assert update["body"].fileset == f"{OUTPUT_WORKSPACE}/my-adapter"


def test_no_call_targets_the_base_model_workspace():
    for models in (_RecordingModels(), _RecordingModels(conflict=True)):
        _run(models)
        assert all(kwargs["workspace"] != BASE_WORKSPACE for _, kwargs in models.calls)


def test_update_failure_after_conflict_is_reported():
    with pytest.raises(ModelEntityCreationError, match="already exists but update failed"):
        _run(_RecordingModels(conflict=True, fail_update=True))
