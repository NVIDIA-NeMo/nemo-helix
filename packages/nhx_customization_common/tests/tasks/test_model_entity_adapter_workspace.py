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
    """A task config whose output workspace differs from the base model's workspace."""
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
    """The error the platform raises when the adapter name already exists."""

    def __init__(self) -> None:
        """Skip ``ConflictError``'s response-based constructor, which needs an HTTP response."""
        Exception.__init__(self, "already exists")


class _Result:
    """Stands in for a client response; ``data()`` returns the payload."""

    def __init__(self, value: Any) -> None:
        """Hold the payload ``data()`` will return."""
        self._value = value

    def data(self) -> Any:
        """Return the payload."""
        return self._value


class _Adapter:
    """A minimal adapter payload."""

    def __init__(self, model: str | None = f"{BASE_WORKSPACE}/base-model") -> None:
        """Hold the base model reference the adapter reports."""
        self.model = model

    def model_dump(self) -> dict:
        """Return the adapter as a plain dict."""
        return {"name": "my-adapter"}


class _RecordingModels:
    """Records adapter calls. Nested (base-model-workspace) adapter endpoints are not provided on purpose."""

    def __init__(
        self,
        *,
        conflict: bool = False,
        fail_update: bool = False,
        existing_model: str | None = f"{BASE_WORKSPACE}/base-model",
    ) -> None:
        """Choose whether create conflicts, whether the update fails, and the existing adapter's base model."""
        self.existing_model = existing_model
        self.conflict = conflict
        self.fail_update = fail_update
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_adapter(self, **kwargs: Any) -> _Result:
        """Record the call, raising a conflict when configured to."""
        self.calls.append(("create_adapter", kwargs))
        if self.conflict:
            raise _Conflict()
        return _Result(_Adapter())

    def get_adapter(self, **kwargs: Any) -> _Result:
        """Record the lookup and return the existing adapter."""
        self.calls.append(("get_adapter", kwargs))
        return _Result(_Adapter(self.existing_model))

    def update_adapter(self, **kwargs: Any) -> _Result:
        """Record the call, failing when configured to."""
        self.calls.append(("update_adapter", kwargs))
        if self.fail_update:
            raise RuntimeError("boom")
        return _Result(_Adapter())


def _run(models: _RecordingModels) -> tuple[dict, Any]:
    """Run the runner's adapter step against ``models`` with the base model in a different workspace."""
    base_me: Any = SimpleNamespace(name="base-model", workspace=BASE_WORKSPACE)
    runner_args: Any = (models, None, None)
    runner = ModelEntityRunner(*runner_args)
    config: Any = _config()
    return runner._create_or_update_adapter(config, base_me, f"{OUTPUT_WORKSPACE}/my-adapter")


def test_create_registers_adapter_in_output_workspace_with_qualified_base_model():
    """The adapter is created in the output workspace and names the base model as ``workspace/name``."""
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
    """A name conflict updates the existing adapter in the output workspace."""
    models = _RecordingModels(conflict=True)

    _run(models)

    assert [name for name, _ in models.calls] == ["create_adapter", "get_adapter", "update_adapter"]
    update = models.calls[2][1]
    assert update["workspace"] == OUTPUT_WORKSPACE
    assert update["name"] == "my-adapter"
    assert update["body"].fileset == f"{OUTPUT_WORKSPACE}/my-adapter"


def test_no_call_targets_the_base_model_workspace():
    """No adapter call is made against the base model's workspace."""
    for models in (_RecordingModels(), _RecordingModels(conflict=True)):
        _run(models)
        assert all(kwargs["workspace"] != BASE_WORKSPACE for _, kwargs in models.calls)


def test_update_failure_after_conflict_is_reported():
    """A failed update after a conflict is raised, not swallowed."""
    with pytest.raises(ModelEntityCreationError, match="already exists but update failed"):
        _run(_RecordingModels(conflict=True, fail_update=True))


def test_conflict_with_a_different_base_model_is_rejected_without_updating():
    """An adapter of that name on another base model is not repointed, and the error is not relabeled."""
    models = _RecordingModels(conflict=True, existing_model="ws-other/other-model")

    with pytest.raises(ModelEntityCreationError, match="refusing to update") as excinfo:
        _run(models)

    assert "update failed" not in str(excinfo.value)
    assert [name for name, _ in models.calls] == ["create_adapter", "get_adapter"]


def test_unqualified_existing_model_resolves_against_the_adapter_workspace():
    """A bare model name on the existing adapter means the adapter's workspace, so it does not match ws-base."""
    models = _RecordingModels(conflict=True, existing_model="base-model")

    with pytest.raises(ModelEntityCreationError, match="refusing to update"):
        _run(models)

    assert "update_adapter" not in [name for name, _ in models.calls]
