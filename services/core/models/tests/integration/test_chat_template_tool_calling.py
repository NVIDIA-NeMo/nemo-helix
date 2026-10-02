# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for chat_template and tool_call_config end-to-end flow.

Design under test
-----------------
Users configure chat_template and tool_call_config on a **fileset** via
``metadata.model.tool_calling``, NOT directly on the model entity.  The async
model-spec background task reads the fileset's ``metadata.model.tool_calling``,
merges them into the ``ModelSpec``, and writes the result back to the
model entity with ``sdk.models.update(spec=...)``.  Downstream, the Docker
backend reads ``model_entity.spec`` (and optional deployment-level overrides)
to build the NIM container's environment variables.

These tests verify the full pipeline:
1. Fileset metadata.model.tool_calling → _merge_fileset_metadata → ModelSpec
2. Model spec task updates model entity via API → retrieve preserves spec
3. _compile_env_vars produces correct NIM_* env vars from real entities

Unlike the removed docker backend unit tests (which used MagicMock),
these tests use:
- Real Pydantic ModelSpec / ToolCallConfig objects for the merge step
- The actual Models API (via in-memory test client) for the CRUD step
- Real ModelDeploymentConfigModelSpec / ContainerExecutorConfig schema for the env-vars step
"""

import sys
import types
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nhx.common.config import Runtime
from nhx.common.files.metadata import FilesetMetadata, ModelMetadataContent, ToolCallingMetadataContent
from nhx.core.models.app import ModelWeightsType, get_model_weights_type
from nhx.core.models.config import config as models_config
from nhx.core.models.controllers.backends.common import deployment_config_view
from nhx.core.models.controllers.backends.deployments_plugin.compiler import compile_model_deployment
from nhx.core.models.controllers.backends.deployments_plugin.config import DeploymentsPluginConfig
from nhx.core.models.controllers.backends.deployments_plugin.nim_compiler import (
    _TOOL_CALL_PLUGIN_PATH,
    compile_nim_server_env,
)
from nhx.core.models.controllers.backends.deployments_plugin.resolve import ResolvedPluginDeployment
from nhx.core.models.schemas import (
    ContainerExecutorConfig,
    ModelDeploymentConfigModelSpec,
    ModelSpec,
    ToolCallConfig,
)
from nhx.core.models.tasks.model_spec.run import ModelSpecRunner
from nhx.core.models.tasks.model_spec.schemas import ModelSpecTaskConfig, NHXJobContext
from nhx.testing import ClientContext

# ============================================================================
# Constants
# ============================================================================

DEFAULT_WORKSPACE = "default"

# Minimal valid ModelSpec used throughout the tests
MINIMAL_SPEC = {
    "checkpoint_model_name": "meta-llama/Llama-3.2-1b-instruct",
    "family": "llama",
    "num_layers": 16,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_kv_heads": 8,
    "ffn_hidden_size": 8192,
    "vocab_size": 128256,
    "tied_embeddings": True,
    "gated_mlp": True,
    "base_num_parameters": 1_236_000_000,
    "precision": "bfloat16",
    "is_chat": True,
    "context_size": 131072,
}

SAMPLE_CHAT_TEMPLATE = (
    "{%- set loop_messages = messages %}"
    "{%- for message in loop_messages %}"
    "{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + message['content'] | trim + '<|eot_id|>' %}"
    "{%- if loop.index0 == 0 %}{%- set content = '<|begin_of_text|>' + content %}{%- endif %}"
    "{{ content }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}"
)


def _set_llama_config(config, *, chat_template=None, tool_call_config=None) -> None:
    """Populate a config mock with the engine-split deployment shape for the standard llama model.

    Splits the model-side and executor-side fields into real schema objects so the
    backend's ``deployment_config_view`` reads resolve correctly.
    """
    config.engine = "nim"
    config.model_spec = ModelDeploymentConfigModelSpec(
        model_name="llama-3.2-1b-instruct",
        model_namespace="meta",
        lora_enabled=False,
        chat_template=chat_template,
        tool_call_config=tool_call_config,
    )
    config.executor_config = ContainerExecutorConfig(
        gpu=1,
        disk_size="50Gi",
        image_name="nvcr.io/nim/meta/llama-3.2-1b-instruct",
        image_tag="1.8.6",
        additional_envs=None,
    )


def _resolved_plugin(
    config, model_entity, deployment, *, runtime: Runtime = Runtime.DOCKER
) -> ResolvedPluginDeployment:
    view = deployment_config_view(config)
    return ResolvedPluginDeployment(
        deployment=deployment,
        config=config,
        model_entity=model_entity,
        view=view,
        weights_type=get_model_weights_type(
            model_deployment=deployment,
            model_deployment_config=config,
            model_entity=model_entity,
        ),
        model_namespace=view.model_namespace,
        model_name=view.model_name,
        model_revision=view.model_revision,
        files_hf_url="http://testserver/apis/files/v2/hf",
        huggingface_model_puller="nvcr.io/nvidia/model-puller:latest",
        runtime=runtime,
    )


def _compile_nim_env(
    config,
    model_entity,
    deployment,
    *,
    weighted: bool = False,
    tool_call_plugin_path: str | None = None,
) -> dict[str, str]:
    return compile_nim_server_env(
        _resolved_plugin(config, model_entity, deployment),
        DeploymentsPluginConfig(),
        weighted=weighted,
        tool_call_plugin_path=tool_call_plugin_path,
    )


def _compile_k8s_deployment(config, model_entity, deployment):
    view = deployment_config_view(config)
    resolved = ResolvedPluginDeployment(
        deployment=deployment,
        config=config,
        model_entity=model_entity,
        view=view,
        weights_type=ModelWeightsType.FILES_SERVICE,
        model_namespace=view.model_namespace,
        model_name=view.model_name,
        model_revision=view.model_revision,
        files_hf_url="http://testserver/apis/files/v2/hf",
        huggingface_model_puller="nvcr.io/nvidia/model-puller:latest",
        runtime=Runtime.KUBERNETES,
    )
    return compile_model_deployment(
        resolved,
        DeploymentsPluginConfig(busybox_image="busybox", busybox_image_tag="latest"),
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_deployment():
    """Minimal deployment mock."""
    dep = MagicMock()
    dep.workspace = DEFAULT_WORKSPACE
    dep.name = "integ-deployment"
    dep.entity_version = "v1"
    dep.status = "PENDING"
    dep.config = None
    dep.config_version = None
    dep.model_provider_id = None
    return dep


# ============================================================================
# Part 1: Full analyze_checkpoint Flow (Update Fileset → Task → Verify Model)
#
# These tests verify the complete pipeline:
#   1. Create model entity with fileset reference (no spec)
#   2. User updates the fileset with metadata.model.tool_calling
#   3. Model-spec background task runs (analyze_checkpoint):
#      — reads the updated fileset, merges metadata into the ModelSpec,
#        and writes the result back to the model entity via sdk.models.update()
#   4. Retrieve the model entity and verify the spec was populated
#
# The Files service is not running in these tests, so fileset operations
# (update + retrieve) are mocked.  Model entity CRUD flows through the
# real Models API.
# ============================================================================

FILESET_NAME = "my-llama-fileset"


def _update_fileset_and_run_task(test_clients, model_name, metadata, tmp_path):
    """Simulate: user patches fileset → model-spec task runs.

    1. **Update fileset** — mock the files client to simulate the user adding
       ``metadata`` to their fileset.
    2. **Run analyze_checkpoint** — the task calls ``client_from_platform(sdk, FilesClient).get_fileset()``
       (mocked to return the now-updated fileset), merges ``metadata.model.tool_calling``
       into a ``ModelSpec``, and calls the *real* ``sdk.models.update(spec=...)``.

    ``nhx.core.models.parallelism.api`` depends on torch/accelerate (GPU deps
    not available in the test environment), so we inject a mock module into
    ``sys.modules`` so the lazy import inside ``analyze_checkpoint`` resolves
    without touching the real GPU module.
    """
    sdk = test_clients.sdk

    # -- Step 1: Build the updated fileset representation ---------------------
    updated_fileset = SimpleNamespace(
        name=FILESET_NAME,
        workspace=DEFAULT_WORKSPACE,
        metadata=metadata,
        storage=None,
    )

    mock_files_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data.return_value = updated_fileset
    mock_files_client.get_fileset.return_value = mock_response

    # -- Step 2: Model-spec background task runs ------------------------------
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    job_ctx = NHXJobContext(
        workspace=DEFAULT_WORKSPACE,
        job_id="test-job",
        attempt_id="attempt-0",
        step="model-spec",
        task="model-spec",
        storage_path=tmp_path,
        config_path=None,
        jobs_url=None,
        files_url=None,
        models_url=None,
    )

    runner = ModelSpecRunner(sdk=sdk, job_ctx=job_ctx)

    # Base ModelSpec that infer_model_cfg_from_hf would normally produce
    base_spec = ModelSpec(**MINIMAL_SPEC)

    # Mock the torch-dependent parallelism module (not available in test env)
    mock_api = types.ModuleType("nhx.core.models.parallelism.api")
    mock_api.infer_model_cfg_from_hf = MagicMock(return_value=base_spec)
    mock_api.find_minimum_gpus_from_metadata = MagicMock(return_value=(1, {}))

    modules_patch = {"nhx.core.models.parallelism.api": mock_api}
    if "nhx.core.models.parallelism" not in sys.modules:
        parent = types.ModuleType("nhx.core.models.parallelism")
        parent.__path__ = []
        modules_patch["nhx.core.models.parallelism"] = parent

    # Task calls client_from_platform(sdk, FilesClient).get_fileset() → gets the updated fileset
    with (
        patch.dict(sys.modules, modules_patch),
        patch("nhx.core.models.tasks.model_spec.run.client_from_platform", return_value=mock_files_client),
        patch.object(sdk.files, "list", return_value=SimpleNamespace(data=[])),
        patch.object(runner.filesystem_sdk, "get"),
        patch("nhx.core.models.tasks.model_spec.run.os.listdir", return_value=["config.json"]),
    ):
        config = ModelSpecTaskConfig(workspace=DEFAULT_WORKSPACE, name=model_name)
        runner.analyze_checkpoint(config)


@pytest.mark.skip(reason="Fileset validation returns 400 — model creation requires valid fileset")
def test_update_fileset_chat_template_populates_model_entity_spec(test_clients: ClientContext, tmp_path):
    """User updates fileset with chat_template via metadata → task populates model entity spec."""
    model_name = f"integ-task-tmpl-{uuid.uuid4().hex[:8]}"

    # 1. Create model entity pointing to a fileset (no spec yet)
    model_data = {"name": model_name, "fileset": f"{DEFAULT_WORKSPACE}/{FILESET_NAME}"}
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 201
    assert response.json().get("spec") is None

    try:
        # 2. User updates fileset metadata  →  3. Task runs
        metadata = FilesetMetadata(
            model=ModelMetadataContent(tool_calling=ToolCallingMetadataContent(chat_template=SAMPLE_CHAT_TEMPLATE))
        )
        _update_fileset_and_run_task(test_clients, model_name, metadata, tmp_path)

        # 4. Retrieve model entity — spec should have chat_template from fileset
        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")
        assert response.status_code == 200
        spec = response.json()["spec"]
        assert spec["chat_template"] == SAMPLE_CHAT_TEMPLATE
        assert spec.get("tool_call_config") is None
    finally:
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")


@pytest.mark.skip(reason="Fileset validation returns 400 — model creation requires valid fileset")
def test_update_fileset_tool_call_config_populates_model_entity_spec(test_clients: ClientContext, tmp_path):
    """User updates fileset with tool_call_config via metadata → task populates model entity spec."""
    model_name = f"integ-task-tcc-{uuid.uuid4().hex[:8]}"

    model_data = {"name": model_name, "fileset": f"{DEFAULT_WORKSPACE}/{FILESET_NAME}"}
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 201

    try:
        metadata = FilesetMetadata(
            model=ModelMetadataContent(
                tool_calling=ToolCallingMetadataContent(
                    tool_call_parser="llama3_json",
                    tool_call_plugin="default/my-plugin",
                    auto_tool_choice=True,
                )
            )
        )
        _update_fileset_and_run_task(test_clients, model_name, metadata, tmp_path)

        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")
        assert response.status_code == 200
        tcc = response.json()["spec"]["tool_call_config"]
        assert tcc["tool_call_parser"] == "llama3_json"
        assert tcc["tool_call_plugin"] == "default/my-plugin"
        assert tcc["auto_tool_choice"] is True
    finally:
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")


@pytest.mark.skip(reason="Fileset validation returns 400 — model creation requires valid fileset")
def test_update_fileset_both_fields_populates_model_entity_spec(test_clients: ClientContext, tmp_path):
    """User updates fileset with both chat_template + tool_call_config via metadata → model spec."""
    model_name = f"integ-task-both-{uuid.uuid4().hex[:8]}"

    model_data = {"name": model_name, "fileset": f"{DEFAULT_WORKSPACE}/{FILESET_NAME}"}
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 201

    try:
        metadata = FilesetMetadata(
            model=ModelMetadataContent(
                tool_calling=ToolCallingMetadataContent(
                    chat_template=SAMPLE_CHAT_TEMPLATE,
                    tool_call_parser="hermes",
                    auto_tool_choice=False,
                )
            )
        )
        _update_fileset_and_run_task(test_clients, model_name, metadata, tmp_path)

        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")
        assert response.status_code == 200
        spec = response.json()["spec"]
        assert spec["chat_template"] == SAMPLE_CHAT_TEMPLATE
        assert spec["tool_call_config"]["tool_call_parser"] == "hermes"
        assert spec["tool_call_config"]["auto_tool_choice"] is False
    finally:
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")


@pytest.mark.skip(reason="Fileset validation returns 400 — model creation requires valid fileset")
def test_update_fileset_empty_metadata_leaves_no_chat_tool_fields(test_clients: ClientContext, tmp_path):
    """User updates fileset with empty metadata → spec has no chat/tool fields."""
    model_name = f"integ-task-empty-{uuid.uuid4().hex[:8]}"

    model_data = {"name": model_name, "fileset": f"{DEFAULT_WORKSPACE}/{FILESET_NAME}"}
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=model_data)
    assert response.status_code == 201

    try:
        metadata = FilesetMetadata()
        _update_fileset_and_run_task(test_clients, model_name, metadata, tmp_path)

        response = test_clients.test_client.get(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")
        assert response.status_code == 200
        spec = response.json()["spec"]
        assert spec.get("chat_template") is None
        assert spec.get("tool_call_config") is None
    finally:
        test_clients.test_client.delete(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models/{model_name}")


# ============================================================================
# Part 2: _merge_fileset_metadata — Real ModelSpec Objects
# ============================================================================


class TestMergeFilesetMetadata:
    """Test the static _merge_fileset_metadata method with real Pydantic objects."""

    @staticmethod
    def _make_model_spec(**overrides) -> ModelSpec:
        """Create a minimal ModelSpec with optional overrides."""
        defaults = {
            "checkpoint_model_name": "meta-llama/Llama-3.2-1b-instruct",
            "family": "llama",
            "num_layers": 16,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "ffn_hidden_size": 8192,
            "vocab_size": 128256,
            "tied_embeddings": True,
            "gated_mlp": True,
            "base_num_parameters": 1_236_000_000,
            "precision": "bfloat16",
        }
        defaults.update(overrides)
        return ModelSpec(**defaults)

    def test_merge_chat_template_from_fileset(self):
        """Fileset metadata with chat_template is merged into ModelSpec."""
        model_spec = self._make_model_spec()
        assert model_spec.chat_template is None

        fileset = SimpleNamespace(
            metadata=FilesetMetadata(
                model=ModelMetadataContent(tool_calling=ToolCallingMetadataContent(chat_template=SAMPLE_CHAT_TEMPLATE))
            )
        )
        ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.chat_template == SAMPLE_CHAT_TEMPLATE

    def test_merge_tool_call_config_from_fileset(self):
        """Fileset metadata with tool_call_config is merged into ModelSpec."""
        model_spec = self._make_model_spec()
        assert model_spec.tool_call_config is None

        fileset = SimpleNamespace(
            metadata=FilesetMetadata(
                model=ModelMetadataContent(
                    tool_calling=ToolCallingMetadataContent(
                        tool_call_parser="llama3_json",
                        tool_call_plugin="default/my-plugin",
                        auto_tool_choice=True,
                    )
                )
            )
        )
        with patch.object(models_config.tool_call_plugin, "enabled", True):
            ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.tool_call_config is not None
        assert isinstance(model_spec.tool_call_config, ToolCallConfig)
        assert model_spec.tool_call_config.tool_call_parser == "llama3_json"
        assert model_spec.tool_call_config.tool_call_plugin == "default/my-plugin"
        assert model_spec.tool_call_config.auto_tool_choice is True

    def test_merge_both_chat_template_and_tool_call_config(self):
        """Both fields from fileset metadata are merged."""
        model_spec = self._make_model_spec()

        fileset = SimpleNamespace(
            metadata=FilesetMetadata(
                model=ModelMetadataContent(
                    tool_calling=ToolCallingMetadataContent(
                        chat_template="my-template",
                        tool_call_parser="hermes",
                        auto_tool_choice=False,
                    )
                )
            )
        )
        ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.chat_template == "my-template"
        assert model_spec.tool_call_config.tool_call_parser == "hermes"
        assert model_spec.tool_call_config.auto_tool_choice is False
        assert model_spec.tool_call_config.tool_call_plugin is None

    def test_merge_no_tool_calling_metadata_is_noop(self):
        """No tool_calling metadata leaves ModelSpec unchanged."""
        model_spec = self._make_model_spec()

        fileset = SimpleNamespace(metadata=FilesetMetadata())
        ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.chat_template is None
        assert model_spec.tool_call_config is None

    def test_merge_none_metadata_is_noop(self):
        """None metadata leaves ModelSpec unchanged."""
        model_spec = self._make_model_spec()

        fileset = SimpleNamespace(metadata=None)
        ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.chat_template is None
        assert model_spec.tool_call_config is None

    def test_merge_overwrites_existing_values(self):
        """Fileset merge overwrites existing spec values (by design — fileset is the source of truth)."""
        model_spec = self._make_model_spec(
            chat_template="old-template",
            tool_call_config=ToolCallConfig(tool_call_parser="old-parser"),
        )

        fileset = SimpleNamespace(
            metadata=FilesetMetadata(
                model=ModelMetadataContent(
                    tool_calling=ToolCallingMetadataContent(
                        chat_template="new-template",
                        tool_call_parser="new-parser",
                        auto_tool_choice=True,
                    )
                )
            )
        )
        ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.chat_template == "new-template"
        assert model_spec.tool_call_config.tool_call_parser == "new-parser"
        assert model_spec.tool_call_config.auto_tool_choice is True

    def test_merge_partial_tool_call_config(self):
        """Only some tool_call_config fields set — others default to None."""
        model_spec = self._make_model_spec()

        fileset = SimpleNamespace(
            metadata=FilesetMetadata(
                model=ModelMetadataContent(tool_calling=ToolCallingMetadataContent(tool_call_parser="pythonic"))
            )
        )
        ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.tool_call_config.tool_call_parser == "pythonic"
        assert model_spec.tool_call_config.tool_call_plugin is None
        assert model_spec.tool_call_config.auto_tool_choice is None

    def test_merge_strips_tool_call_plugin_when_disabled(self):
        """tool_call_plugin is dropped from metadata merge when platform flag is disabled."""
        model_spec = self._make_model_spec()

        fileset = SimpleNamespace(
            metadata=FilesetMetadata(
                model=ModelMetadataContent(
                    tool_calling=ToolCallingMetadataContent(
                        tool_call_parser="pythonic",
                        tool_call_plugin="default/my-plugin",
                        auto_tool_choice=True,
                    )
                )
            )
        )
        with patch.object(models_config.tool_call_plugin, "enabled", False):
            ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.tool_call_config.tool_call_parser == "pythonic"
        assert model_spec.tool_call_config.tool_call_plugin is None
        assert model_spec.tool_call_config.auto_tool_choice is True

    def test_merge_keeps_tool_call_plugin_when_enabled(self):
        """tool_call_plugin is kept in metadata merge when platform flag is enabled."""
        model_spec = self._make_model_spec()

        fileset = SimpleNamespace(
            metadata=FilesetMetadata(
                model=ModelMetadataContent(
                    tool_calling=ToolCallingMetadataContent(
                        tool_call_parser="pythonic",
                        tool_call_plugin="default/my-plugin",
                    )
                )
            )
        )
        with patch.object(models_config.tool_call_plugin, "enabled", True):
            ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

        assert model_spec.tool_call_config.tool_call_parser == "pythonic"
        assert model_spec.tool_call_config.tool_call_plugin == "default/my-plugin"


# ============================================================================
# Part 3: Full Pipeline — fileset merge → env-var compilation
#
# Tests that the entire chain works with real Pydantic objects:
#   fileset metadata.model.tool_calling → ModelSpec → _compile_env_vars → NIM env vars
# ============================================================================


def test_pipeline_fileset_metadata_to_env_vars(sample_deployment):
    """End-to-end: fileset metadata → ModelSpec merge → _compile_env_vars.

    1. Start with a bare ModelSpec.
    2. Merge fileset metadata.model.tool_calling (chat_template + tool_call_config).
    3. Build a real ModelEntity mock with the merged spec.
    4. Call _compile_env_vars and verify the NIM_* env vars.
    """
    # Step 1: Create real ModelSpec
    model_spec = ModelSpec(**MINIMAL_SPEC)
    assert model_spec.chat_template is None
    assert model_spec.tool_call_config is None

    # Step 2: Merge fileset metadata
    fileset = SimpleNamespace(
        metadata=FilesetMetadata(
            model=ModelMetadataContent(
                tool_calling=ToolCallingMetadataContent(
                    chat_template=SAMPLE_CHAT_TEMPLATE,
                    tool_call_parser="llama3_json",
                    auto_tool_choice=True,
                )
            )
        )
    )
    ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)
    assert model_spec.chat_template == SAMPLE_CHAT_TEMPLATE
    assert model_spec.tool_call_config.tool_call_parser == "llama3_json"

    # Step 3: Build model entity with the merged spec
    model_entity = MagicMock()
    model_entity.workspace = DEFAULT_WORKSPACE
    model_entity.name = "pipeline-test-model"
    model_entity.trust_remote_code = False
    model_entity.spec = model_spec

    # Step 4: Compile env vars with a deployment config that has no overrides
    config = MagicMock()
    _set_llama_config(config)

    env_vars = _compile_nim_env(
        config,
        model_entity,
        sample_deployment,
        weighted=True,
    )

    assert env_vars["NIM_CHAT_TEMPLATE"] == SAMPLE_CHAT_TEMPLATE
    assert env_vars["NIM_TOOL_CALL_PARSER"] == "llama3_json"
    assert env_vars["NIM_ENABLE_AUTO_TOOL_CHOICE"] == "1"
    assert "NIM_TOOL_PARSER_PLUGIN" not in env_vars


def test_pipeline_deployment_overrides_fileset_values(sample_deployment):
    """End-to-end: deployment config overrides values merged from fileset."""
    model_spec = ModelSpec(**MINIMAL_SPEC)
    fileset = SimpleNamespace(
        metadata=FilesetMetadata(
            model=ModelMetadataContent(
                tool_calling=ToolCallingMetadataContent(
                    chat_template="fileset-template",
                    tool_call_parser="llama3_json",
                    auto_tool_choice=False,
                )
            )
        )
    )
    ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

    model_entity = MagicMock()
    model_entity.workspace = DEFAULT_WORKSPACE
    model_entity.name = "pipeline-override-model"
    model_entity.trust_remote_code = False
    model_entity.spec = model_spec

    # Deployment config provides its own overrides
    config = MagicMock()
    _set_llama_config(
        config,
        chat_template="deployment-override-template",
        tool_call_config={
            "tool_call_parser": "openai",
            "auto_tool_choice": True,
        },
    )

    env_vars = _compile_nim_env(
        config,
        model_entity,
        sample_deployment,
        weighted=True,
    )

    # Deployment-level values should win over fileset values
    assert env_vars["NIM_CHAT_TEMPLATE"] == "deployment-override-template"
    assert env_vars["NIM_TOOL_CALL_PARSER"] == "openai"
    assert env_vars["NIM_ENABLE_AUTO_TOOL_CHOICE"] == "1"


def test_pipeline_mixed_sources(sample_deployment):
    """End-to-end: chat_template from fileset, tool_call_config from deployment."""
    model_spec = ModelSpec(**MINIMAL_SPEC)

    fileset = SimpleNamespace(
        metadata=FilesetMetadata(
            model=ModelMetadataContent(
                tool_calling=ToolCallingMetadataContent(chat_template="fileset-only-chat-template")
            )
        )
    )
    ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)
    assert model_spec.tool_call_config is None

    model_entity = MagicMock()
    model_entity.workspace = DEFAULT_WORKSPACE
    model_entity.name = "pipeline-mixed-model"
    model_entity.trust_remote_code = False
    model_entity.spec = model_spec

    # Deployment provides only tool_call_config (no chat_template)
    config = MagicMock()
    _set_llama_config(
        config,
        tool_call_config={"tool_call_parser": "hermes", "auto_tool_choice": True},
    )

    env_vars = _compile_nim_env(
        config,
        model_entity,
        sample_deployment,
        weighted=True,
    )

    # chat_template from fileset (via model spec), tool config from deployment
    assert env_vars["NIM_CHAT_TEMPLATE"] == "fileset-only-chat-template"
    assert env_vars["NIM_TOOL_CALL_PARSER"] == "hermes"
    assert env_vars["NIM_ENABLE_AUTO_TOOL_CHOICE"] == "1"


def test_pipeline_plugin_path_flows_through(sample_deployment):
    """End-to-end: tool_call_plugin from fileset metadata → env var with plugin path."""
    model_spec = ModelSpec(**MINIMAL_SPEC)

    fileset = SimpleNamespace(
        metadata=FilesetMetadata(
            model=ModelMetadataContent(
                tool_calling=ToolCallingMetadataContent(
                    chat_template=SAMPLE_CHAT_TEMPLATE,
                    tool_call_parser="pythonic",
                    tool_call_plugin="default/my-tool-plugin",
                    auto_tool_choice=True,
                )
            )
        )
    )
    with patch.object(models_config.tool_call_plugin, "enabled", True):
        ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)

    model_entity = MagicMock()
    model_entity.workspace = DEFAULT_WORKSPACE
    model_entity.name = "pipeline-plugin-model"
    model_entity.trust_remote_code = False
    model_entity.spec = model_spec

    config = MagicMock()
    _set_llama_config(config)

    env_vars = _compile_nim_env(
        config,
        model_entity,
        sample_deployment,
        weighted=True,
        tool_call_plugin_path="/model-store/tool_call_plugin/my_plugin.py",
    )

    assert env_vars["NIM_CHAT_TEMPLATE"] == SAMPLE_CHAT_TEMPLATE
    assert env_vars["NIM_TOOL_CALL_PARSER"] == "pythonic"
    assert env_vars["NIM_TOOL_PARSER_PLUGIN"] == "/model-store/tool_call_plugin/my_plugin.py"
    assert env_vars["NIM_ENABLE_AUTO_TOOL_CHOICE"] == "1"


def test_pipeline_no_fileset_metadata_no_deployment_overrides(sample_deployment):
    """End-to-end: no fileset metadata, no deployment overrides → no tool env vars."""
    model_spec = ModelSpec(**MINIMAL_SPEC)

    fileset = SimpleNamespace(metadata=FilesetMetadata())  # No model metadata
    ModelSpecRunner._merge_fileset_metadata(fileset, model_spec)
    assert model_spec.chat_template is None
    assert model_spec.tool_call_config is None

    model_entity = MagicMock()
    model_entity.workspace = DEFAULT_WORKSPACE
    model_entity.name = "pipeline-baseline-model"
    model_entity.trust_remote_code = False
    model_entity.spec = model_spec

    config = MagicMock()
    _set_llama_config(config)

    env_vars = _compile_nim_env(
        config,
        model_entity,
        sample_deployment,
        weighted=True,
    )

    assert "NIM_CHAT_TEMPLATE" not in env_vars
    assert "NIM_TOOL_CALL_PARSER" not in env_vars
    assert "NIM_TOOL_PARSER_PLUGIN" not in env_vars
    assert "NIM_ENABLE_AUTO_TOOL_CHOICE" not in env_vars


def test_k8s_nimservice_adds_plugin_init_containers_from_model_entity(sample_deployment):
    """Model-entity tool_call_config with tool_call_plugin adds plugin init containers."""
    model_spec = ModelSpec(**MINIMAL_SPEC)
    model_spec.tool_call_config = ToolCallConfig(
        tool_call_parser="entity-parser",
        tool_call_plugin="default/entity-plugin-fileset",
        auto_tool_choice=True,
    )

    model_entity = MagicMock()
    model_entity.workspace = DEFAULT_WORKSPACE
    model_entity.name = "integ-k8s-entity-plugin-model"
    model_entity.trust_remote_code = False
    model_entity.spec = model_spec

    config = MagicMock()
    config.workspace = DEFAULT_WORKSPACE
    config.name = "integ-k8s-config"
    config.entity_version = "v1"
    _set_llama_config(config)

    compiled = _compile_k8s_deployment(config, model_entity, sample_deployment)

    assert len(compiled.server_config.init_containers) == 3
    pull_container = compiled.server_config.init_containers[1]
    assert pull_container.command == ["download", "default/entity-plugin-fileset", "--local-dir", "/scratch/plugin"]

    env_dict = {item.name: item.value for item in compiled.server_config.containers[0].env}
    assert env_dict["NIM_MODEL_NAME"] == "/model-store"
    assert env_dict["NIM_SERVED_MODEL_NAME"] == "meta/llama-3.2-1b-instruct"
    assert env_dict["NHX_MODEL_ENTITY_WORKSPACE"] == DEFAULT_WORKSPACE
    assert env_dict["NHX_MODEL_ENTITY_NAME"] == "integ-k8s-entity-plugin-model"
    assert env_dict["NIM_TOOL_CALL_PARSER"] == "entity-parser"
    assert env_dict["NIM_TOOL_PARSER_PLUGIN"] == _TOOL_CALL_PLUGIN_PATH
    assert env_dict["NIM_ENABLE_AUTO_TOOL_CHOICE"] == "1"


def test_k8s_nimservice_deployment_tool_config_takes_priority(sample_deployment):
    """Deployment-level tool_call_config overrides model-entity tool_call_config for k8s plugin init containers."""
    model_spec = ModelSpec(**MINIMAL_SPEC)
    model_spec.tool_call_config = ToolCallConfig(
        tool_call_parser="entity-parser",
        tool_call_plugin="default/entity-plugin-fileset",
        auto_tool_choice=False,
    )

    model_entity = MagicMock()
    model_entity.workspace = DEFAULT_WORKSPACE
    model_entity.name = "integ-k8s-priority-model"
    model_entity.trust_remote_code = False
    model_entity.spec = model_spec

    config = MagicMock()
    config.workspace = DEFAULT_WORKSPACE
    config.name = "integ-k8s-priority-config"
    config.entity_version = "v1"
    _set_llama_config(
        config,
        tool_call_config={
            "tool_call_parser": "deployment-parser",
            "tool_call_plugin": "default/deployment-plugin-fileset",
            "auto_tool_choice": True,
        },
    )

    compiled = _compile_k8s_deployment(config, model_entity, sample_deployment)

    assert len(compiled.server_config.init_containers) == 3
    pull_container = compiled.server_config.init_containers[1]
    assert pull_container.command == ["download", "default/deployment-plugin-fileset", "--local-dir", "/scratch/plugin"]

    env_dict = {item.name: item.value for item in compiled.server_config.containers[0].env}
    assert env_dict["NIM_MODEL_NAME"] == "/model-store"
    assert env_dict["NIM_SERVED_MODEL_NAME"] == "meta/llama-3.2-1b-instruct"
    assert env_dict["NHX_MODEL_ENTITY_WORKSPACE"] == DEFAULT_WORKSPACE
    assert env_dict["NHX_MODEL_ENTITY_NAME"] == "integ-k8s-priority-model"
    assert env_dict["NIM_TOOL_CALL_PARSER"] == "deployment-parser"
    assert env_dict["NIM_TOOL_PARSER_PLUGIN"] == _TOOL_CALL_PLUGIN_PATH
    assert env_dict["NIM_ENABLE_AUTO_TOOL_CHOICE"] == "1"
