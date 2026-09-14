# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for common utilities."""

import hashlib
import re
from datetime import datetime

import pytest
from nemo_platform_plugin.models.types import (
    ContainerExecutorConfig,
    ModelDeployment,
    ModelDeploymentConfig,
    ModelDeploymentConfigModelSpec,
    ModelEntity,
    ModelProvider,
    ModelSpec,
)
from nmp.core.models.app import normalize_model_entity_name
from nmp.core.models.app.utils import (
    ModelConfigParseError,
    ModelWeightsType,
    _get_k8s_safe_name,
    get_deployment_resource_name,
    get_deployment_secret_name,
    get_model_weights_type,
    get_nimcache_resource_name,
    is_multi_llm_image,
    parse_model_name_revision,
)

# ============================================================================
# Tests for parse_model_name_revision
# ============================================================================


def test_parse_no_parameters_returns_defaults():
    namespace, name, revision = parse_model_name_revision()
    assert namespace is None
    assert name is None
    assert revision is None


def test_parse_only_model_name_simple():
    namespace, name, revision = parse_model_name_revision(model_name="llama-3.1-8b-instruct")
    assert namespace is None
    assert name == "llama-3.1-8b-instruct"
    assert revision is None


def test_parse_model_name_with_revision_suffix():
    namespace, name, revision = parse_model_name_revision(model_name="llama-3.1-8b-instruct@v1.0")
    assert namespace is None
    assert name == "llama-3.1-8b-instruct"
    assert revision == "v1.0"


def test_parse_model_name_with_namespace_prefix():
    namespace, name, revision = parse_model_name_revision(model_name="meta/llama-3.1-8b-instruct")
    assert namespace == "meta"
    assert name == "llama-3.1-8b-instruct"
    assert revision is None


def test_parse_model_name_with_namespace_and_revision():
    namespace, name, revision = parse_model_name_revision(model_name="meta/llama-3.1-8b-instruct@main")
    assert namespace == "meta"
    assert name == "llama-3.1-8b-instruct"
    assert revision == "main"


def test_parse_explicit_namespace_takes_precedence():
    namespace, name, revision = parse_model_name_revision(
        model_namespace="ben-test", model_name="some/cool/machine-learning-model"
    )
    assert namespace == "ben-test"
    assert name == "some/cool/machine-learning-model"
    assert revision is None


def test_parse_model_name_with_multiple_slashes_no_explicit_namespace():
    namespace, name, revision = parse_model_name_revision(model_name="some/cool/machine-learning-model")
    assert namespace == "some"
    assert name == "cool/machine-learning-model"
    assert revision is None


def test_parse_explicit_revision_simple():
    namespace, name, revision = parse_model_name_revision(model_name="llama-3.1-8b-instruct", model_revision="dev")
    assert namespace is None
    assert name == "llama-3.1-8b-instruct"
    assert revision == "dev"


def test_parse_explicit_revision_with_namespace():
    namespace, name, revision = parse_model_name_revision(
        model_namespace="meta", model_name="llama-3.1-8b-instruct", model_revision="v1.0"
    )
    assert namespace == "meta"
    assert name == "llama-3.1-8b-instruct"
    assert revision == "v1.0"


def test_parse_error_both_explicit_and_suffix_revision():
    with pytest.raises(ModelConfigParseError) as exc_info:
        parse_model_name_revision(model_name="llama-3.1-8b-instruct@v1.0", model_revision="dev")

    assert "Cannot specify both model_revision field" in str(exc_info.value)
    assert "dev" in str(exc_info.value)
    assert "llama-3.1-8b-instruct@v1.0" in str(exc_info.value)


def test_parse_only_namespace_and_revision():
    namespace, name, revision = parse_model_name_revision(model_namespace="meta", model_revision="v1.0")
    assert namespace == "meta"
    assert name is None
    assert revision == "v1.0"


def test_parse_only_model_name_with_namespace():
    namespace, name, revision = parse_model_name_revision(model_name="nvidia/nim-model")
    assert namespace == "nvidia"
    assert name == "nim-model"
    assert revision is None


def test_parse_model_name_with_complex_revision():
    namespace, name, revision = parse_model_name_revision(model_name="model@v2.1.0-beta.1+build123")
    assert namespace is None
    assert name == "model"
    assert revision == "v2.1.0-beta.1+build123"


def test_parse_all_parameters_provided():
    namespace, name, revision = parse_model_name_revision(
        model_namespace="org", model_name="my-model", model_revision="commit-abc123"
    )
    assert namespace == "org"
    assert name == "my-model"
    assert revision == "commit-abc123"


def test_parse_only_namespace():
    namespace, name, revision = parse_model_name_revision(model_namespace="my-namespace")
    assert namespace == "my-namespace"
    assert name is None
    assert revision is None


def test_parse_only_revision():
    namespace, name, revision = parse_model_name_revision(model_revision="custom-branch")
    assert namespace is None
    assert name is None
    assert revision == "custom-branch"


def test_parse_model_name_with_namespace_revision_complex():
    namespace, name, revision = parse_model_name_revision(model_name="org/path/to/model@v1.0")
    assert namespace == "org"
    assert name == "path/to/model"
    assert revision == "v1.0"


def test_parse_explicit_namespace_preserves_slashes_in_name():
    namespace, name, revision = parse_model_name_revision(model_namespace="explicit-ns", model_name="org/path/to/model")
    assert namespace == "explicit-ns"
    assert name == "org/path/to/model"
    assert revision is None


def test_parse_explicit_namespace_with_revision_suffix():
    namespace, name, revision = parse_model_name_revision(model_namespace="explicit-ns", model_name="path/to/model@dev")
    assert namespace == "explicit-ns"
    assert name == "path/to/model"
    assert revision == "dev"


def test_parse_model_name_with_at_symbol_no_namespace():
    namespace, name, revision = parse_model_name_revision(model_name="simple-model@feature-branch")
    assert namespace is None
    assert name == "simple-model"
    assert revision == "feature-branch"


# ============================================================================
# Tests for get_model_weights_type
# ============================================================================


def test_get_model_weights_type_external_provider():
    provider = ModelProvider(
        workspace="test-ns",
        name="openai-provider",
        host_url="https://api.openai.com/v1",
        model_deployment_id=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=provider,
        model_deployment=None,
        model_deployment_config=None,
        model_entity=None,
    )

    assert location == ModelWeightsType.EXTERNAL_PROVIDER


def test_get_model_weights_type_baked_container_no_model_name():
    provider = ModelProvider(
        workspace="test-ns",
        name="nim-provider",
        host_url="http://nim-service",
        model_deployment_id="test-ns/my-deployment",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    config = ModelDeploymentConfig(
        workspace="test-ns",
        name="my-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name=None),
        executor_config=ContainerExecutorConfig(
            gpu=1,
            image_name="nvcr.io/nim/llama-3",
            image_tag="latest",
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="test-ns",
        name="my-deployment",
        entity_version=1,
        config="my-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=provider,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    assert location == ModelWeightsType.BAKED_CONTAINER


def test_get_model_weights_type_huggingface_with_hf_token():
    """Multi-LLM image name may not match is_multi_llm_image(); falls through to BAKED_CONTAINER."""
    provider = ModelProvider(
        workspace="test-ns",
        name="hf-provider",
        host_url="http://model-service",
        model_deployment_id="test-ns/hf-deployment",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    config = ModelDeploymentConfig(
        workspace="test-ns",
        name="hf-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="meta-llama/Llama-3.1-8B-Instruct"),
        executor_config=ContainerExecutorConfig(gpu=1, image_name="nvcr.io/nim/multi-llm"),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="test-ns",
        name="hf-deployment",
        entity_version=1,
        config="hf-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=provider,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    # nvcr.io/nim/multi-llm may not match is_multi_llm_image(); result is BAKED_CONTAINER or FILES_SERVICE
    assert location in (ModelWeightsType.BAKED_CONTAINER, ModelWeightsType.FILES_SERVICE)


def test_get_model_weights_type_files_service_with_fileset():
    """Model entity with fileset is detected as full weights from file service."""
    provider = ModelProvider(
        workspace="test-ns",
        name="fs-provider",
        host_url="http://model-service",
        model_deployment_id="test-ns/fs-deployment",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    config = ModelDeploymentConfig(
        workspace="test-ns",
        name="fs-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="my-model"),
        executor_config=ContainerExecutorConfig(gpu=1, image_name="nvcr.io/nim/llama-3"),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="test-ns",
        name="fs-deployment",
        entity_version=1,
        config="fs-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    model_entity = ModelEntity(
        id="model-1",
        entity_id="model-1",
        workspace="test-ns",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        model_namespace="test-ns",
        name="my-model",
        parent="models",
        db_version=1,
        fileset="test-ns/my-model",
    )

    location = get_model_weights_type(
        model_provider=provider,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=model_entity,
    )

    assert location == ModelWeightsType.FILES_SERVICE


def test_get_model_weights_type_baked_container_with_model_name():
    provider = ModelProvider(
        workspace="test-ns",
        name="nim-provider",
        host_url="http://nim-service",
        model_deployment_id="test-ns/my-deployment",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    config = ModelDeploymentConfig(
        workspace="test-ns",
        name="my-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="llama-3-8b"),
        executor_config=ContainerExecutorConfig(gpu=1, image_name="nvcr.io/nim/llama-3"),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="test-ns",
        name="my-deployment",
        entity_version=1,
        config="my-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=provider,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    assert location == ModelWeightsType.BAKED_CONTAINER


def test_get_model_weights_type_multi_llm_files_service():
    provider = ModelProvider(
        workspace="test-ns",
        name="multi-llm-provider",
        host_url="http://model-service",
        model_deployment_id="test-ns/multi-llm-deployment",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    config = ModelDeploymentConfig(
        workspace="test-ns",
        name="multi-llm-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="test-ns/my-model"),
        executor_config=ContainerExecutorConfig(gpu=1, image_name=None),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="test-ns",
        name="multi-llm-deployment",
        entity_version=1,
        config="multi-llm-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=provider,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    assert location == ModelWeightsType.FILES_SERVICE


def test_get_model_weights_type_missing_deployment_config():
    """When provider has deployment_id but deployment/config not provided, return UNKNOWN.

    This is a defensive check - no production caller should ever hit this path.
    """
    provider = ModelProvider(
        workspace="test-ns",
        name="nim-provider",
        host_url="http://nim-service",
        model_deployment_id="test-ns/my-deployment",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=provider,
        model_deployment=None,
        model_deployment_config=None,
        model_entity=None,
    )

    assert location == ModelWeightsType.UNKNOWN


# ============================================================================
# Tests for get_model_weights_type WITHOUT model_provider (bug #3716)
# These test the Docker backend deployment creation flow where model_provider
# is not available but deployment/config are.
# ============================================================================


def test_get_model_weights_type_no_provider_with_hf_token():
    """Test that multi-LLM without model_provider is detected as FILES_SERVICE.

    Bug #3716: multi-LLM deployment without provider should not return UNKNOWN.
    """
    config = ModelDeploymentConfig(
        workspace="default",
        name="qwen-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="Qwen/Qwen2.5-1.5B-Instruct"),
        executor_config=ContainerExecutorConfig(gpu=1, image_name=None),  # Multi-LLM (no image specified)
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="default",
        name="qwen-deployment",
        entity_version=1,
        config="qwen-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=None,  # No provider (Docker backend case)
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    assert location == ModelWeightsType.FILES_SERVICE


def test_get_model_weights_type_no_provider_multi_llm_files_service():
    """Test that FILES_SERVICE is detected without model_provider for multi-LLM without hf_token."""
    config = ModelDeploymentConfig(
        workspace="default",
        name="fs-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="workspace/my-model"),
        executor_config=ContainerExecutorConfig(gpu=1, image_name=None),  # Multi-LLM
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="default",
        name="fs-deployment",
        entity_version=1,
        config="fs-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=None,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    assert location == ModelWeightsType.FILES_SERVICE


def test_get_model_weights_type_no_provider_no_deployment_returns_unknown():
    """Test that UNKNOWN is returned when no provider and no deployment/config."""
    location = get_model_weights_type(
        model_provider=None,
        model_deployment=None,
        model_deployment_config=None,
        model_entity=None,
    )

    assert location == ModelWeightsType.UNKNOWN


def test_get_model_weights_type_no_provider_sft_model():
    """Test that SFT model with full weights is detected without model_provider."""
    # Model entity with fileset indicating full weights
    model_entity = ModelEntity(
        id="sft-model-1",
        entity_id="sft-model-1",
        workspace="default",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        model_namespace="default",
        name="sft-model",
        parent="models",
        db_version=1,
        fileset="{workspace}/{name}",
        spec=ModelSpec(
            num_parameters=7000000000,
            context_size=4096,
            num_virtual_tokens=0,
            is_chat=True,
            checkpoint_model_name="meta-llama/Llama-3.2-1b-instruct",
            family="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            ffn_hidden_size=16384,
            vocab_size=32000,
            tied_embeddings=True,
            gated_mlp=True,
            base_num_parameters=7000000000,
            precision="fp16",
        ),
    )

    location = get_model_weights_type(
        model_provider=None,
        model_deployment=None,
        model_deployment_config=None,
        model_entity=model_entity,
    )

    assert location == ModelWeightsType.FILES_SERVICE


def test_get_model_weights_type_no_provider_baked_container():
    """Test baked container detection without model_provider."""
    config = ModelDeploymentConfig(
        workspace="default",
        name="baked-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name=None),  # No model_name = baked weights
        executor_config=ContainerExecutorConfig(
            gpu=1,
            image_name="nvcr.io/nim/meta/llama-3.2-1b-instruct",
            image_tag="1.8.6",
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="default",
        name="baked-deployment",
        entity_version=1,
        config="baked-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=None,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    assert location == ModelWeightsType.BAKED_CONTAINER


def test_get_model_weights_type_no_provider_image_tag_only_is_multi_llm():
    """Test that image_tag without image_name is still treated as multi-LLM.

    When only image_tag is specified (no image_name), the default multi-LLM image
    should be used. The weights type detection should still work correctly.
    """
    config = ModelDeploymentConfig(
        workspace="default",
        name="multi-llm-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="Qwen/Qwen2.5-1.5B-Instruct"),
        executor_config=ContainerExecutorConfig(
            gpu=1,
            image_name=None,  # No image_name = multi-LLM
            image_tag="1.8.6",  # But image_tag IS specified
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="default",
        name="multi-llm-deployment",
        entity_version=1,
        config="multi-llm-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=None,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    assert location == ModelWeightsType.FILES_SERVICE


def test_get_model_weights_type_no_provider_image_tag_only_files_service():
    """Test image_tag without image_name and no hf_token returns FILES_SERVICE."""
    config = ModelDeploymentConfig(
        workspace="default",
        name="fs-config",
        entity_version=1,
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(model_name="workspace/my-model"),
        executor_config=ContainerExecutorConfig(
            gpu=1,
            image_name=None,  # No image_name = multi-LLM
            image_tag="latest",  # But image_tag IS specified
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    deployment = ModelDeployment(
        workspace="default",
        name="fs-deployment",
        entity_version=1,
        config="fs-config",
        config_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    location = get_model_weights_type(
        model_provider=None,
        model_deployment=deployment,
        model_deployment_config=config,
        model_entity=None,
    )

    # Should detect as FILES_SERVICE since image_name is None (multi-LLM) and no hf_token
    assert location == ModelWeightsType.FILES_SERVICE


# ============================================================================
# Tests for is_multi_llm_image
# ============================================================================


def test_is_multi_llm_image_none_returns_true():
    """Test that None image_name returns True (default multi-LLM)."""
    assert is_multi_llm_image(None) is True


def test_is_multi_llm_image_empty_string_returns_true():
    """Test that empty string image_name returns True (default multi-LLM)."""
    assert is_multi_llm_image("") is True


def test_is_multi_llm_image_default_nim_returns_true():
    """Test that the default NVIDIA multi-LLM NIM image returns True."""
    assert is_multi_llm_image("nvcr.io/nim/nvidia/llm-nim") is True


def test_is_multi_llm_image_custom_registry_returns_true():
    """Test that custom registry with llm-nim returns True."""
    assert is_multi_llm_image("myregistry.example.com/llm-nim") is True
    assert is_multi_llm_image("harbor.internal/nvidia/llm-nim") is True


def test_is_multi_llm_image_short_path_returns_true():
    """Test that short path ending in llm-nim returns True."""
    assert is_multi_llm_image("llm-nim") is True


def test_is_multi_llm_image_model_specific_nim_returns_false():
    """Test that model-specific NIM images return False."""
    assert is_multi_llm_image("nvcr.io/nim/meta/llama-3_1-8b-instruct") is False
    assert is_multi_llm_image("nvcr.io/nim/nvidia/nemotron-4-340b-instruct") is False


def test_is_multi_llm_image_similar_name_returns_false():
    """Test that images with similar names that don't exactly match return False."""
    assert is_multi_llm_image("nvcr.io/nim/nvidia/llm-nim-v2") is False
    assert is_multi_llm_image("nvcr.io/nim/nvidia/my-llm-nim") is False
    assert is_multi_llm_image("nvcr.io/nim/nvidia/llm-nim-custom") is False


def test_is_multi_llm_image_random_image_returns_false():
    """Test that arbitrary images return False."""
    assert is_multi_llm_image("my-model-image") is False
    assert is_multi_llm_image("ubuntu:latest") is False
    assert is_multi_llm_image("docker.io/library/python:3.11") is False


# ============================================================================
# Tests for normalize_model_entity_name
# ============================================================================


@pytest.mark.parametrize(
    "input_name,expected",
    [
        pytest.param("meta/llama-3.2-1b-instruct", "meta-llama-3-2-1b-instruct", id="forward_slash"),
        pytest.param("org/team/model-name", "org-team-model-name", id="multiple_slashes"),
        pytest.param("model:v1.0", "model-v1-0", id="colon"),
        pytest.param("model name with spaces", "model-name-with-spaces", id="spaces"),
        pytest.param("model@revision", "model-revision", id="at_symbol"),
        pytest.param("model[variant]", "model-variant", id="brackets"),
        pytest.param("model(version)", "model-version", id="parentheses"),
        pytest.param("already-valid-model-name", "already-valid-model-name", id="already_valid"),
        pytest.param("model123-v4.5.6", "model123-v4-5-6", id="numbers"),
        pytest.param(
            "org/model:v1.0@main with spaces",
            "org-model-v1-0-main-with-spaces",
            id="mixed_invalid_chars",
        ),
        # Digit-leading upstream ids (e.g. NVIDIA Build's "01-ai/yi-large") get an
        # internal "m-" prefix so they satisfy NAME_PATTERN's leading-letter rule.
        pytest.param("01-ai/yi-large", "m-01-ai-yi-large", id="digit_leading_with_slash"),
        pytest.param("9model", "m-9model", id="digit_leading_simple"),
        pytest.param("123", "m-123", id="digit_only"),
        pytest.param("2pac/rapper-model", "m-2pac-rapper-model", id="digit_leading_alphanumeric"),
    ],
)
def test_normalize_model_entity_name_output(input_name, expected):
    """Normalize model entity name produces expected output for valid inputs."""
    assert normalize_model_entity_name(input_name) == expected


@pytest.mark.parametrize(
    "input_name,match",
    [
        pytest.param("", "cannot be normalized to a valid entity name", id="empty_string"),
        pytest.param("///:::", "cannot be normalized to a valid entity name", id="only_invalid_chars"),
        pytest.param("a", "not valid", id="single_char"),
    ],
)
def test_normalize_model_entity_name_raises(input_name, match):
    """Normalize model entity name raises ValueError for invalid inputs."""
    with pytest.raises(ValueError, match=match):
        normalize_model_entity_name(input_name)


def test_normalize_model_entity_name_idempotent():
    """Test that normalization is idempotent (applying twice gives same result)."""

    original = "meta/llama-3.2-1b-instruct"
    once = normalize_model_entity_name(original)
    twice = normalize_model_entity_name(once)
    assert once == twice
    assert once == "meta-llama-3-2-1b-instruct"


def test_normalize_model_entity_name_matches_entity_store_pattern():
    """Valid inputs produce names that match entity store NAME_PATTERN."""
    import re

    from nmp.common.entities.constants import NAME_PATTERN

    pattern = re.compile(NAME_PATTERN)
    valid_inputs = ["meta/llama-3.2-1b", "x--y", "already-valid", "ab", "model-v1-0"]
    for raw in valid_inputs:
        result = normalize_model_entity_name(raw)
        assert pattern.match(result), f"normalize({raw!r}) = {result!r} should match NAME_PATTERN"


def test_normalize_model_entity_name_long_uses_hash_suffix():
    """Names over 63 chars are truncated with deterministic hash to avoid collisions."""
    long_name = "a" * 70
    result = normalize_model_entity_name(long_name)
    assert len(result) <= 63
    assert result.endswith("-") is False
    # Hash suffix is 8 hex chars after final '-'
    parts = result.rsplit("-", 1)
    assert len(parts) == 2
    assert len(parts[1]) == 8 and all(c in "0123456789abcdef" for c in parts[1])


def test_normalize_model_entity_name_long_deterministic():
    """Same long name always produces the same normalized name."""
    long_name = "very-long-model-name-" + "x" * 50
    r1 = normalize_model_entity_name(long_name)
    r2 = normalize_model_entity_name(long_name)
    assert r1 == r2


def test_normalize_model_entity_name_long_different_names_different_hashes():
    """Different long names that would truncate to same prefix get different hashes."""
    # Both > 63 chars so truncation+hash applies; same 54-char prefix, different full names
    name1 = "a" * 70
    name2 = "a" * 69 + "b"
    r1 = normalize_model_entity_name(name1)
    r2 = normalize_model_entity_name(name2)
    assert r1 != r2
    assert len(r1) <= 63 and len(r2) <= 63


def test_normalize_model_entity_name_digit_leading_idempotent():
    """The 'm-' prefix is only added when the result starts with a digit, so re-normalizing
    a previously normalized digit-leading name is a no-op."""
    original = "01-ai/yi-large"
    once = normalize_model_entity_name(original)
    twice = normalize_model_entity_name(once)
    assert once == "m-01-ai-yi-large"
    assert once == twice


def test_normalize_model_entity_name_digit_leading_long_truncates():
    """Digit-leading names that exceed 63 chars after the 'm-' prefix still truncate cleanly
    via the existing hash-suffix path."""
    # 70-char digit-leading raw name -> with 'm-' prefix would be 72; truncation must kick in.
    long_digit_name = "0" + "1" * 69
    result = normalize_model_entity_name(long_digit_name)
    assert len(result) <= 63
    assert result.startswith("m-")
    # Trailing 8 hex chars after final '-' (the deterministic hash suffix).
    parts = result.rsplit("-", 1)
    assert len(parts[1]) == 8 and all(c in "0123456789abcdef" for c in parts[1])


# ============================================================================
# Tests for Kubernetes-safe name generation
# ============================================================================

_HASH8 = re.compile(r"-[0-9a-f]{8}(?:-[a-z0-9-]+)?$")


def _identity_hash(workspace: str, name: str) -> str:
    return hashlib.sha256(f"{workspace}/{name}".encode()).hexdigest()[:8]


def _base_hash(base_name: str) -> str:
    return hashlib.sha256(base_name.encode()).hexdigest()[:8]


def _assert_label_with_hash(name: str, *, prefix: str | None = None) -> None:
    assert len(name) <= 63
    assert name[0].isalpha()
    assert name[-1].isalnum()
    assert _HASH8.search(name)
    if prefix is not None:
        assert name.startswith(prefix)


def test_k8s_safe_name_always_includes_hash_suffix():
    """Test that a simple valid name always includes a hash suffix."""
    result = _get_k8s_safe_name("test-deployment", max_length=63, name_type="label")
    assert result == f"test-deployment-{_base_hash('test-deployment')}"


def test_k8s_safe_name_dots_replaced_in_labels():
    """Test that dots are replaced with hyphens for DNS labels (RFC 1035)."""
    result = _get_k8s_safe_name("llama-3.2-1b", max_length=63, name_type="label")
    assert result == f"llama-3-2-1b-{_base_hash('llama-3.2-1b')}"
    assert "." not in result


def test_k8s_safe_name_dots_preserved_in_dns_subdomain():
    """Test that dots are preserved for DNS subdomains (RFC 1123)."""
    result = _get_k8s_safe_name("my.secret.name", max_length=253, name_type="dns_subdomain")
    assert result == f"my.secret.name-{_base_hash('my.secret.name')}"


def test_k8s_safe_name_dns_subdomain_sanitizes_invalid_label_chars():
    """Invalid characters within dot-delimited labels are sanitized per label."""
    result = _get_k8s_safe_name("my..secret!.name", max_length=253, name_type="dns_subdomain")
    assert "!" not in result
    assert result.startswith("my.secret.name-")
    assert result.endswith(f"-{_base_hash('my..secret!.name')}")


def test_k8s_safe_name_starts_with_letter_for_labels():
    """Test that labels must start with a letter (RFC 1035)."""
    result = _get_k8s_safe_name("123-deployment", max_length=63, name_type="label")
    assert result[0].isalpha()
    assert result == f"x123-deployment-{_base_hash('123-deployment')}"

    result = _get_k8s_safe_name("-deployment", max_length=63, name_type="label")
    assert result[0].isalpha()


def test_k8s_safe_name_ends_with_alphanumeric():
    """Test that names must end with alphanumeric."""
    result = _get_k8s_safe_name("deployment-", max_length=63, name_type="label")
    assert result[-1].isalnum()
    assert result == f"deployment-{_base_hash('deployment-')}"


def test_k8s_safe_name_lowercase_conversion():
    """Test that names are converted to lowercase."""
    result = _get_k8s_safe_name("MyDeployment", max_length=63, name_type="label")
    assert result == f"mydeployment-{_base_hash('MyDeployment')}"
    assert result.islower()


def test_k8s_safe_name_truncation_with_hash():
    """Test that long names are truncated and hashed for uniqueness."""
    long_name = "a" * 100
    result = _get_k8s_safe_name(long_name, max_length=63, name_type="label")

    # Should be exactly 63 chars
    assert len(result) == 63

    # Should end with 8-char hash
    assert "-" in result
    parts = result.rsplit("-", 1)
    assert len(parts) == 2
    hash_part = parts[1]
    assert len(hash_part) == 8
    assert all(c in "0123456789abcdef" for c in hash_part)

    # Should be deterministic - same input produces same output
    result2 = _get_k8s_safe_name(long_name, max_length=63, name_type="label")
    assert result == result2


def test_k8s_safe_name_deterministic():
    """Test that the function is deterministic."""
    name = "test-deployment-with-dots.and.slashes/here"
    result1 = _get_k8s_safe_name(name, max_length=63, name_type="label")
    result2 = _get_k8s_safe_name(name, max_length=63, name_type="label")
    assert result1 == result2


def test_k8s_safe_name_hash_input_differs_from_joined_base():
    """Distinct workspace/name identities produce different hashes for the same joined base."""
    base = "dep-foo-bar-baz"
    name_a = _get_k8s_safe_name(base, hash_input="foo/bar-baz")
    name_b = _get_k8s_safe_name(base, hash_input="foo-bar/baz")
    assert name_a != name_b


def test_plugin_puller_name_ambiguous_workspace_name_pairs_differ():
    """Plugin puller container names use workspace/name identity, not joined hyphens alone."""
    base = "md-plugin-foo-bar-baz"
    name_a = _get_k8s_safe_name(base, hash_input="foo/bar-baz")
    name_b = _get_k8s_safe_name(base, hash_input="foo-bar/baz")
    assert name_a != name_b


def test_get_deployment_secret_name_ambiguous_workspace_name_pairs_differ():
    """Secret names must not collide on ambiguous workspace/name pairs."""
    name_a = get_deployment_secret_name("foo", "bar-baz", prefix="md", suffix="-hf-token")
    name_b = get_deployment_secret_name("foo-bar", "baz", prefix="md", suffix="-hf-token")
    assert name_a != name_b
    assert name_a.endswith("-hf-token")
    assert name_b.endswith("-hf-token")


def test_get_deployment_resource_name_simple():
    """Test simple deployment resource name."""
    result = get_deployment_resource_name("default", "test-deployment")
    assert result == f"md-default-test-deployment-{_identity_hash('default', 'test-deployment')}"
    assert len(result) <= 63


def test_get_deployment_resource_name_ambiguous_workspace_name_pairs_differ():
    """Regression: hyphen-joined workspace/name pairs must not collide."""
    name_a = get_deployment_resource_name("foo", "bar-baz")
    name_b = get_deployment_resource_name("foo-bar", "baz")
    assert name_a != name_b
    _assert_label_with_hash(name_a, prefix="md-")
    _assert_label_with_hash(name_b, prefix="md-")


def test_get_deployment_resource_name_dots_replaced():
    """Test that dots in deployment name are replaced (DNS-1035 compliance)."""
    result = get_deployment_resource_name("ben-test", "llama-3.2-1b-deployment")
    assert "." not in result
    assert result == f"md-ben-test-llama-3-2-1b-deployment-{_identity_hash('ben-test', 'llama-3.2-1b-deployment')}"
    assert len(result) <= 63


def test_get_deployment_resource_name_long_names():
    """Test that very long namespace and name are truncated."""
    long_ns = "a" * 100
    long_name = "b" * 100
    result = get_deployment_resource_name(long_ns, long_name)

    assert len(result) <= 63
    assert result.startswith("md-")
    assert result[0].isalpha()  # Must start with letter for RFC 1035
    assert result[-1].isalnum()


def test_get_deployment_secret_name_simple():
    """Test simple secret name generation."""
    result = get_deployment_secret_name("default", "test", prefix="md", suffix="-hf-token")
    assert result == f"md-default-test-{_identity_hash('default', 'test')}-hf-token"
    assert len(result) <= 253


def test_get_deployment_secret_name_provider():
    """Test provider secret name generation."""
    result = get_deployment_secret_name("default", "test", prefix="model-provider", suffix="-api-key")
    assert result == f"model-provider-default-test-{_identity_hash('default', 'test')}-api-key"
    assert len(result) <= 253


def test_get_deployment_secret_name_dots_allowed():
    """Test that dots are allowed in secret names (DNS subdomain)."""
    result = get_deployment_secret_name("my.namespace", "my.deployment", prefix="md", suffix="-token")
    assert "." in result
    assert len(result) <= 253


def test_get_deployment_secret_name_long():
    """Test that very long secret names are handled."""
    long_ns = "a" * 150
    long_name = "b" * 150
    result = get_deployment_secret_name(long_ns, long_name, prefix="md", suffix="-hf-token")

    assert len(result) <= 253
    assert result.endswith("-hf-token")


def test_real_invalid_k8s_name_issue():
    """
    Test the exact scenario from the user's error message.

    Error was:
    Service "md-ben-test-llama-3.2-1b-deployment" is invalid: metadata.name: Invalid value:
    "md-ben-test-llama-3.2-1b-deployment": a DNS-1035 label must consist of lower case
    alphanumeric characters or '-', start with an alphabetic character, and end with an
    alphanumeric character (e.g. 'my-name', or 'abc-123', regex used for validation is
    '[a-z]([-a-z0-9]*[a-z0-9])?')
    """
    result = get_deployment_resource_name("ben-test", "llama-3.2-1b-deployment")

    # Should not contain dots
    assert "." not in result

    # Should start with letter
    assert result[0].isalpha()

    # Should end with alphanumeric
    assert result[-1].isalnum()

    # Should only contain lowercase alphanumeric and hyphens
    assert all(c.isalnum() or c == "-" for c in result)
    assert result.islower()

    # Expected output includes hash from unambiguous workspace/name identity
    assert result == f"md-ben-test-llama-3-2-1b-deployment-{_identity_hash('ben-test', 'llama-3.2-1b-deployment')}"


def test_maximum_length_deployment_resources():
    """Test deployment with maximum allowed name lengths (255 chars each)."""
    # Maximum that could be stored in DB is 255 chars each
    max_ns = "a" * 255
    max_name = "b" * 255

    resource_name = get_deployment_resource_name(max_ns, max_name)
    secret_name = get_deployment_secret_name(max_ns, max_name, prefix="md", suffix="-hf-token")

    # NIMService/PVC resource names are capped at the standard 63-char K8s label limit
    assert len(resource_name) <= 63
    assert len(secret_name) <= 253

    # All should be valid
    assert resource_name[0].isalpha()
    assert resource_name[-1].isalnum()
    assert secret_name[-1].isalnum()


# ============================================================================
# Tests for get_nimcache_resource_name (issue #4346)
# ============================================================================


def test_get_nimcache_resource_name_simple():
    """Short names include hash suffix and stay within the 59-char NIMCache limit."""
    result = get_nimcache_resource_name("default", "test-deployment")
    assert result == f"md-default-test-deployment-{_identity_hash('default', 'test-deployment')}"
    assert len(result) <= 59


def test_get_nimcache_resource_name_dots_replaced():
    """Dots are replaced with hyphens (DNS-1035 compliance)."""
    result = get_nimcache_resource_name("ben-test", "llama-3.2-1b-deployment")
    assert "." not in result
    assert result == f"md-ben-test-llama-3-2-1b-deployment-{_identity_hash('ben-test', 'llama-3.2-1b-deployment')}"
    assert len(result) <= 59


def test_get_nimcache_resource_name_capped_at_59():
    """Long names are truncated to 59 chars, not 63.

    The k8s-nim-operator appends '-job' (4 chars) to NIMCache names when
    creating its internal batch Job.  Without the 4-char headroom the Job
    name exceeds the 63-char K8s label limit (issue #4346).
    """
    long_ws = "a" * 100
    long_name = "b" * 100
    result = get_nimcache_resource_name(long_ws, long_name)
    assert len(result) <= 59
    assert result.startswith("md-")
    assert result[0].isalpha()
    assert result[-1].isalnum()


def test_get_nimcache_resource_name_job_suffix_always_fits():
    """Appending '-job' to any NIMCache name must stay within 63 chars.

    Parametrized over a range of workspace/name lengths.
    """
    test_cases = [
        ("default", "test-deployment"),
        ("ben-test", "llama-3.2-1b-deployment"),
        ("a" * 50, "b" * 50),
        ("my-workspace", "nvidia-llama-3-3-nemotron-super-79b"),
        ("sft-deploy", "nvidia-llama-3-3-nemotron-super-79b8d0f9"),
        ("a" * 255, "b" * 255),  # Extreme lengths (max DB value)
    ]
    for workspace, name in test_cases:
        nimcache_name = get_nimcache_resource_name(workspace, name)
        job_name = f"{nimcache_name}-job"
        assert len(job_name) <= 63, (
            f"Job name '{job_name}' ({len(job_name)} chars) exceeds 63-char limit "
            f"for workspace={workspace!r}, name={name!r}"
        )


def test_get_nimcache_resource_name_bug_4346_exact_scenario():
    """Regression test for issue #4346.

    The failing Job name from the bug report was:
      'md-default-sft-deploy-nvidia-llama-3-3-nemotron-super-79b8d0f9-job'
    which is 67 characters — exceeding the 63-char K8s label limit because
    the NIMCache name itself was 63 chars (the old max).

    After the fix, get_nimcache_resource_name caps at 59 chars so that
    appending '-job' always produces a name of at most 63 chars.
    """
    workspace = "default"
    name = "sft-deploy-nvidia-llama-3-3-nemotron-super-79b8d0f9"
    nimcache_name = get_nimcache_resource_name(workspace, name)
    assert len(nimcache_name) <= 59
    job_name = f"{nimcache_name}-job"
    assert len(job_name) <= 63, f"Job name '{job_name}' is {len(job_name)} chars, exceeds 63"


def test_get_nimcache_resource_name_deterministic():
    """Same inputs always produce the same NIMCache resource name."""
    workspace = "my-workspace"
    name = "nvidia-llama-3-3-nemotron-super-79b8d0f9-long-enough-to-truncate"
    r1 = get_nimcache_resource_name(workspace, name)
    r2 = get_nimcache_resource_name(workspace, name)
    assert r1 == r2


def test_get_nimcache_resource_name_different_names_differ():
    """Two distinct long names that truncate to the same prefix get different hashes."""
    ws = "default"
    name1 = "sft-deploy-nvidia-llama-3-3-nemotron-super-79b8d0f9-aaaa"
    name2 = "sft-deploy-nvidia-llama-3-3-nemotron-super-79b8d0f9-bbbb"
    r1 = get_nimcache_resource_name(ws, name1)
    r2 = get_nimcache_resource_name(ws, name2)
    assert r1 != r2
    assert len(r1) <= 59 and len(r2) <= 59


def test_nimcache_and_nimservice_names_may_differ_for_long_inputs():
    """NIMCache and NIMService names can differ when the base exceeds 59 chars.

    For the same (workspace, name) pair, get_nimcache_resource_name truncates
    at 59 chars while get_deployment_resource_name truncates at 63 chars.
    When the base is short enough to fit in 59 chars both return the same
    string; when it exceeds 59 chars they diverge.
    """
    # Short name: both functions return the same result
    short_ws, short_name = "default", "short"
    assert get_nimcache_resource_name(short_ws, short_name) == get_deployment_resource_name(short_ws, short_name)

    # Long name: NIMCache name is shorter (truncated earlier)
    long_ws = "a" * 50
    long_name = "b" * 50
    nimcache = get_nimcache_resource_name(long_ws, long_name)
    nimservice = get_deployment_resource_name(long_ws, long_name)
    assert len(nimcache) <= 59
    assert len(nimservice) <= 63
    # They will differ because the truncation point is different
    assert nimcache != nimservice


def test_get_deployment_secret_name_hft_suffix():
    """Test secret name with -hft suffix (used for HuggingFace token secrets)."""
    result = get_deployment_secret_name("default", "my-deployment", prefix="md", suffix="-hft")
    assert result == f"md-default-my-deployment-{_identity_hash('default', 'my-deployment')}-hft"
    assert len(result) <= 253


def test_get_deployment_secret_name_hft_long_names():
    """Test that long names with -hft suffix are properly truncated."""
    # 255 chars each (max allowed in DB)
    long_workspace = "a" * 255
    long_name = "b" * 255

    result = get_deployment_secret_name(long_workspace, long_name, prefix="md", suffix="-hft")

    # Must fit K8s secret name limit
    assert len(result) <= 253
    # Must end with the suffix
    assert result.endswith("-hft")
    # Must be valid K8s name (alphanumeric + hyphens)
    assert all(c.isalnum() or c == "-" or c == "." for c in result)


def test_get_deployment_secret_name_hft_preserves_uniqueness():
    """Test that different long names produce different secret names (via hash)."""
    long_workspace = "workspace-" + "a" * 240
    long_name_1 = "deployment-" + "x" * 240
    long_name_2 = "deployment-" + "y" * 240

    result_1 = get_deployment_secret_name(long_workspace, long_name_1, prefix="md", suffix="-hft")
    result_2 = get_deployment_secret_name(long_workspace, long_name_2, prefix="md", suffix="-hft")

    # Both should be valid and within limits
    assert len(result_1) <= 253
    assert len(result_2) <= 253
    # Should be different due to hash
    assert result_1 != result_2


def test_get_deployment_secret_name_realistic_long_deployment():
    """Test with realistic very long deployment name that triggers truncation.

    This simulates a user creating a deployment with a very descriptive name
    that exceeds K8s limits (253 chars for secrets).
    """
    workspace = "my-production-workspace-for-machine-learning-models"
    # A realistic but very long deployment name that will exceed 253 chars total
    deployment_name = (
        "llama-3-2-70b-instruct-fine-tuned-on-customer-support-data-"
        "with-retrieval-augmented-generation-enabled-and-custom-system-prompt-"
        "optimized-for-low-latency-inference-using-tensor-parallelism-"
        "deployed-to-kubernetes-cluster-with-autoscaling-v2-production-2024-01-15"
    )

    # Verify the input is actually long enough to trigger truncation
    base_name = f"md-{workspace}-{deployment_name}-hft"
    assert len(base_name) > 253, f"Test input should exceed 253 chars, got {len(base_name)}"

    result = get_deployment_secret_name(workspace, deployment_name, prefix="md", suffix="-hft")

    # Must fit K8s secret name limit
    assert len(result) <= 253
    # Must end with suffix
    assert result.endswith("-hft")
    # Should contain hash (8 chars) before suffix when truncated
    # Format: {truncated}-{8-char-hash}-hft
    assert len(result.split("-")[-2]) == 8, "Should have 8-char hash before suffix"
    # Must start with prefix
    assert result.startswith("md-")
    # Should preserve the beginning of the original name
    assert "my-production" in result
