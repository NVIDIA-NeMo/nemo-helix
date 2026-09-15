# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``--output-format code`` generation against typed clients."""

import pytest
from nemo_platform_ext.cli.core.code_generator import generate_python_code
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_platform_plugin.inference_gateway.types import JsonBody
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import CreateModelDeploymentRequest
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest


def test_generate_python_code_simple_list():
    code = generate_python_code(SecretsClient, "list_secrets", {}, result="list")

    assert "from nemo_platform_plugin.secrets.client import SecretsClient" in code
    assert "client = SecretsClient.from_config()" in code
    assert "response = client.list_secrets()" in code
    assert "for item in response.page().items:" in code
    assert "print(item)" in code


def test_generate_python_code_with_base_url():
    code = generate_python_code(SecretsClient, "list_secrets", {}, base_url="http://test.example.com", result="list")

    assert 'client = SecretsClient(base_url="http://test.example.com")' in code


def test_generate_python_code_with_args_and_entity_result():
    code = generate_python_code(SecretsClient, "get_secret", {"name": "my-secret", "workspace": "default"})

    assert 'response = client.get_secret(name="my-secret", workspace="default")' in code
    assert "print(response.data())" in code


def test_generate_python_code_skips_none_args():
    code = generate_python_code(SecretsClient, "get_secret", {"name": "my-secret", "workspace": None})

    assert 'response = client.get_secret(name="my-secret")' in code


def test_generate_python_code_renders_request_model_and_imports_it():
    body = PlatformSecretCreateRequest(name="hf-token", value="s3cret", description="HF token")
    code = generate_python_code(SecretsClient, "create_secret", {"workspace": "default", "body": body})

    assert "from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest" in code
    assert 'body=PlatformSecretCreateRequest(name="hf-token", description="HF token", value="***")' in code
    assert "s3cret" not in code
    assert code.index("from nemo_platform_plugin.secrets.client") < code.index("client = SecretsClient")


def test_generate_python_code_renders_root_model_body_positionally():
    body = JsonBody({"model": "default/llama", "messages": [{"role": "user", "content": "hi"}]})
    code = generate_python_code(
        InferenceGatewayClient,
        "provider_post",
        {"name": "nvidia", "trailing_uri": "v1/chat/completions", "body": body},
    )

    assert "from nemo_platform_plugin.inference_gateway.types import JsonBody" in code
    assert 'body=JsonBody({"model": "default/llama", "messages": [{"role": "user", "content": "hi"}]})' in code
    compile(code, "<generated>", "exec")


def test_generate_python_code_renders_query_params_dict_and_lists():
    code = generate_python_code(
        SecretsClient,
        "list_secrets",
        {"query_params": {"page": 2, "page_size": 10, "sort": ["name", "-created_at"]}},
        result="list",
    )

    assert 'query_params={"page": 2, "page_size": 10, "sort": ["name", "-created_at"]}' in code


def test_generate_python_code_renders_datetimes_as_iso_strings():
    from datetime import UTC, datetime

    from pydantic import BaseModel

    class Stamped(BaseModel):
        started_at: datetime

    body = Stamped(started_at=datetime(2026, 8, 14, tzinfo=UTC))
    code = generate_python_code(SecretsClient, "create_secret", {"body": body})

    assert 'Stamped(started_at="2026-08-14T00:00:00+00:00")' in code
    assert "datetime.datetime" not in code


def test_generate_python_code_renders_root_models_by_root_value():
    from pydantic import RootModel

    class Payload(RootModel[dict[str, object]]):
        pass

    body = Payload({"schema_version": "v1", "agent": {"name": "b"}})
    code = generate_python_code(SecretsClient, "create_secret", {"body": body})

    assert 'body=Payload({"schema_version": "v1", "agent": {"name": "b"}})' in code


def test_generate_python_code_multiline_for_many_args():
    code = generate_python_code(
        SecretsClient,
        "get_secret",
        {"name": "a" * 50, "workspace": "default", "extra": 1, "more": 2},
    )

    assert "response = client.get_secret(\n" in code
    assert '    workspace="default",' in code


def test_generate_python_code_no_result_block():
    code = generate_python_code(SecretsClient, "delete_secret", {"name": "x"}, result="none")

    assert code.rstrip().endswith('response = client.delete_secret(name="x")')


def test_generate_python_code_binary_result():
    code = generate_python_code(SecretsClient, "download", {"name": "x"}, result="binary")

    assert "with response.stream() as chunks:" in code


def test_generate_python_code_rejects_wait_and_watch_together():
    with pytest.raises(ValueError, match="Only one of wait_config or watch_config"):
        generate_python_code(
            ModelsClient,
            "create_deployment",
            {},
            wait_config={"type": "inference_deployment"},
            wait_options={"timeout": 10},
            watch_config={"type": "inference_deployment"},
            watch_options={"timeout": 10},
        )


def test_generate_python_code_rejects_unknown_lifecycle():
    with pytest.raises(ValueError, match="Unsupported lifecycle config type"):
        generate_python_code(ModelsClient, "create_deployment", {}, wait_config={"type": "bogus"}, wait_options={})


def test_generate_python_code_inference_deployment_wait_requires_timeout():
    with pytest.raises(ValueError, match="requires timeout"):
        generate_python_code(
            ModelsClient,
            "create_deployment",
            {},
            wait_config={"type": "inference_deployment"},
            wait_options={"poll_interval": 3},
        )


def test_generate_python_code_inference_deployment_wait_block():
    body = CreateModelDeploymentRequest(name="dep-a", config="cfg")
    code = generate_python_code(
        ModelsClient,
        "create_deployment",
        {"workspace": "default", "body": body},
        base_url="http://localhost:8080",
        wait_config={"type": "inference_deployment", "resource_label": "deployment"},
        wait_options={"timeout": 120, "poll_interval": 5},
    )

    assert "import time" in code
    assert "from nemo_platform_plugin.client.errors import NemoHTTPError, NemoTransportError, NotFoundError" in code
    assert "from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient" in code
    assert 'resource_name = getattr(response.data(), "name", None) or "dep-a"' in code
    assert "deadline = time.monotonic() + 120" in code
    assert 'client.get_deployment(name=resource_name, workspace="default").data()' in code
    assert "gateway = InferenceGatewayClient.from_client(client)" in code
    assert "gateway.provider_ready(name=provider_name, workspace=provider_workspace)" in code
    assert "time.sleep(min(5, remaining))" in code
    assert "--wait" in code
    # No Stainless artefacts anywhere in the emitted snippet.
    assert "NeMoPlatform" not in code
    assert "from nemo_platform import" not in code


def test_generate_python_code_watch_mode_mentions_watch_flag():
    code = generate_python_code(
        ModelsClient,
        "create_deployment",
        {"workspace": "default"},
        watch_config={"type": "inference_deployment"},
        watch_options={"timeout": 10},
    )

    assert "--watch" in code
    assert "--wait" not in code


def test_generated_snippet_is_valid_python():
    body = CreateModelDeploymentRequest(name="dep-a", config="cfg")
    code = generate_python_code(
        ModelsClient,
        "create_deployment",
        {"workspace": "default", "body": body},
        base_url="http://localhost:8080",
        wait_config={"type": "inference_deployment"},
        wait_options={"timeout": 120},
    )

    compile(code, "<generated>", "exec")
