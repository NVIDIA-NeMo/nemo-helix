# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import AsyncMock

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.inference import requests_log_var
from nhx_evals_sdk.metrics.protocol import MetricOutput, MetricResult
from nhx_evals_sdk.metrics.remote import NemoAgentToolkitRemoteMetric, RemoteMetric, _post_to_remote_endpoint
from nhx_evals_sdk.values.common import SecretRef
from nhx_evals_sdk.values.scores import JSONScoreParser, RemoteScore
from pytest_mock import MockerFixture


class _SecretResolver:
    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self._secrets = secrets or {}

    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        return self._secrets.get(secret_ref.root)


class TestRemoteMetric:
    @pytest.mark.asyncio
    async def test_post_to_remote_endpoint_includes_auth_header(self, mocker: MockerFixture):
        mock_response = mocker.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"ok": True}
        post = mocker.AsyncMock(return_value=mock_response)
        client = mocker.Mock()
        client.post = post
        async_client = mocker.Mock()
        async_client.__aenter__ = mocker.AsyncMock(return_value=client)
        async_client.__aexit__ = mocker.AsyncMock(return_value=None)
        mocker.patch("nhx_evals_sdk.metrics.remote.httpx.AsyncClient", return_value=async_client)

        async def passthrough(_endpoint_key, callback, *, max_attempts):
            assert max_attempts == 3
            return await callback()

        mocker.patch("nhx_evals_sdk.metrics.remote.run_with_resilience", side_effect=passthrough)

        result = await _post_to_remote_endpoint(
            url="https://remote.example.test",
            payload={"input": "hello"},
            api_key="secret-value",
            max_retries=2,
        )

        assert result == {"ok": True}
        assert post.await_args.kwargs["headers"]["Authorization"] == "Bearer secret-value"

    @pytest.mark.asyncio
    async def test_post_to_remote_endpoint_logs_and_reraises_failures(self, mocker: MockerFixture):
        logger = mocker.Mock()
        mocker.patch("nhx_evals_sdk.metrics.remote.httpx.AsyncClient")
        mocker.patch(
            "nhx_evals_sdk.metrics.remote.run_with_resilience",
            side_effect=RuntimeError("boom"),
        )

        with pytest.raises(RuntimeError, match="boom"):
            await _post_to_remote_endpoint("https://remote.example.test", {"input": "hello"}, log=logger)

        logger.exception.assert_called_once()

    def test_invalid_jsonpath_raises(self):
        with pytest.raises(ValueError, match="invalid JSONPath expression"):
            RemoteMetric(
                url="https://remote.example.test",
                body={"input": "{{item.prompt}}"},
                scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.["))],
            )

    def test_score_names_match_declared_scores(self):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )
        assert output_names(metric) == ["quality"]

    @pytest.mark.asyncio
    async def test_compute_scores(self, mocker: MockerFixture):
        mock_post = mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {"quality": 0.75}},
        )
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )

        result = await compute_scores(metric, {"prompt": "hello"}, {})

        mock_post.assert_awaited_once()
        assert result.outputs[0].name == "quality"
        assert result.outputs[0].value == 0.75

    @pytest.mark.asyncio
    async def test_missing_template_key_raises_clear_error(self):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}", "missing": "{{item.answer}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )

        with pytest.raises(ValueError) as exc_info:
            await compute_scores(metric, {}, {})

        assert "could not render its 'body' template for this row" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_extracted_score_returns_nan(self, mocker: MockerFixture):
        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {}},
        )
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )

        result = await compute_scores(metric, {"prompt": "hello"}, {})
        assert math.isnan(result.outputs[0].value)

    @pytest.mark.asyncio
    async def test_compute_scores_passes_rendered_body_dict_to_remote_endpoint(self, mocker: MockerFixture):
        mock_post = mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {"quality": 0.75}},
        )
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )

        await compute_scores(metric, {"prompt": "hello"}, {})

        await_args = mock_post.await_args
        assert await_args is not None
        assert await_args.kwargs["payload"] == {"input": "hello"}

    @pytest.mark.asyncio
    async def test_compute_scores_logs_and_reraises_invalid_score_values(self, mocker: MockerFixture):
        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {"quality": {"nested": "value"}}},
        )
        log_exception = mocker.patch("nhx_evals_sdk.metrics.remote._logger.exception")
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )

        with pytest.raises(TypeError):
            await compute_scores(metric, {"prompt": "hello"}, {})

        log_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_metric_returns_multiple_scores_when_configured(self, mocker: MockerFixture):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[
                RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality")),
                RemoteScore(name="helpfulness", parser=JSONScoreParser(json_path="$.result.helpfulness")),
            ],
        )
        mocker.patch.object(
            RemoteMetric,
            "compute_scores",
            new=AsyncMock(
                return_value=MetricResult(
                    outputs=[
                        MetricOutput(name="quality", value=0.7),
                        MetricOutput(name="helpfulness", value=0.8),
                    ]
                )
            ),
        )
        result = await compute_scores(metric, {"prompt": "hello"}, {})
        assert [score.name for score in result.outputs] == ["quality", "helpfulness"]
        assert [score.value for score in result.outputs] == [0.7, 0.8]

    @pytest.mark.asyncio
    async def test_metric_uses_single_score_when_only_one_score_is_returned(self, mocker: MockerFixture):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )
        mocker.patch.object(
            RemoteMetric,
            "compute_scores",
            new=AsyncMock(return_value=MetricResult(outputs=[MetricOutput(name="quality", value=0.7)])),
        )
        assert (await compute_scores(metric, {"prompt": "hello"}, {})).outputs[0].value == 0.7

    @pytest.mark.asyncio
    async def test_metric_runs_inside_existing_event_loop(self, mocker: MockerFixture):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )
        mocker.patch.object(
            RemoteMetric,
            "compute_scores",
            new=AsyncMock(return_value=MetricResult(outputs=[MetricOutput(name="quality", value=0.7)])),
        )

        assert (await compute_scores(metric, {"prompt": "hello"}, {})).outputs[0].value == 0.7

    @pytest.mark.asyncio
    async def test_resolve_secrets(self, mocker: MockerFixture):
        captured_api_keys: list[str | None] = []

        async def fake_post(*, api_key: str | None = None, **kwargs):
            del kwargs
            captured_api_keys.append(api_key)
            return {"result": {"quality": 1.0}}

        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            side_effect=fake_post,
        )
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
            api_key_secret=SecretRef(root="remote-api-key"),
        )

        await metric.resolve_secrets(_SecretResolver({"remote-api-key": "secret-value"}))
        await compute_scores(metric, {"prompt": "hello"}, {})

        assert captured_api_keys == ["secret-value"]

    @pytest.mark.asyncio
    async def test_resolve_secrets_raises_when_secret_missing(self):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
            api_key_secret=SecretRef(root="remote-api-key"),
        )

        with pytest.raises(ValueError, match="Missing secret 'remote-api-key'"):
            await metric.resolve_secrets(_SecretResolver())

    def test_secrets_returns_expected_mapping(self):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
            api_key_secret=SecretRef(root="remote-api-key"),
        )
        assert metric.secrets() == {"remote_api_key": SecretRef(root="remote-api-key")}

    def test_secrets_returns_empty_mapping_without_secret(self):
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )
        assert metric.secrets() == {}

    @pytest.mark.asyncio
    async def test_model_post_init_loads_api_key_from_env(self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
        captured_api_keys: list[str | None] = []

        async def fake_post(*, api_key: str | None = None, **kwargs):
            del kwargs
            captured_api_keys.append(api_key)
            return {"result": {"quality": 1.0}}

        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            side_effect=fake_post,
        )
        monkeypatch.setenv("remote_api_key", "env-secret")
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
            api_key_secret=SecretRef(root="remote-api-key"),
        )
        assert metric._get_api_key() == "env-secret"

        await compute_scores(metric, {"prompt": "hello"}, {})

        assert captured_api_keys == ["env-secret"]

    @pytest.mark.asyncio
    async def test_compute_scores_appends_request_log(self, mocker: MockerFixture):
        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {"quality": 0.75}},
        )
        metric = RemoteMetric(
            url="https://remote.example.test",
            body={"input": "{{item.prompt}}"},
            scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
        )

        token = requests_log_var.set([])
        try:
            await compute_scores(metric, {"prompt": "hello"}, {})
            assert requests_log_var.get() == [
                {"request": {"input": "hello"}, "response": {"result": {"quality": 0.75}}}
            ]
        finally:
            requests_log_var.reset(token)


class TestNemoAgentToolkitRemoteMetric:
    @pytest.mark.asyncio
    async def test_nemo_agent_toolkit_remote_metric(self, mocker: MockerFixture):
        mock_post = mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {"score": 1.0}},
        )
        metric = NemoAgentToolkitRemoteMetric(
            url="https://remote.example.test",
            evaluator_name="tool-accuracy",
        )

        result = await compute_scores(metric, {"prompt": "hello"}, {})

        await_args = mock_post.await_args
        assert await_args is not None
        payload = await_args.kwargs["payload"]
        assert payload["evaluator_name"] == "tool-accuracy"
        assert payload["item"] == {"prompt": "hello"}
        assert result.outputs[0].name == "tool-accuracy"

    @pytest.mark.asyncio
    async def test_nemo_agent_toolkit_missing_score_returns_nan(self, mocker: MockerFixture):
        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {}},
        )
        metric = NemoAgentToolkitRemoteMetric(
            url="https://remote.example.test",
            evaluator_name="tool-accuracy",
        )

        result = await compute_scores(metric, {"prompt": "hello"}, {})

        assert math.isnan(result.outputs[0].value)

    def test_nemo_agent_toolkit_score_names(self):
        metric = NemoAgentToolkitRemoteMetric(
            url="https://remote.example.test",
            evaluator_name="tool-accuracy",
        )
        assert output_names(metric) == ["tool-accuracy"]

    @pytest.mark.asyncio
    async def test_nemo_agent_toolkit_resolve_secrets(self, mocker: MockerFixture):
        captured_api_keys: list[str | None] = []

        async def fake_post(*, api_key: str | None = None, **kwargs):
            del kwargs
            captured_api_keys.append(api_key)
            return {"result": {"score": 1.0}}

        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            side_effect=fake_post,
        )
        metric = NemoAgentToolkitRemoteMetric(
            url="https://remote.example.test",
            evaluator_name="tool-accuracy",
            api_key_secret=SecretRef(root="remote-api-key"),
        )

        await metric.resolve_secrets(_SecretResolver({"remote-api-key": "secret-value"}))
        await compute_scores(metric, {"prompt": "hello"}, {})

        assert metric._get_api_key() == "secret-value"
        assert captured_api_keys == ["secret-value"]

    @pytest.mark.asyncio
    async def test_nemo_agent_toolkit_model_post_init_loads_api_key_from_env(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ):
        captured_api_keys: list[str | None] = []

        async def fake_post(*, api_key: str | None = None, **kwargs):
            del kwargs
            captured_api_keys.append(api_key)
            return {"result": {"score": 1.0}}

        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            side_effect=fake_post,
        )
        monkeypatch.setenv("remote_api_key", "env-secret")
        metric = NemoAgentToolkitRemoteMetric(
            url="https://remote.example.test",
            evaluator_name="tool-accuracy",
            api_key_secret=SecretRef(root="remote-api-key"),
        )

        await compute_scores(metric, {"prompt": "hello"}, {})

        assert metric._get_api_key() == "env-secret"
        assert captured_api_keys == ["env-secret"]

    @pytest.mark.asyncio
    async def test_nemo_agent_toolkit_compute_scores_appends_request_log(self, mocker: MockerFixture):
        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {"score": 1.0}},
        )
        metric = NemoAgentToolkitRemoteMetric(
            url="https://remote.example.test",
            evaluator_name="tool-accuracy",
        )

        token = requests_log_var.set([])
        try:
            await compute_scores(metric, {"prompt": "hello"}, {})
            assert requests_log_var.get() == [
                {
                    "request": {"evaluator_name": "tool-accuracy", "item": {"prompt": "hello"}},
                    "response": {"result": {"score": 1.0}},
                }
            ]
        finally:
            requests_log_var.reset(token)

    @pytest.mark.asyncio
    async def test_nemo_agent_toolkit_metric_returns_score(self, mocker: MockerFixture):
        metric = NemoAgentToolkitRemoteMetric(
            url="https://remote.example.test",
            evaluator_name="tool-accuracy",
        )
        mocker.patch(
            "nhx_evals_sdk.metrics.remote._post_to_remote_endpoint",
            new_callable=AsyncMock,
            return_value={"result": {"score": 0.9}},
        )

        assert (await compute_scores(metric, {"prompt": "hello"}, {})).outputs[0].value == 0.9
