# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote metric runtime implementation."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, cast

import httpx
from httpx import Timeout
from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JsonPathParserError
from nhx_evals_sdk.inference import requests_log_var
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import (
    TemplateSample,
    build_template_context,
    render_template_or_raise,
    template_metric_repr,
)
from nhx_evals_sdk.resilience.api import run_with_resilience
from nhx_evals_sdk.resilience.classifier import endpoint_identity
from nhx_evals_sdk.resolver_protocols import SecretResolver
from nhx_evals_sdk.values.common import SecretRef
from nhx_evals_sdk.values.metrics import NemoAgentToolkitRemote, Remote, _RemoteBase
from nhx_evals_sdk.values.scores import RemoteScore
from pydantic import Field, SecretStr, field_validator

__all__ = ["RemoteMetric", "NemoAgentToolkitRemoteMetric", "SecretRef"]

_logger = logging.getLogger(__name__)


async def _post_to_remote_endpoint(
    url: str,
    payload: dict[str, Any],
    api_key: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 0,
    log: logging.Logger = _logger,
) -> dict[str, Any]:
    """Make a POST request to the remote endpoint."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    log.debug("Calling remote metric url: %r", url)

    endpoint_key = endpoint_identity(url, model_id="remote-metric", auth_identity=api_key)
    max_attempts = max(1, max_retries + 1)

    async with httpx.AsyncClient(timeout=Timeout(timeout)) as client:

        async def _invoke_post() -> dict[str, Any]:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

        try:
            return await run_with_resilience(endpoint_key, _invoke_post, max_attempts=max_attempts)
        except Exception:
            log.exception("Remote metric request failed after %d attempts", max_attempts)
            raise


class _RemoteMetricBase(_RemoteBase, ABC):
    """Shared runtime lifecycle for metrics backed by remote HTTP endpoints."""

    metric_threshold_score: str | None = Field(default=None)
    _api_key: SecretStr | None = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize private API key from env when configured."""
        if self.api_key_secret:
            env_var_name = self.api_key_secret.root.replace("-", "_")
            self._set_api_key(os.getenv(env_var_name))
        return super().model_post_init(__context)

    def _set_api_key(self, api_key: str | None) -> None:
        """Store API key as SecretStr private attribute."""
        self._api_key = SecretStr(api_key) if api_key else None

    def _get_api_key(self) -> str | None:
        """Read API key value from SecretStr private attribute."""
        return self._api_key.get_secret_value() if self._api_key else None

    def _append_request_log(self, *, payload: dict[str, Any], response: dict[str, Any]) -> None:
        """Append the remote request and response payloads to the shared request log."""
        requests_log = requests_log_var.get([])
        requests_log.append({"request": payload, "response": response})

    async def _post_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one rendered payload to the remote endpoint and log the result."""
        result_data = await _post_to_remote_endpoint(
            url=self.url,
            payload=payload,
            api_key=self._get_api_key(),
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
            log=_logger,
        )
        self._append_request_log(payload=payload, response=result_data)
        return result_data

    async def resolve_secrets(self, secret_resolver: SecretResolver) -> None:
        """Resolve API key secret if configured. Must be called before live evaluation."""
        if self.api_key_secret:
            secret_name = self.api_key_secret.root
            resolved_key = await secret_resolver.resolve_secret(self.api_key_secret)
            if resolved_key:
                self._set_api_key(resolved_key)
            elif not self._get_api_key():
                raise ValueError(
                    f"Missing secret '{secret_name}' for API key authentication with remote metric server."
                )

    def secrets(self) -> dict[str, SecretRef]:
        """Return secret env mappings required by this metric."""
        if self.api_key_secret:
            env_var_name = self.api_key_secret.root.replace("-", "_")
            return {env_var_name: self.api_key_secret}
        return {}

    def _select_metric_score(self, metric_result: MetricResult) -> float:
        """Select the default score value for one-row metric results."""
        return float(metric_result.outputs[0].value)

    @abstractmethod
    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute structured score output for one item/sample pair."""
        ...


class RemoteMetric(Remote, _RemoteMetricBase):
    """A metric that computes scores via a remote endpoint."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(score.name) for score in self.scores]

    @field_validator("scores")
    @classmethod
    def _validate_scores(cls, scores: list[RemoteScore]) -> list[RemoteScore]:
        for score in scores:
            try:
                jsonpath_parse(score.parser.json_path)
            except JsonPathParserError as error:
                raise ValueError(
                    f"Score '{score.name}' has invalid JSONPath expression '{score.parser.json_path}': {error}"
                ) from error
        return scores

    def _select_metric_score(self, metric_result: MetricResult) -> float:
        """Select the score value used for single-score consumers."""
        if self.metric_threshold_score:
            output_names = [output.name for output in metric_result.outputs]
            if self.metric_threshold_score not in output_names:
                raise ValueError(
                    f"Score name '{self.metric_threshold_score}' not found in remote metric response. "
                    f"Available scores: {output_names}"
                )
            return float(
                next(output for output in metric_result.outputs if output.name == self.metric_threshold_score).value
            )

        if len(metric_result.outputs) == 1:
            return float(metric_result.outputs[0].value)

        raise ValueError(
            f"Remote metric returned multiple scores {[output.name for output in metric_result.outputs]}. "
            "Please set metric_threshold_score to specify which score to use."
        )

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute structured score output via the remote endpoint."""
        item = input.row.data
        sample: TemplateSample = input.candidate
        context = build_template_context(item, sample)
        rendered_args = render_template_or_raise(
            template_name="body",
            template=self.body,
            context=context,
            item=item,
            sample=sample,
            metric_repr=template_metric_repr(self),
        )
        payload = cast(dict[str, Any], rendered_args) if isinstance(rendered_args, dict) else {"args": rendered_args}
        result_data = await self._post_payload(payload)

        try:
            _logger.debug("Remote metric result received for payload: %r", payload)
            outputs: list[MetricOutput] = []
            for score_config in self.scores:
                jsonpath_expr = jsonpath_parse(score_config.parser.json_path)
                matches = jsonpath_expr.find(result_data)
                if not matches:
                    _logger.warning(
                        "Could not extract score %r from path: %r, setting to NaN",
                        score_config.name,
                        score_config.parser.json_path,
                    )
                    outputs.append(MetricOutput(name=score_config.name, value=float("nan")))
                else:
                    score_value = matches[0].value
                    outputs.append(MetricOutput(name=score_config.name, value=float(score_value)))

            return MetricResult(outputs=outputs)
        except Exception:
            _logger.exception("Error validating remote metric response")
            raise


class NemoAgentToolkitRemoteMetric(NemoAgentToolkitRemote, _RemoteMetricBase):
    """A remote metric that interfaces with NeMo Agent Toolkit evaluators."""

    _RESULT_SCORE_JSONPATH = jsonpath_parse("$.result.score")

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(self.evaluator_name)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute structured score output via the NeMo Agent Toolkit evaluator endpoint."""
        item = input.row.data
        sample: TemplateSample = input.candidate
        context = build_template_context(item, sample)
        rendered_item = render_template_or_raise(
            template_name="body.item",
            template="{{ item | tojson }}",
            context=context,
            item=item,
            sample=sample,
            metric_repr=template_metric_repr(self),
        )
        payload = {
            "evaluator_name": self.evaluator_name,
            "item": rendered_item,
        }
        result_data = await self._post_payload(payload)

        matches = self._RESULT_SCORE_JSONPATH.find(result_data)
        if not matches:
            _logger.warning("Could not extract NeMo Agent Toolkit score from response, setting to NaN")
            score = float("nan")
        else:
            score = float(matches[0].value)

        return MetricResult(outputs=[MetricOutput(name=self.evaluator_name, value=score)])
