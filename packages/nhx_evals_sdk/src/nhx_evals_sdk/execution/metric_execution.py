# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric evaluation orchestration for evals SDK runtime."""

# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from collections.abc import Callable, Coroutine, Sequence
from functools import partial
from logging import getLogger
from types import MappingProxyType
from typing import Any, TypeVar, cast, overload
from urllib.parse import urlparse

import httpx
import nhx_evals_sdk.inference as inference
from nhx_evals_sdk.agent_inference import (
    AgentInferenceFn,
    AgentInvocationResult,
    invoke_agent,
    make_agent_inference_request,
    new_agent_inference_client,
)
from nhx_evals_sdk.execution.config import fail_fast_from_params, resolve_params
from nhx_evals_sdk.execution.pipeline import (
    GeneratedSampleEvent,
    GeneratedSampleScoringPipeline,
    PipelineRuntime,
)
from nhx_evals_sdk.execution.samples import build_offline_sample, build_retrieval_sample
from nhx_evals_sdk.execution.scoring import (
    empty_evaluation_result,
    finalize_evaluation_result,
    nan_metric_result,
    score_row,
)
from nhx_evals_sdk.execution.values import EvaluationError, EvaluationPhase
from nhx_evals_sdk.inference import InferenceMetricBase
from nhx_evals_sdk.metrics.protocol import (
    Metric,
    MetricResult,
)
from nhx_evals_sdk.metrics.utils import metric_type_name
from nhx_evals_sdk.resilience.api import run_indexed_tasks, use_resilience_session
from nhx_evals_sdk.resilience.errors import get_evaluation_error
from nhx_evals_sdk.session import begin_evaluation_session
from nhx_evals_sdk.structured_output import (
    InferenceStructuredOutput,
    detect_structured_output_mode,
)
from nhx_evals_sdk.templates import render_request
from nhx_evals_sdk.values import (
    Agent,
    AgentBase,
    EvaluationResult,
    Model,
    RowScore,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nhx_evals_sdk.values.retrieval import Retrieval
from nhx_evals_sdk.values.targets import EvalTarget
from openai import AsyncOpenAI

log = getLogger(__name__)
T = TypeVar("T")
_QUEUE_END = object()
_DATASET_INPUT_FAILURE_HINT = (
    "To prevent failure of evaluation, fix the dataset row or set "
    "params.ignore_request_failure=true to skip invalid rows."
)
_INFERENCE_FAILURE_HINT = (
    "To prevent failure of evaluation from inference request failures, check the model endpoint, "
    "credentials, request timeout, and retry settings, or set params.ignore_request_failure=true "
    "to mark failed rows as NaN."
)


def _has_empty_message_content(row: dict[str, object]) -> bool:
    """Return True when a row includes a chat message with empty string content."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict) and cast(dict[str, object], message).get("content") == "" for message in messages
    )


def _has_empty_prompt(row: dict[str, object]) -> bool:
    """Return True when a row includes an empty prompt string."""
    return row.get("prompt") == ""


def _format_exception_summary(error: Exception) -> str:
    """Return a concise one-line cause summary for user-facing row errors."""
    cause = str(error).strip()
    if not cause and error.__cause__ is not None:
        cause = str(error.__cause__).strip()
    if not cause:
        cause = type(error).__name__
    return " ".join(cause.split())


# ---------------------------------------------------------------------------
# Sync bridge
# ---------------------------------------------------------------------------


def run_sync(awaitable_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async factory from synchronous code.

    This helper uses ``asyncio.run`` directly when there is no running event
    loop. If a loop is already active (for example in notebook environments),
    it runs the coroutine inside a dedicated thread to avoid nested-loop errors.

    Args:
        awaitable_factory: Zero-argument callable that returns a coroutine.

    Returns:
        The resolved coroutine result.

    Raises:
        BaseException: Any exception raised by the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        has_running_loop = False
    else:
        has_running_loop = True

    if not has_running_loop:
        return asyncio.run(awaitable_factory())

    results: list[T] = []
    errors: list[BaseException] = []
    context = contextvars.copy_context()

    def _runner() -> None:
        """Execute the coroutine inside a thread-local event loop.

        Returns:
            ``None``. Results and exceptions are captured in outer-scope lists.
        """
        try:
            # A separate thread gives notebook-style environments a fresh event loop
            # without nesting asyncio.run() inside the caller's running loop.
            results.append(context.run(lambda: asyncio.run(awaitable_factory())))
        except BaseException as exc:  # pragma: no cover - re-raised on caller thread
            errors.append(exc)

    thread = threading.Thread(target=_runner, name="nhx-evals-sdk-sync-runner")
    thread.start()
    thread.join()

    if errors:
        raise errors[0]

    return results[0]


# ---------------------------------------------------------------------------
# Online helpers
# ---------------------------------------------------------------------------


def _is_completions_endpoint(url: str) -> bool:
    """Return whether the configured model URL targets completions rather than chat."""
    path = urlparse(url).path.rstrip("/")
    return path.endswith("/completions") and not path.endswith("/chat/completions")


def _default_online_request_template(row: dict[str, Any], model: Model) -> dict:
    """Pick the request template used for online sample generation if possible to infer from the row."""
    prompt_candidates = ("prompt", "input", "question", "query")
    prompt_candidates_text = ", ".join(prompt_candidates)
    inference_error = (
        "Unable to infer prompt template from row. "
        f"Use a custom prompt_template or provide one of these row fields: {prompt_candidates_text}."
    )

    if _is_completions_endpoint(model.url):
        for field_name in prompt_candidates:
            if field_name in row:
                return {"prompt": f"{{{{item.{field_name}}}}}"}
        raise ValueError(inference_error)

    if "messages" in row:
        return {"messages": "{{item.messages}}"}
    for field_name in prompt_candidates:
        if field_name in row:
            return {"messages": [{"role": "user", "content": f"{{{{item.{field_name}}}}}"}]}
    raise ValueError(inference_error)


def _resolve_online_prompt_template(
    prompt_template: str | dict[str, Any] | None,
    model: Model,
    first_row: dict[str, Any],
) -> str | dict[str, Any]:
    if prompt_template is not None:
        return prompt_template
    resolved = _default_online_request_template(first_row, model)
    log.warning(
        "No prompt_template provided for online evaluation. "
        "Setting prompt_template is required when providing an online model for evaluation. "
        "Making best effort to infer it from the first row.\n"
        "Inferred prompt_template from the first row:\n%s",
        json.dumps(resolved, indent=2),
        extra={"prompt_template": resolved},
    )
    return resolved


def _merge_online_hooks(
    *,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None,
    target: EvalTarget,
    preprocess_hooks: Sequence[inference.PreprocessRequest] | None,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None,
) -> tuple[list[inference.PreprocessRequest], list[inference.PostprocessResponse]]:
    """Build deterministic hook lists for SDK local online generation.

    Online sample generation should only use run-level generation hooks.
    Metric hooks belong to metric.compute_scores(input) and must not
    affect the evaluated-model generation stage.
    """

    # build the default hooks shared by sdk and service.
    # new_hooks() always returns at least the log hook in each list:
    #   preprocess  -> [..., log_hook]
    #   postprocess -> [log_hook, ...]
    built_preprocess_hooks, built_postprocess_hooks = inference.new_hooks(
        params if isinstance(params, RunConfigOnlineModel) else None,
    )
    if not built_preprocess_hooks or not built_postprocess_hooks:
        raise ValueError(
            f"inference.new_hooks() must return at least the log hook in each list. built_preprocess_hooks: {len(built_preprocess_hooks)}, built_postprocess_hooks: {len(built_postprocess_hooks)}"
        )

    built_preprocess_core = built_preprocess_hooks[:-1]
    preprocess_log_hook = built_preprocess_hooks[-1]
    postprocess_log_hook = built_postprocess_hooks[0]
    built_postprocess_tail = built_postprocess_hooks[1:]

    # peels off the two log hooks, then splices in caller-supplied
    # preprocess_hooks / postprocess_hooks in the middle
    return (
        [
            *built_preprocess_core,
            *(preprocess_hooks or ()),
            preprocess_log_hook,
        ],
        [
            postprocess_log_hook,
            *(postprocess_hooks or ()),
            *built_postprocess_tail,
        ],
    )


async def resolve_target_structured_output_mode(
    *,
    preprocess_hooks: Sequence[inference.PreprocessRequest],
    model: Model,
    inference_fn: inference.InferenceFn,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None,
) -> None:
    """Probe the target endpoint and set the generation hook's encoding.

    Only the hook built from ``params.structured_output`` is retuned; a caller that supplies its own
    InferenceStructuredOutput has chosen a mode deliberately.
    """
    if not (isinstance(params, RunConfigOnlineModel) and params.structured_output):
        return

    structured_hook = next(
        (hook for hook in preprocess_hooks if isinstance(hook, InferenceStructuredOutput)),
        None,
    )
    if structured_hook is None:
        return

    mode = await detect_structured_output_mode(
        model=model,
        inference_fn=inference_fn,
        api_key=model.api_key,
    )
    structured_hook.set_mode(mode)
    log.info("Structured output mode selected for target %s: %s", model.name, mode.value)


def _maybe_set_default_max_tokens(
    *,
    request: dict[str, Any],
    params: RunConfigOnlineModel | None,
) -> None:
    """Apply the default max token cap when neither params nor request set one."""
    inference_params = params.inference if isinstance(params, RunConfigOnlineModel) else None
    if inference_params is not None and (
        inference_params.max_tokens is not None or inference_params.max_completion_tokens is not None
    ):
        return
    if "max_tokens" in request or "max_completion_tokens" in request:
        return
    request["max_tokens"] = 4096


def _process_online_response(
    response: dict[str, Any],
    *,
    index: int,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None,
) -> tuple[dict[str, Any], str | None]:
    """Apply response hooks and extract model text output."""
    processed_response = response
    for hook in postprocess_hooks or ():
        processed_response = hook.postprocess(processed_response, id=str(index))
    output_text = inference.process_output(processed_response, hooks=[], id=str(index))
    return processed_response, output_text


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


@overload
async def generate_online_sample(
    *,
    target: Model,
    row: dict[str, Any],
    index: int,
    prompt_template: str | dict[str, Any],
    params: RunConfigOnlineModel | None = None,
    inference_fn: inference.InferenceFn,
    client: AsyncOpenAI | None = None,
    preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    default_headers: dict[str, str] | None = None,
    template_context: dict[str, Any] | None = None,
) -> dict[str, Any]: ...


@overload
async def generate_online_sample(
    *,
    target: Agent,
    row: dict[str, Any],
    index: int,
    prompt_template: str | dict[str, Any],
    params: RunConfigOnline | None = None,
    inference_fn: AgentInferenceFn,
    client: httpx.AsyncClient | None = None,
    preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    default_headers: dict[str, str] | None = None,
    template_context: dict[str, Any] | None = None,
) -> dict[str, Any]: ...


async def generate_online_sample(
    *,
    target: Model | Agent,
    row: dict[str, Any],
    index: int,
    prompt_template: str | dict[str, Any],
    params: RunConfigOnline | RunConfigOnlineModel | None = None,
    inference_fn: inference.InferenceFn | AgentInferenceFn,
    client: AsyncOpenAI | httpx.AsyncClient | None = None,
    preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    default_headers: dict[str, str] | None = None,
    template_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one sample payload using shared request rendering and inference logic.

    Two valid call shapes, pinned by the ``@overload``s above:

    - ``target: Model`` pairs with ``inference_fn: InferenceFn``. Default
      max-token caps are applied and ``default_headers`` is forwarded
      to the inference fn.
    - ``target: Agent`` pairs with ``inference_fn: AgentInferenceFn``.
      ``default_headers`` is forwarded to the inference fn.
    """
    request = render_request(prompt_template, context={**row, "item": row, **(template_context or {})})
    if isinstance(target, Model):
        model_params = params if isinstance(params, RunConfigOnlineModel) else None
        _maybe_set_default_max_tokens(request=request, params=model_params)
    request = inference.preprocess_request(request, list(preprocess_hooks or ()), id=str(index))

    max_retries = params.max_retries if params is not None else 3
    timeout = params.request_timeout if params is not None else None

    # ``InferenceFn`` and ``AgentInferenceFn`` are structurally identical at
    # runtime (both expose only ``__call__``), so ``isinstance`` can't
    # discriminate them. ``target`` is the real discriminator and the
    # overloads statically pin the pairing — ``cast`` just records that.
    if isinstance(target, Model):
        model_fn = cast(inference.InferenceFn, inference_fn)
        response = await model_fn(
            target,
            request,
            max_retries,
            client=cast(AsyncOpenAI | None, client),
            default_headers=default_headers,
            timeout=timeout,
        )
    else:
        agent_fn = cast(AgentInferenceFn, inference_fn)
        response = await agent_fn(
            target,
            request,
            client=cast(httpx.AsyncClient | None, client),
            max_retries=max_retries,
            default_headers=default_headers,
            timeout=timeout,
        )

    if isinstance(response, AgentInvocationResult):
        invocation = response
        response_payload = response.response
    else:
        invocation = None
        response_payload = cast(dict[str, Any], response)
    processed_response, processed_output_text = _process_online_response(
        response_payload,
        index=index,
        postprocess_hooks=postprocess_hooks,
    )
    output_text = processed_output_text
    if invocation is not None and not isinstance(output_text, str):
        output_text = invocation.output_text

    sample: dict[str, Any] = {}
    if output_text:
        sample["output_text"] = output_text
    if processed_response:
        sample["response"] = processed_response
    # Agent runtimes return trajectory information alongside the response; surface
    # it at the top of the sample so metric evaluators can read it without digging
    # through the nested response payload.
    if isinstance(processed_response, dict) and "trajectory" in processed_response:
        sample["trajectory"] = processed_response["trajectory"]
    if invocation is not None:
        sample["invocation_status"] = invocation.status.value
        sample["invocation_metadata"] = invocation.metadata
        if invocation.evidence is not None:
            sample["evidence"] = invocation.evidence
    return sample


async def generate_online_sample_agent(
    *,
    agent: Agent,
    row: dict[str, Any],
    index: int,
    prompt_template: str | dict[str, Any],
    params: RunConfigOnline | None = None,
    preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    agent_inference_fn: AgentInferenceFn | None = None,
    client: httpx.AsyncClient | None = None,
    default_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate one agent sample through the unified online sample helper."""
    return await generate_online_sample(
        target=agent,
        row=row,
        index=index,
        prompt_template=prompt_template,
        params=params,
        inference_fn=agent_inference_fn or invoke_agent,
        client=client,
        preprocess_hooks=preprocess_hooks,
        postprocess_hooks=postprocess_hooks,
        default_headers=default_headers,
    )


# ---------------------------------------------------------------------------
# Concrete pipeline
# ---------------------------------------------------------------------------


class ComputeMetricPipeline:
    """Pipeline configuration for row-based metric execution.

    Used by both online evaluation, where rows are turned into generated samples
    before scoring, and offline evaluation, where rows are scored without model
    inference.

    Overloaded constructors enforce the valid pairings between ``target`` and
    ``inference_fn`` at type-check time:

    - ``target: Agent`` pairs with ``inference_fn: AgentInferenceFn``.
    - ``target: Model`` pairs with ``inference_fn: inference.InferenceFn``.
    - ``target: None`` is offline mode — no inference function is used.

    Attributes:
        rows: Input dataset rows to evaluate.
        parallelism: Maximum row-level worker fanout for the shared pipeline.
            In online mode this limits concurrent sample generation and scorer
            workers; in offline mode it still limits scorer fanout.
        metric: Runtime metric implementation used to score each row.
        target: Optional target used to generate per-row samples before scoring.
            If None, the pipeline runs without inference and starts from the
            offline sample built from the row.
        metric_key: Metric identifier used for `RowScore.metrics` and for
            synthesized NaN/error results.
        prompt_template: Request template used to render online inference
            requests. Required when `target` is set.
        params: Evaluation parameters used for inference requests and failure policy.
        inference_fn: Inference function used for online sample
            generation. Must match ``target`` (see overloads); ``None`` is only
            valid when ``target`` is ``None`` (offline).
        default_headers: Optional default headers passed to online inference
            requests.
        preprocess_hooks: Hooks applied before online inference requests are
            sent.
        postprocess_hooks: Hooks applied after online inference responses are
            received, and also to the offline sample when no target is configured.
    """

    rows: list[dict[str, Any]]
    parallelism: int
    metric: Metric
    target: EvalTarget
    metric_key: str
    prompt_template: str | dict[str, Any] | None
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel
    inference_fn: inference.InferenceFn | AgentInferenceFn | None
    client: AsyncOpenAI | httpx.AsyncClient | None
    default_headers: dict[str, str] | None
    preprocess_hooks: list[inference.PreprocessRequest]
    postprocess_hooks: list[inference.PostprocessResponse]

    @overload
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        parallelism: int,
        metric: Metric,
        target: Agent,
        metric_key: str,
        prompt_template: str | dict[str, Any],
        inference_fn: AgentInferenceFn,
        client: httpx.AsyncClient | None = None,
        params: RunConfigOnline,
        default_headers: dict[str, str] | None = None,
        preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
        postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        parallelism: int,
        metric: Metric,
        target: Model,
        metric_key: str,
        prompt_template: str | dict[str, Any],
        inference_fn: inference.InferenceFn,
        client: AsyncOpenAI | None = None,
        params: RunConfigOnlineModel,
        default_headers: dict[str, str] | None = None,
        preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
        postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        parallelism: int,
        metric: Metric,
        target: None,
        metric_key: str,
        params: RunConfig,
        prompt_template: None = None,
        inference_fn: None = None,
        client: None = None,
        default_headers: None = None,
        preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
        postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    ) -> None: ...

    # Rankings are already resolved by the time the pipeline runs, so this target scores rows
    # without inference: no prompt template, no client, no inference function.
    @overload
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        parallelism: int,
        metric: Metric,
        target: Retrieval,
        metric_key: str,
        params: RunConfigOnlineModel,
        prompt_template: None = None,
        inference_fn: None = None,
        client: None = None,
        default_headers: None = None,
        preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
        postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        parallelism: int,
        metric: Metric,
        target: EvalTarget,
        metric_key: str,
        prompt_template: str | dict[str, Any] | None = None,
        params: RunConfig | RunConfigOnline | RunConfigOnlineModel,
        inference_fn: inference.InferenceFn | AgentInferenceFn | None = None,
        client: AsyncOpenAI | httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
        preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
        postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
    ) -> None:
        self.rows = rows
        self.parallelism = parallelism
        self.metric = metric
        self.target = target
        self.metric_key = metric_key
        self.prompt_template = prompt_template
        self.params = params
        self.inference_fn = inference_fn
        self.client = client
        self.default_headers = default_headers
        self.preprocess_hooks = list(preprocess_hooks) if preprocess_hooks is not None else []
        self.postprocess_hooks = list(postprocess_hooks) if postprocess_hooks is not None else []

    async def generate_sample(self, index: int, row: dict[str, Any]) -> dict[str, Any]:
        """Generate the sample payload for one dataset row, including offline row-derived fields."""
        if self.target is None:
            response = build_offline_sample(row)
            for hook in self.postprocess_hooks or ():
                response = hook.postprocess(response, id=f"{index}")
            return response
        if isinstance(self.target, Retrieval):
            return build_retrieval_sample(row, self.target.rankings)

        if self.prompt_template is None:
            raise ValueError("prompt_template is required for service online evaluation")

        # The actual discriminator is ``self.target`` (Agent vs Model). The
        # overloaded ``__init__`` pins the valid pairings statically, and this
        # guard makes the Agent-side contract fail loudly at runtime if a
        # caller bypasses those types.
        if isinstance(self.target, AgentBase):
            if self.inference_fn is None:
                raise TypeError("expected AgentInferenceFn for Agent target")

            # Safe by the Agent↔AgentInferenceFn overload (see above).
            agent_target = cast(Agent, self.target)
            agent_fn = cast(AgentInferenceFn, self.inference_fn)
            agent_params = self.params if isinstance(self.params, RunConfigOnline) else None
            return await generate_online_sample(
                target=agent_target,
                row=row,
                index=index,
                prompt_template=self.prompt_template,
                inference_fn=agent_fn,
                client=cast(httpx.AsyncClient | None, self.client),
                params=agent_params,
                preprocess_hooks=self.preprocess_hooks,
                postprocess_hooks=self.postprocess_hooks,
                default_headers=self.default_headers,
            )

        model_params = self.params if isinstance(self.params, RunConfigOnlineModel) else None
        model_fn: inference.InferenceFn = (
            # Safe by the Model↔InferenceFn overload (see above).
            cast(inference.InferenceFn, self.inference_fn)
            if self.inference_fn is not None
            else inference.make_inference_request
        )
        return await generate_online_sample(
            target=self.target,
            row=row,
            index=index,
            prompt_template=self.prompt_template,
            inference_fn=model_fn,
            client=cast(AsyncOpenAI | None, self.client),
            params=model_params,
            preprocess_hooks=self.preprocess_hooks,
            postprocess_hooks=self.postprocess_hooks,
            default_headers=self.default_headers,
        )

    def handle_generation_error(
        self,
        index: int,
        row: dict[str, object],
        error: Exception,
        generation_requests: list[dict[str, object]],
    ) -> tuple[int, MetricResult | None, RowScore]:
        """Convert inference failures into NaN rows when the job allows ignoring them."""
        # Prefer row-derived root causes when we can identify them. Generic
        # inference failures fall back to the original exception summary.
        if _has_empty_message_content(row):
            error_message = (
                f"Row {index} has empty message content and failed inference: "
                f"{_format_exception_summary(error)}. {_DATASET_INPUT_FAILURE_HINT}"
            )
        elif _has_empty_prompt(row):
            error_message = (
                f"Row {index} has empty prompt and failed inference: "
                f"{_format_exception_summary(error)}. {_DATASET_INPUT_FAILURE_HINT}"
            )
        else:
            error_message = (
                f"Row {index} failed inference: {_format_exception_summary(error)}. {_INFERENCE_FAILURE_HINT}"
            )

        if fail_fast_from_params(self.params):
            raise EvaluationError(
                index,
                error_message,
                phase=EvaluationPhase.SAMPLE_GENERATION,
                metric_key=self.metric_key,
            ) from error

        log.warning("Inference failed, marking as NaN", extra={"item_index": index, "error": error_message})
        sample = {"output_text": None, "response": {}, "inference_error": error_message}
        nan_result = nan_metric_result(self.metric.output_spec())

        return (
            index,
            nan_result,
            RowScore(
                row_index=index,
                item=row,
                sample=sample,
                metrics={self.metric_key: nan_result.outputs},
                requests=generation_requests,
                metric_errors={self.metric_key: error_message},
            ),
        )

    async def score_row(
        self,
        index: int,
        row: dict[str, object],
        sample: dict[str, object],
        generation_requests: list[dict[str, object]],
    ) -> tuple[int, MetricResult | None, RowScore]:
        """Score one online row using the generated sample payload."""
        return await score_row(
            metric=self.metric,
            metric_key=self.metric_key,
            row=row,
            sample=sample,
            index=index,
            fail_fast=fail_fast_from_params(self.params),
            generation_requests=generation_requests,
            logger=log,
        )


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


async def _generate_pipeline_item(index: int, runtime: PipelineRuntime) -> None:
    """Run the producer half of the shared queue pipeline for one row index."""
    row = runtime.pipeline.rows[index]
    generation_requests: list[dict[str, Any]] = []
    inference.requests_log_var.set(generation_requests)

    try:
        sample = await runtime.pipeline.generate_sample(index, row)
    except Exception as error:
        runtime.results[index] = runtime.pipeline.handle_generation_error(index, row, error, generation_requests)
        return

    await runtime.sample_queue.put(
        GeneratedSampleEvent(
            row_index=index,
            item=MappingProxyType(row),
            sample=MappingProxyType(sample),
            requests_log=generation_requests,
        )
    )


async def _score_pipeline_samples(runtime: PipelineRuntime) -> None:
    """Drain generated samples from the queue and score them until the sentinel is received."""
    while True:
        queued = await runtime.sample_queue.get()
        if queued is _QUEUE_END:
            runtime.sample_queue.task_done()
            return

        if not isinstance(queued, GeneratedSampleEvent):
            raise ValueError(f"Expected GeneratedSampleEvent, got: {type(queued).__name__}")

        event = queued
        try:
            runtime.results[event.row_index] = await runtime.pipeline.score_row(
                event.row_index,
                dict(event.item),
                dict(event.sample),
                event.requests_log,
            )
        finally:
            runtime.sample_queue.task_done()


async def run_generated_sample_scoring_pipeline(
    pipeline: GeneratedSampleScoringPipeline,
) -> list[tuple[int, MetricResult | None, RowScore]]:
    """Run a pipeline object through the shared generated-sample queue flow."""
    if not pipeline.rows:
        return []

    worker_count = min(len(pipeline.rows), max(1, pipeline.parallelism))
    queue_capacity = max(1, worker_count * 2)
    runtime = PipelineRuntime(
        pipeline=pipeline,
        sample_queue=asyncio.Queue(maxsize=queue_capacity),
        results=[None] * len(pipeline.rows),
    )

    async with use_resilience_session():
        async with asyncio.TaskGroup() as tg:
            for _ in range(worker_count):
                tg.create_task(_score_pipeline_samples(runtime))
            try:
                await run_indexed_tasks(
                    list(range(len(pipeline.rows))),
                    partial(_generate_pipeline_item, runtime=runtime),
                    parallelism=worker_count,
                )
            finally:
                for _ in range(worker_count):
                    await runtime.sample_queue.put(_QUEUE_END)

    if any(result is None for result in runtime.results):
        raise RuntimeError("Internal error: missing row evaluation result after online execution")

    return [result for result in runtime.results if result is not None]


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


async def evaluate_metric(
    metric: Metric,
    *,
    rows: list[dict[str, Any]],
    target: EvalTarget = None,
    prompt_template: str | dict[str, Any] | None = None,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = None,
    preprocess_hooks: Sequence[inference.PreprocessRequest] | None = None,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None = None,
) -> EvaluationResult:
    """Generate model outputs for prepared rows and evaluate a prepared metric."""
    if not rows:
        log.warning("No rows found in dataset, returning empty evaluation result")
        return empty_evaluation_result()

    async with begin_evaluation_session():
        return await _evaluate_metric_in_session(
            metric=metric,
            rows=rows,
            target=target,
            prompt_template=prompt_template,
            params=params,
            preprocess_hooks=preprocess_hooks,
            postprocess_hooks=postprocess_hooks,
        )


async def _evaluate_metric_in_session(
    *,
    metric: Metric,
    rows: list[dict[str, Any]],
    target: EvalTarget,
    prompt_template: str | dict[str, Any] | None,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None,
    preprocess_hooks: Sequence[inference.PreprocessRequest] | None,
    postprocess_hooks: Sequence[inference.PostprocessResponse] | None,
) -> EvaluationResult:
    params = resolve_params(params, target)

    client_close_fn = None

    if isinstance(target, Retrieval):
        # Guaranteed by resolve_params, which coerces every config to online-model form for a
        # retrieval target.
        params = cast(RunConfigOnlineModel, params)
        pipeline = ComputeMetricPipeline(
            rows=rows,
            parallelism=params.parallelism,
            metric=metric,
            target=target,
            metric_key=metric_type_name(metric),
            params=params,
            preprocess_hooks=preprocess_hooks,
            postprocess_hooks=postprocess_hooks,
        )
    else:
        merged_preprocess_hooks, merged_postprocess_hooks = _merge_online_hooks(
            params=params,
            target=target,
            preprocess_hooks=preprocess_hooks,
            postprocess_hooks=postprocess_hooks,
        )
        if isinstance(target, Model):
            params = cast(RunConfigOnlineModel, params)
            inference_fn = (
                metric.inference_fn if isinstance(metric, InferenceMetricBase) else inference.make_inference_request
            )
            await resolve_target_structured_output_mode(
                preprocess_hooks=merged_preprocess_hooks,
                model=target,
                inference_fn=inference_fn,
                params=params,
            )
            resolved_prompt_template = _resolve_online_prompt_template(prompt_template, target, rows[0])
            client = inference.new_inference_client(target)
            client_close_fn = client.close
            pipeline = ComputeMetricPipeline(
                rows=rows,
                parallelism=params.parallelism,
                metric=metric,
                target=target,
                metric_key=metric_type_name(metric),
                prompt_template=resolved_prompt_template,
                params=params,
                inference_fn=inference_fn,
                client=client,
                default_headers=None,
                preprocess_hooks=merged_preprocess_hooks,
                postprocess_hooks=merged_postprocess_hooks,
            )
        elif isinstance(target, AgentBase):
            params = cast(RunConfigOnline, params)
            if prompt_template is None:
                raise ValueError("prompt_template is required for agent online evaluation")

            client = new_agent_inference_client()
            client_close_fn = client.aclose

            pipeline = ComputeMetricPipeline(
                rows=rows,
                parallelism=params.parallelism,
                metric=metric,
                target=target,
                metric_key=metric_type_name(metric),
                prompt_template=prompt_template,
                params=params,
                inference_fn=make_agent_inference_request,
                client=client,
                preprocess_hooks=merged_preprocess_hooks,
                postprocess_hooks=merged_postprocess_hooks,
            )
        else:
            pipeline = ComputeMetricPipeline(
                rows=rows,
                parallelism=params.parallelism,
                metric=metric,
                target=None,
                metric_key=metric_type_name(metric),
                params=params,
                preprocess_hooks=merged_preprocess_hooks,
                postprocess_hooks=merged_postprocess_hooks,
            )

    try:
        completed = await run_generated_sample_scoring_pipeline(pipeline)
    except Exception as e:
        evaluation_error = get_evaluation_error(e)
        if isinstance(evaluation_error, EvaluationError) and evaluation_error.__cause__ is not None:
            raise evaluation_error from evaluation_error.__cause__
        raise evaluation_error from e
    finally:
        if client_close_fn:
            await client_close_fn()

    return await finalize_evaluation_result(metric, completed)
