# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor agent that answers a shard of dataset rows with one model, for metrics scored by the task's verifier.

A dataset evaluation runs as Harbor tasks, one per shard of rows. The task carries the rows in its instruction, and its
verifier scores the answers with the task's own metric code inside the sandbox. This agent is the trusted part in the
middle: it calls the model for every row and leaves the responses in the sandbox for the verifier. Nothing in it is
specific to a dataset or a metric.

It is an *external* agent. Harbor runs :meth:`DatasetRunnerAgent.run` in its own process, so the model endpoint and key
stay there and never enter the sandbox. The only thing it puts in the sandbox is the responses file.

The instruction is a JSON document::

    {
      "schema": "nemo.dataset-shard/v1",
      "rows": [
        {"id": "reasoning/0", "messages": [{"role": "user", "content": "..."}], "params": {"max_tokens": 256}}
      ]
    }

``messages`` and ``params`` form one OpenAI chat-completions request, to which the agent adds only ``model``. Each row
becomes one line of ``/logs/artifacts/responses.jsonl``::

    {"id": "reasoning/0", "status": "ok", "response": {...}, "attempts": 1, "latency_ms": 812, "error": null}

``status`` is ``ok`` with the provider's chat completion in ``response``, or ``error`` / ``timeout`` with ``response``
null and the reason in ``error``. A failed row never fails the shard: the verifier decides what it is worth.

Select it with ``import_path`` and give it the endpoint through ``env_vars``::

    {
      "kind": "harbor",
      "source": {"import_path": "nhx_evals_sdk.agent_eval.runtimes.harbor.dataset_runner:DatasetRunnerAgent",
                 "model_name": "default/my-model"},
      "env_vars": {"DATASET_RUNNER_BASE_URL": "http://<api>/apis/inference-gateway/v2/workspaces/default/openai/-/v1"},
      "agent_kwargs": {"max_concurrency": 8}
    }
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from httpx import AsyncClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError

#: The instruction's ``schema`` value this agent reads.
SHARD_SCHEMA = "nemo.dataset-shard/v1"
#: Where the responses file goes in the sandbox: Harbor's artifacts directory, which a verifier reads in place.
RESPONSES_PATH = "/logs/artifacts/responses.jsonl"
#: Environment variables the endpoint and key are read from, from the agent's ``env_vars`` / ``env_secrets`` first.
BASE_URL_ENV = "DATASET_RUNNER_BASE_URL"
API_KEY_ENV = "DATASET_RUNNER_API_KEY"
#: Sent when no key is configured: a gateway with platform auth off ignores it, and OpenAI clients require one.
PLACEHOLDER_API_KEY = "not-used"
#: HTTP statuses worth another attempt; any other error status is the row's answer.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class ShardRow(BaseModel):
    """One row: an id and the chat request to send for it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class Shard(BaseModel):
    """The instruction: the rows this task answers."""

    model_config = ConfigDict(extra="forbid")

    schema_: str = Field(alias="schema")
    rows: list[ShardRow] = Field(min_length=1)


def parse_shard(instruction: str) -> Shard:
    """The shard an instruction holds.

    Raises:
        ValueError: The instruction isn't a shard, uses another schema, repeats a row id, or a row's ``params`` sets
            ``model`` or ``messages``, which the agent owns.
    """
    try:
        shard = Shard.model_validate_json(instruction)
    except ValidationError as error:
        raise ValueError(f"the instruction is not a {SHARD_SCHEMA} document: {error}") from error
    if shard.schema_ != SHARD_SCHEMA:
        raise ValueError(f"the instruction's schema is {shard.schema_!r}; this agent reads {SHARD_SCHEMA!r}")
    ids = [row.id for row in shard.rows]
    if len(set(ids)) != len(ids):
        raise ValueError("the instruction repeats a row id")
    for row in shard.rows:
        if owned := {"model", "messages", "stream"} & set(row.params):
            raise ValueError(f"row {row.id!r} sets {sorted(owned)} in params; the agent sets those")
    return shard


class DatasetRunnerAgent(BaseAgent):
    """Answers every row of a shard with one model, a bounded number at a time, and writes the responses file.

    Args:
        logs_dir: Harbor's host-side log directory for this trial.
        model_name: The model to call, sent as the request's ``model``.
        max_concurrency: Rows in flight at once.
        row_timeout_sec: Limit on one request; a row that runs out of attempts on timeouts is ``timeout``.
        row_max_attempts: Attempts per row, counting the first, on a timeout, a connection error or a
            :data:`RETRYABLE_STATUSES` status.
        retry_backoff_sec: Base of the exponential backoff between attempts, with full jitter.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_concurrency: int | str = 8,
        row_timeout_sec: float | str = 600,
        row_max_attempts: int | str = 3,
        retry_backoff_sec: float | str = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        # Harbor's `--ak key=value` hands every value over as a string.
        self.max_concurrency = int(max_concurrency)
        self.row_timeout_sec = float(row_timeout_sec)
        self.row_max_attempts = int(row_max_attempts)
        self.retry_backoff_sec = float(retry_backoff_sec)
        for name in ("max_concurrency", "row_timeout_sec", "row_max_attempts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"DatasetRunnerAgent {name} must be positive; got {getattr(self, name)}")
        if self.retry_backoff_sec < 0:
            raise ValueError(f"DatasetRunnerAgent retry_backoff_sec must not be negative; got {self.retry_backoff_sec}")

    @staticmethod
    def name() -> str:
        return "nemo-dataset-runner"

    def version(self) -> str | None:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the agent runs in Harbor's process."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        shard = parse_shard(instruction)
        if not self.model_name:
            raise ValueError("DatasetRunnerAgent needs a model: set the source's model_name")
        base_url = self._setting(BASE_URL_ENV)
        if not base_url:
            raise ValueError(f"DatasetRunnerAgent needs the endpoint in {BASE_URL_ENV} (env_vars)")
        api_key = self._setting(API_KEY_ENV) or PLACEHOLDER_API_KEY

        started = time.monotonic()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        limits = httpx.Limits(max_connections=self.max_concurrency, max_keepalive_connections=self.max_concurrency)
        async with AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(self.row_timeout_sec),
            limits=limits,
        ) as client:

            async def answer(row: ShardRow) -> dict[str, Any]:
                async with semaphore:
                    return await self._answer(client, row)

            records = await asyncio.gather(*(answer(row) for row in shard.rows))

        await self._upload(records, environment)
        self._fill_context(context, records, time.monotonic() - started)

    def _setting(self, name: str) -> str | None:
        return self._extra_env.get(name) or os.environ.get(name)

    async def _answer(self, client: httpx.AsyncClient, row: ShardRow) -> dict[str, Any]:
        """One row's record: the chat completion, or why there is none."""
        body = {**row.params, "model": self.model_name, "messages": row.messages}
        started = time.monotonic()
        status, error, response = "error", None, None
        attempt = 0
        for attempt in range(1, self.row_max_attempts + 1):
            try:
                reply = await client.post("/chat/completions", json=body)
            except httpx.TimeoutException as exc:
                status, error = "timeout", f"no response within {self.row_timeout_sec:g} s ({type(exc).__name__})"
            except httpx.TransportError as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
            else:
                if reply.is_success:
                    try:
                        response = reply.json()
                    except ValueError:
                        status, error = "error", f"HTTP {reply.status_code} with a body that isn't JSON"
                    else:
                        status, error = "ok", None
                        break
                else:
                    status, error = "error", f"HTTP {reply.status_code}: {reply.text[:500]}"
                    if reply.status_code not in RETRYABLE_STATUSES:
                        break
            if attempt < self.row_max_attempts:
                await asyncio.sleep(random.uniform(0, self.retry_backoff_sec * 2 ** (attempt - 1)))
        return {
            "id": row.id,
            "status": status,
            "response": response,
            "attempts": attempt,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "error": error,
        }

    async def _upload(self, records: list[dict[str, Any]], environment: BaseEnvironment) -> None:
        """Write the records as JSONL beside the trial's logs, then copy the file into the sandbox."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.logs_dir, prefix="responses-", suffix=".jsonl", delete=False
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            local = Path(handle.name)
        await environment.exec(f"mkdir -p {Path(RESPONSES_PATH).parent}")
        await environment.upload_file(local, RESPONSES_PATH)

    @staticmethod
    def _fill_context(context: AgentContext, records: list[dict[str, Any]], elapsed_sec: float) -> None:
        usages = [_usage(record["response"]) for record in records if record["status"] == "ok"]
        context.n_input_tokens = sum(usage.get("prompt_tokens", 0) for usage in usages)
        context.n_output_tokens = sum(usage.get("completion_tokens", 0) for usage in usages)
        statuses = [record["status"] for record in records]
        context.metadata = {
            **(context.metadata or {}),
            "dataset_runner": {
                "rows": len(records),
                "ok": statuses.count("ok"),
                "error": statuses.count("error"),
                "timeout": statuses.count("timeout"),
                "attempts": sum(record["attempts"] for record in records),
                "elapsed_sec": round(elapsed_sec, 3),
            },
        }


def _usage(response: Mapping[str, Any] | None) -> dict[str, int]:
    usage = (response or {}).get("usage") or {}
    return {key: value for key, value in usage.items() if isinstance(value, int)}
