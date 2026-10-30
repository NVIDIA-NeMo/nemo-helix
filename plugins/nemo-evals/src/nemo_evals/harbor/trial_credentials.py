# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give each Harbor trial its own trial token for the Inference Gateway, so the gateway can attribute its calls.

Before a Harbor job runs, each trial it will run gets a trial token (:mod:`nemo_evals.gateway.tokens`) in its agent's
environment, in place of the agent's gateway key. With platform auth off, the token is unsigned. With auth on, it is
a Scoped Access Key minted for the trial on the user's behalf, scoped to inference, and revoked once Harbor won't
run the trial again. If any key can't be minted, the job fails before a trial starts.

The trial's config holds only a ``${NAME}`` template; the token itself is in this process's environment under that
name while the job runs, so it is never written to the job's files, whatever the agent calls its key variable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Callable, Iterable
from datetime import UTC, datetime

import httpx
from harbor.job import Job
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialHookEvent
from nemo_evals.gateway.policy import ModelAccess, ModelPolicy
from nemo_evals.gateway.spans import export_request, trial_root_span
from nemo_evals.gateway.tokens import TOKEN_LIFETIME_SEC, TrialClaims, unsigned_token
from nemo_evals.intake.mapping import session_id_for
from nemo_helix_plugin.auth.access_keys.client import AsyncAccessKeysClient
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateRequest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.oidc import discover_nhx_config
from nemo_helix_plugin.client.tls import client_verify_from_env
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborJobScope, HarborRuntimeConfig

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SEC = 10.0
MAX_CONCURRENT_KEY_REQUESTS = 8
ACCESS_KEY_SCOPE = "inference"
ACCESS_KEY_NAME_PREFIX = "nemo-evals-trial "
MODEL_FREE_AGENTS = frozenset({"oracle", "nop"})
_OPEN = ModelPolicy(model_access=ModelAccess.OPEN)


class TrialCredentialIssuer:
    """Issues each trial of a Harbor job its trial token, and records each trial's root span in Intake."""

    def __init__(
        self,
        client: AsyncNemoClient,
        *,
        run_id: str,
        intake_workspace: str,
        evaluation_name: str | None = None,
        policy: ModelPolicy = _OPEN,
        records_calls: bool = False,
    ) -> None:
        self.records_calls = records_calls
        self._client = client
        self._run_id = run_id
        self._intake_workspace = intake_workspace
        self._evaluation_name = evaluation_name
        self._policy = policy

    def job_scope(self, api_key_envs: list[str]) -> HarborJobScope:
        """The scope a Harbor runner enters around each job, issuing each trial its token in every ``api_key_envs``."""
        return lambda job: self._issued_to(job, api_key_envs)

    @contextlib.asynccontextmanager
    async def _issued_to(self, job: Job, api_key_envs: list[str]) -> AsyncGenerator[None]:
        # The job runs on its own event loop, so the platform client gets a transport bound to it.
        event_hooks = {event: list(hooks) for event, hooks in self._client._client.event_hooks.items()}
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SEC, verify=client_verify_from_env(), event_hooks=event_hooks
        ) as http_client:
            trials = _JobTrials(self._client.with_http_client(http_client), job, self._claims)
            try:
                auth_enabled = (await asyncio.to_thread(discover_nhx_config, self._client.base_url)).auth_enabled
                # Harbor plans its trials in Job.create and keeps the configs it will run in a private attribute.
                await trials.issue(job._remaining_trial_configs, api_key_envs, auth_enabled=auth_enabled)
                job.on_trial_started(trials.started)
                job.on_trial_ended(trials.ended)
                job.on_trial_cancelled(trials.cancelled)
                yield
            finally:
                trials.forget_tokens()
                cleanup = asyncio.ensure_future(trials.finish())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    # The revocations need the transport, which closes when this block exits.
                    await cleanup
                    raise

    def _claims(self, config: TrialConfig) -> TrialClaims:
        return TrialClaims(
            session=session_id_for(self._run_id, config.trial_name),
            intake_workspace=self._intake_workspace,
            evaluation=self._evaluation_name,
            task=_test_case_name(config),
            model_access=self._policy.model_access,
            models=self._policy.models,
        )


class _JobTrials:
    """The tokens issued to one Harbor job's trials, ended as Harbor finishes with each trial."""

    def __init__(self, client: AsyncNemoClient, job: Job, claims: Callable[[TrialConfig], TrialClaims]) -> None:
        self._client = client
        self._claims = claims
        self._retry = job.config.retry
        self._trials: dict[str, TrialClaims] = {}
        self._live_keys: dict[str, str] = {}
        self._cancelled: set[str] = set()
        self._ends: dict[str, int] = {}
        self._started_at: dict[str, datetime] = {}
        self._running: set[str] = set()
        self._token_variables: list[str] = []
        self._key_requests = asyncio.Semaphore(MAX_CONCURRENT_KEY_REQUESTS)

    async def issue(self, configs: Iterable[TrialConfig], api_key_envs: list[str], *, auth_enabled: bool) -> None:
        configs = list(configs)

        async def issue_one(config: TrialConfig) -> str:
            trial = self._claims(config)
            self._trials[config.trial_name] = trial
            if not auth_enabled:
                return unsigned_token(trial)
            async with self._key_requests:
                key = (
                    await AsyncAccessKeysClient.from_client(self._client).create_access_key(
                        body=AccessKeyCreateRequest(
                            name=f"{ACCESS_KEY_NAME_PREFIX}{trial.session}"[:128],
                            description="Inference Gateway key for one evaluation trial; revoked when the trial ends.",
                            expires_in_seconds=TOKEN_LIFETIME_SEC,
                            scope=[ACCESS_KEY_SCOPE],
                            service_attributes=trial.service_attributes(),
                        )
                    )
                ).data()
            self._live_keys[config.trial_name] = key.jti
            return key.token

        results = await asyncio.gather(*(issue_one(config) for config in configs), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError(
                f"could not issue {len(failures)} of {len(configs)} trials their own Inference Gateway key"
            ) from failures[0]
        for config, token in zip(configs, results, strict=True):
            assert isinstance(token, str)
            variable = f"NEMO_TRIAL_KEY_{uuid.uuid4().hex.upper()}"
            os.environ[variable] = token
            self._token_variables.append(variable)
            config.agent = config.agent.model_copy(
                update={"env": {**config.agent.env, **dict.fromkeys(api_key_envs, f"${{{variable}}}")}}
            )

    def forget_tokens(self) -> None:
        """Remove the trials' tokens from this process's environment."""
        for variable in self._token_variables:
            os.environ.pop(variable, None)
        self._token_variables.clear()

    async def started(self, event: TrialHookEvent) -> None:
        name = event.config.trial_name
        if name in self._trials:
            self._started_at.setdefault(name, event.result.started_at or datetime.now(UTC))
            self._running.add(name)

    async def ended(self, event: TrialHookEvent) -> None:
        """Harbor ended an attempt at a trial: record the trial's root span, and revoke its key if it's done."""
        name = event.config.trial_name
        trial = self._trials.get(name)
        if trial is None:
            return
        self._running.discard(name)
        self._ends[name] = self._ends.get(name, 0) + 1
        started_at = self._started_at.setdefault(name, event.result.started_at or datetime.now(UTC))
        await self._record_root_span(
            trial,
            started_at,
            event.result.finished_at or datetime.now(UTC),
            failed=_failed(event.result),
        )
        if name in self._cancelled or not self._will_retry(event):
            await self._revoke(name)

    async def cancelled(self, event: TrialHookEvent) -> None:
        self._cancelled.add(event.config.trial_name)
        await self._revoke(event.config.trial_name)

    def _will_retry(self, event: TrialHookEvent) -> bool:
        """Whether Harbor will run the trial again, by the rule its trial queue applies after each attempt."""
        error = event.result.exception_info
        if error is None:
            return False
        if self._retry.exclude_exceptions and error.exception_type in self._retry.exclude_exceptions:
            return False
        if self._retry.include_exceptions and error.exception_type not in self._retry.include_exceptions:
            return False
        return self._ends[event.config.trial_name] <= self._retry.max_retries

    async def finish(self) -> None:
        """Record a failed root span for each trial the job stopped mid-attempt, then revoke every live key.

        Harbor ends no trial when its process is told to stop, as a cancelled task Job is.
        """
        now = datetime.now(UTC)
        await asyncio.gather(
            *(
                self._record_root_span(self._trials[name], self._started_at[name], now, failed=True)
                for name in sorted(self._running)
            )
        )
        self._running.clear()
        await self.revoke_all()

    async def revoke_all(self) -> None:
        await asyncio.gather(*(self._revoke(name) for name in list(self._live_keys)))

    async def _record_root_span(
        self, trial: TrialClaims, started_at: datetime, ended_at: datetime, *, failed: bool
    ) -> None:
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SEC * 2):
                await AsyncIntakeClient.from_client(self._client).create_otlp_traces(
                    workspace=trial.intake_workspace,
                    content=export_request(
                        [trial_root_span(trial, started_at=started_at, ended_at=ended_at, failed=failed)]
                    ),
                )
        except Exception:
            logger.warning("Could not record the root span of trial %s", trial.session, exc_info=True)

    async def _revoke(self, name: str) -> None:
        jti = self._live_keys.get(name)
        if jti is None:
            return
        try:
            async with self._key_requests, asyncio.timeout(REQUEST_TIMEOUT_SEC):
                await AsyncAccessKeysClient.from_client(self._client).revoke_access_key(jti=jti)
        except Exception:
            logger.warning("Could not revoke a trial's access key %s", jti, exc_info=True)
            return
        self._live_keys.pop(name, None)


def _failed(result: TrialResult) -> bool:
    """Whether the attempt raised; a multi-step trial records the exception on the step that raised it."""
    return result.exception_info is not None or any(
        step.exception_info is not None for step in result.step_results or ()
    )


def _test_case_name(config: TrialConfig) -> str | None:
    try:
        return Task(config.task.path).name if config.task.path is not None else None
    except Exception:
        logger.debug("Could not read the task of trial %s", config.trial_name, exc_info=True)
        return None


def gateway_job_scope(
    config: HarborRuntimeConfig, issuer: TrialCredentialIssuer, api_key_envs: list[str]
) -> HarborJobScope | None:
    """``issuer``'s job scope for a runner of ``config``, or ``None`` when its agent calls no model."""
    agent = config.agent_import_path or config.agent_name or "oracle"
    return None if agent in MODEL_FREE_AGENTS else issuer.job_scope(api_key_envs)
