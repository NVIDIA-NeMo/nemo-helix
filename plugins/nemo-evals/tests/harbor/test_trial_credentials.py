# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each Harbor trial calls the gateway with its own trial token, which names the trial and ends with it."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from harbor.job import DatasetConfig, Job, JobConfig
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import ExceptionInfo, StepResult, TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.utils.env import resolve_env_vars
from nemo_evals.gateway.policy import ModelAccess, ModelPolicy
from nemo_evals.gateway.tokens import TOKEN_LIFETIME_SEC, TrialClaims, TrialTokenReader
from nemo_evals.harbor import trial_credentials
from nemo_evals.harbor.trial_credentials import TrialCredentialIssuer, gateway_job_scope
from nemo_evals.intake.mapping import session_id_for
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.oidc import NHXOIDCConfig
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Status

RUN = "run-1-task-0"
KEY_ENV = "OPENAI_API_KEY"
ACCESS_KEYS = "/apis/auth/v2/access-keys"
OTLP = "/apis/intake/v2/workspaces/evals/ingest/otlp/v1/traces"
STARTED = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


class Platform:
    """Answers the issuer's platform calls, minting a numbered key per request, and keeps every request it saw."""

    def __init__(self, *, auth_enabled: bool) -> None:
        self.auth_enabled = auth_enabled
        self.refuse_mint_number: int | None = None
        self.hold_mint_number: int | None = None
        self.mint_held = asyncio.Event()
        self.revocations_to_refuse = 0
        self.hold_revocations = False
        self.revocation_held = asyncio.Event()
        self.release_revocations = asyncio.Event()
        self.seen: list[httpx.Request] = []
        self._revoked: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        if request.method == "POST" and request.url.path == ACCESS_KEYS:
            number = len(self.minted()) - 1
            if number == self.refuse_mint_number:
                return httpx.Response(503, json={"detail": "unavailable"})
            if number == self.hold_mint_number:
                self.mint_held.set()
                await asyncio.Event().wait()
            return _minted(number)
        if request.method == "DELETE" and request.url.path.startswith(ACCESS_KEYS):
            if self.hold_revocations:
                self.revocation_held.set()
                await self.release_revocations.wait()
            if self.revocations_to_refuse:
                self.revocations_to_refuse -= 1
                return httpx.Response(503, json={"detail": "unavailable"})
            jti = request.url.path.rsplit("/", 1)[-1]
            self._revoked.append(jti)
            return httpx.Response(200, json={"jti": jti, "revoked": True})
        if request.method == "POST" and request.url.path == OTLP:
            return httpx.Response(200, json={})
        return httpx.Response(500, json={"detail": "unexpected"})

    def minted(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.seen if r.method == "POST" and r.url.path == ACCESS_KEYS]

    def revoked(self) -> list[str]:
        return list(self._revoked)

    def root_span_failures(self) -> list[bool]:
        return [
            ExportTraceServiceRequest.FromString(r.content).resource_spans[0].scope_spans[0].spans[0].status.code
            == Status.STATUS_CODE_ERROR
            for r in self.seen
            if r.url.path == OTLP
        ]

    def root_spans(self) -> list[tuple[int, int]]:
        spans = []
        for request in self.seen:
            if request.url.path == OTLP:
                (span,) = ExportTraceServiceRequest.FromString(request.content).resource_spans[0].scope_spans[0].spans
                spans.append((span.start_time_unix_nano, span.end_time_unix_nano))
        return spans


def _jti(number: int) -> str:
    return f"ak_{number:032d}"


def _minted(number: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "jti": _jti(number),
            "principal": "user@example.com",
            "status": "ACTIVE",
            "issuer": "nemo",
            "audiences": ["nemo"],
            "scope": ["inference"],
            "created_at": "2026-10-26T12:00:00Z",
            "token": f"minted-token-{number}",
            "token_type": "Bearer",
        },
    )


@pytest.fixture(params=[False, True], ids=["auth off", "auth on"])
def platform(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Platform:
    platform = Platform(auth_enabled=request.param)
    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(platform), **kwargs)

    monkeypatch.setattr(trial_credentials.httpx, "AsyncClient", client)
    monkeypatch.setattr(
        trial_credentials, "discover_nhx_config", lambda base_url: NHXOIDCConfig(auth_enabled=platform.auth_enabled)
    )
    return platform


@pytest.fixture
def auth_on(platform: Platform) -> Platform:
    if not platform.auth_enabled:
        pytest.skip("only auth-on platforms mint keys")
    return platform


@pytest.fixture
async def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Job:
    """A planned Harbor job: two attempts at one task, each retried once."""
    task = tmp_path / "dataset" / "hello-world"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('[task]\nname = "trial-rec/hello-world"\n')
    (task / "instruction.md").write_text("Say hello.\n")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    monkeypatch.setenv("USERS_KEY", "users-own-key")
    return await Job.create(
        JobConfig(
            job_name="job",
            jobs_dir=tmp_path / "jobs",
            n_attempts=2,
            retry=RetryConfig(max_retries=1),
            agents=[AgentConfig(name="codex", env={KEY_ENV: "${USERS_KEY}", "OTHER": "kept"})],
            datasets=[DatasetConfig(path=tmp_path / "dataset")],
        )
    )


class Hooks:
    """The trial hooks the issuer registers on the job, fired the way Harbor fires them."""

    def __init__(self, job: Job, monkeypatch: pytest.MonkeyPatch) -> None:
        self.started: list[Any] = []
        self.ended: list[Any] = []
        self.cancelled: list[Any] = []
        monkeypatch.setattr(job, "on_trial_started", self.started.append)
        monkeypatch.setattr(job, "on_trial_ended", self.ended.append)
        monkeypatch.setattr(job, "on_trial_cancelled", self.cancelled.append)

    async def start(self, config: TrialConfig, *, minutes: int = 0) -> None:
        result = TrialResult.model_construct(started_at=STARTED.replace(minute=minutes))
        for hook in self.started:
            await hook(_event(TrialEvent.START, config, result))

    async def end(
        self, config: TrialConfig, *, error: str | None = None, minutes: int = 1, steps: list[StepResult] | None = None
    ) -> None:
        exception_info = (
            None
            if error is None
            else ExceptionInfo(exception_type=error, exception_message="", exception_traceback="", occurred_at=STARTED)
        )
        result = TrialResult.model_construct(
            started_at=STARTED.replace(minute=minutes - 1),
            finished_at=STARTED.replace(minute=minutes),
            exception_info=exception_info,
            step_results=steps,
        )
        for hook in self.ended:
            await hook(_event(TrialEvent.END, config, result))

    async def cancel(self, config: TrialConfig) -> None:
        for hook in self.cancelled:
            await hook(_event(TrialEvent.CANCEL, config, TrialResult.model_construct()))


def _event(event: TrialEvent, config: TrialConfig, result: TrialResult) -> TrialHookEvent:
    return TrialHookEvent.model_construct(event=event, task_name="hello-world", config=config, result=result)


@pytest.fixture
def hooks(job: Job, monkeypatch: pytest.MonkeyPatch) -> Hooks:
    return Hooks(job, monkeypatch)


CLOSED = ModelPolicy(model_access=ModelAccess.CLOSED, models=("default/cb-model",))


def _issuer() -> TrialCredentialIssuer:
    return TrialCredentialIssuer(
        AsyncNemoClient(base_url="http://platform.test"),
        run_id=RUN,
        intake_workspace="evals",
        evaluation_name="run-1",
        policy=CLOSED,
    )


def _claims(config: TrialConfig) -> TrialClaims:
    return TrialClaims(
        session=session_id_for(RUN, config.trial_name),
        intake_workspace="evals",
        evaluation="run-1",
        task="trial-rec/hello-world",
        **CLOSED.model_dump(),
    )


async def test_each_trial_runs_with_its_own_token_in_place_of_the_agents_key(platform: Platform, job: Job) -> None:
    configs = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        tokens = [resolve_env_vars(config.agent.env)[KEY_ENV] for config in configs]
        assert len(set(tokens)) == len(configs) == 2
        assert all(config.agent.env["OTHER"] == "kept" for config in configs)
        if platform.auth_enabled:
            assert tokens == ["minted-token-0", "minted-token-1"]
            assert [body["service_attributes"] for body in platform.minted()] == [
                _claims(config).service_attributes() for config in configs
            ]
        else:
            assert [await TrialTokenReader(None).trial(token) for token in tokens] == [_claims(c) for c in configs]
            assert platform.seen == []


async def test_a_token_is_never_written_into_the_trials_config(auth_on: Platform, job: Job) -> None:
    """Harbor saves each trial's config and masks only variables named like secrets; this one isn't."""
    configs = job._remaining_trial_configs

    async with _issuer().job_scope(["MODEL_BEARER"])(job):
        saved = " ".join(config.model_dump_json() for config in configs)
        tokens = [resolve_env_vars(config.agent.env)["MODEL_BEARER"] for config in configs]
        assert tokens == ["minted-token-0", "minted-token-1"]
        assert "minted-token" not in saved

    with pytest.raises(ValueError, match="not found"):
        resolve_env_vars(configs[0].agent.env)


async def test_a_minted_key_is_an_inference_key_that_expires_in_a_day(auth_on: Platform, job: Job) -> None:
    async with _issuer().job_scope([KEY_ENV])(job):
        pass

    body = auth_on.minted()[0]
    assert body["scope"] == ["inference"]
    assert body["expires_in_seconds"] == TOKEN_LIFETIME_SEC


async def test_a_key_that_cant_be_minted_fails_the_job_before_any_trial_runs(auth_on: Platform, job: Job) -> None:
    auth_on.refuse_mint_number = 1

    with pytest.raises(RuntimeError, match="could not issue 1 of 2 trials"):
        async with _issuer().job_scope([KEY_ENV])(job):
            pytest.fail("the job ran")

    assert auth_on.revoked() == [_jti(0)]


async def test_keys_minted_before_the_job_is_cancelled_are_revoked(auth_on: Platform, job: Job) -> None:
    auth_on.hold_mint_number = 1

    async def run() -> None:
        async with _issuer().job_scope([KEY_ENV])(job):
            pytest.fail("the job ran")

    task = asyncio.create_task(run())
    await auth_on.mint_held.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert auth_on.revoked() == [_jti(0)]


async def test_cancelling_the_job_during_cleanup_still_revokes_every_key(auth_on: Platform, job: Job) -> None:
    """The job's scope doesn't end, closing the platform client, until its revocations have."""
    auth_on.hold_revocations = True

    async def run() -> None:
        async with _issuer().job_scope([KEY_ENV])(job):
            pass

    task = asyncio.create_task(run())
    await auth_on.revocation_held.wait()
    task.cancel()
    asyncio.get_running_loop().call_later(0.05, auth_on.release_revocations.set)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted(auth_on.revoked()) == [_jti(0), _jti(1)]


async def test_a_key_that_couldnt_be_revoked_is_revoked_when_the_job_ends(
    auth_on: Platform, job: Job, hooks: Hooks
) -> None:
    first, _ = job._remaining_trial_configs
    auth_on.revocations_to_refuse = 1

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.end(first)
        assert auth_on.revoked() == []

    assert sorted(auth_on.revoked()) == [_jti(0), _jti(1)]


async def test_a_trial_that_ends_without_error_is_recorded_and_its_key_revoked(
    platform: Platform, job: Job, hooks: Hooks
) -> None:
    first, _ = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.end(first)
        assert platform.root_spans() == [(_nanos(STARTED), _nanos(STARTED.replace(minute=1)))]
        assert platform.root_span_failures() == [False]
        assert platform.revoked() == ([_jti(0)] if platform.auth_enabled else [])


async def test_a_trial_keeps_its_key_while_harbor_will_retry_it(auth_on: Platform, job: Job, hooks: Hooks) -> None:
    first, _ = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.end(first, error="RuntimeError", minutes=1)
        assert auth_on.revoked() == []
        await hooks.end(first, error="RuntimeError", minutes=3)
        assert auth_on.revoked() == [_jti(0)]

    assert auth_on.root_spans()[-1] == (_nanos(STARTED), _nanos(STARTED.replace(minute=3)))


async def test_an_error_harbor_doesnt_retry_ends_the_trial(auth_on: Platform, job: Job, hooks: Hooks) -> None:
    first, _ = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.end(first, error="AgentTimeoutError")
        assert auth_on.revoked() == [_jti(0)]
        assert auth_on.root_span_failures() == [True]


async def test_a_multi_step_trial_whose_step_failed_has_a_failed_root(
    platform: Platform, job: Job, hooks: Hooks
) -> None:
    first, _ = job._remaining_trial_configs
    step = StepResult.model_construct(
        exception_info=ExceptionInfo(
            exception_type="AgentTimeoutError", exception_message="", exception_traceback="", occurred_at=STARTED
        )
    )

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.end(first, steps=[step])

    assert platform.root_span_failures() == [True]


async def test_a_cancelled_trial_has_its_key_revoked_and_is_still_recorded(
    platform: Platform, job: Job, hooks: Hooks
) -> None:
    """Harbor ends a cancelled trial after cancelling it, with an error it would otherwise retry."""
    first, _ = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.cancel(first)
        assert platform.revoked() == ([_jti(0)] if platform.auth_enabled else [])
        await hooks.end(first, error="CancelledError")
        assert len(platform.root_spans()) == 1


async def test_a_trial_the_job_stops_mid_attempt_still_gets_a_failed_root(
    auth_on: Platform, job: Job, hooks: Hooks
) -> None:
    """A cancelled task Job's process is told to stop, and Harbor ends none of its running trials."""
    first, second = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.start(first)
        await hooks.start(second)
        await hooks.end(second)

    assert auth_on.root_span_failures() == [False, True]
    assert auth_on.root_spans()[-1][0] == _nanos(STARTED)
    assert sorted(auth_on.revoked()) == [_jti(0), _jti(1)]


async def test_a_retried_trial_stopped_mid_attempt_keeps_its_first_start(
    auth_on: Platform, job: Job, hooks: Hooks
) -> None:
    first, _ = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.start(first, minutes=0)
        await hooks.end(first, error="RuntimeError", minutes=1)
        await hooks.start(first, minutes=2)

    assert auth_on.root_span_failures() == [True, True]
    assert auth_on.root_spans()[-1][0] == _nanos(STARTED)


async def test_a_trial_that_never_started_gets_no_root(auth_on: Platform, job: Job, hooks: Hooks) -> None:
    async with _issuer().job_scope([KEY_ENV])(job):
        pass

    assert auth_on.root_spans() == []
    assert sorted(auth_on.revoked()) == [_jti(0), _jti(1)]


async def test_a_trial_gets_its_one_token_in_every_variable_its_agent_reads_a_gateway_key_from(
    auth_on: Platform, job: Job
) -> None:
    """A registered agent can call several gateway models, each reading its key from its own variable."""
    configs = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV, "HARNESS_KEY"])(job):
        for config in configs:
            env = resolve_env_vars(config.agent.env)
            assert env[KEY_ENV] == env["HARNESS_KEY"]
    assert len(auth_on.minted()) == 2


async def test_the_job_ending_revokes_every_key_still_live(auth_on: Platform, job: Job, hooks: Hooks) -> None:
    first, _ = job._remaining_trial_configs

    async with _issuer().job_scope([KEY_ENV])(job):
        await hooks.end(first)

    assert sorted(auth_on.revoked()) == [_jti(0), _jti(1)]


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_an_agent_that_calls_no_model_gets_no_tokens(agent: str) -> None:
    assert gateway_job_scope(HarborRuntimeConfig(agent_name=agent), _issuer(), [KEY_ENV]) is None


def _nanos(at: datetime) -> int:
    return int(at.timestamp() * 1_000_000_000)
