# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Broker + Gym host lifecycle (Ray-free). Returns raw Gym host results."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import http.client
import json
import logging
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Coroutine, Mapping
from types import FrameType
from typing import Any, Protocol, TypeVar
from urllib.parse import urlparse

from sandboxed_gym.broker import EpisodeBrokerServer
from sandboxed_gym.config import BrokerEndpoint
from sandboxed_gym.host.models import (
    MIN_PROXY_CUTOFF_S,
    GymHostEgressRule,
    GymHostHandle,
    GymHostSpec,
    GymHostVolumeMount,
    build_bootstrap_env,
)
from sandboxed_gym.host.provider import SandboxedGymHostProvider, get_host_provider
from sandboxed_gym.runtime.gym_host_runtime import (
    ENVIRONMENT_OFFLINE_ENV_KEY,
    ENVIRONMENT_PACKAGE_REQUIRED_ENV_KEY,
    GYM_GLOBAL_CONFIG_ENV_KEY,
)
from sandboxed_gym.serve_config import (
    SandboxedGymServeConfig,
    SandboxedGymSessionDescriptor,
)

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class RolloutTransportError(RuntimeError):
    """A failed rollout POST, tagged with where it failed and whether a retry can help."""

    def __init__(self, message: str, *, retryable: bool, origin: str) -> None:
        super().__init__(message)
        self.retryable = retryable
        #: "proxy" (in transit), "sandbox" (the host reported it), or "client" (our own limits).
        self.origin = origin


def _decode_json_object(body: str) -> Mapping[str, Any] | None:
    try:
        decoded = json.loads(body)
    except (ValueError, TypeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _sandbox_reported_error(body: str) -> str | None:
    """The Gym host's own error message from a response body, or None.

    The host wraps its failures as ``{"error": {"code", "message"}}``. The sandbox proxy uses a
    *flat* ``{"code", "message"}``, so looking only for this nested envelope is what keeps a proxy
    timeout from being reported as an environment failure.
    """
    decoded = _decode_json_object(body)
    if decoded is None or decoded.get("error") is None:
        return None
    error = decoded["error"]
    if isinstance(error, Mapping):
        return f"{error.get('code', 'unknown')}: {error.get('message', '')}".strip()
    return str(error)


def _proxy_reported_error(body: str) -> str | None:
    """The sandbox proxy's own error from a response body, or None.

    The proxy emits a flat ``{"code", "message"}`` with no nested ``error`` key, including when its
    read timeout fires -- where the underlying ``ReadTimeout`` stringifies to an empty message.
    That is a decision the proxy already made, not a dropped connection: retrying re-runs
    generation for the same wall time and fails the same way.
    """
    decoded = _decode_json_object(body)
    if decoded is None or "error" in decoded:
        return None
    code = decoded.get("code")
    message = decoded.get("message", "")
    if not isinstance(code, str) or not isinstance(message, str):
        return None
    return f"{code}: {message}".strip()


def _transit_failure_hint(elapsed: float) -> str:
    """Name the likely cause of a POST that failed in transit, from how long it ran."""
    if elapsed < MIN_PROXY_CUTOFF_S:
        return (
            "too fast to be the proxy's request-duration cap, so the host stopped answering -- "
            "check whether the sandbox is still running (OOMKilled, evicted, or crashed)"
        )
    return (
        "the sandbox proxy caps how long one request may stay open, so lower "
        "sandbox.rollout_chunk_size if this persists"
    )


def _run_coro_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        # `Future.result()` is typed with its own TypeVar, which does not unify with T here.
        return pool.submit(asyncio.run, coro).result()  # ty: ignore[invalid-return-type]


class _SessionAsyncRunner:
    """Run all OpenSandbox SDK calls for one host on the same asyncio loop.

    The SDK's HTTP client is tied to the loop that created the sandbox. If we call
    ``asyncio.run()`` separately for create, wait-ready, and destroy, the first loop
    is already closed when we try to tear the sandbox down, and the pod is leaked.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="sandboxed-gym-sdk", daemon=True)
        self._thread.start()
        # Wait until the loop is running before anyone calls run().
        self._ready.wait()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Wait until this coroutine finishes on the shared loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        """Stop the shared loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


#: Signals a container runtime sends before SIGKILL. SIGKILL and node loss cannot be caught and
#: stay the sandbox ttl_s's problem.
TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def install_termination_cleanup(shutdown: Callable[[], None]) -> None:
    """Run ``shutdown`` when this process exits without it having been called.

    A process that owns a sandbox is the only thing that can name it. Ray tears an actor's worker
    down without running user teardown, so a job that is cancelled, evicted or preempted otherwise
    leaves its sandbox running until ttl_s. ``shutdown`` must tolerate being called twice: an
    ordinary exit runs it directly and then again from ``atexit``.

    For a process the caller owns -- an actor, a CLI. Not for a library embedded in someone else's
    host, whose signal handling is not ours to replace.
    """
    atexit.register(shutdown)

    def _terminate(signum: int, _frame: FrameType | None) -> None:
        # Restored before the cleanup runs, not after: a second signal arriving mid-shutdown then
        # takes the default action and terminates, rather than re-entering this handler on top of
        # an in-flight destroy. Two SIGTERMs mean the sender wants the process gone.
        signal.signal(signum, signal.SIG_DFL)
        LOGGER.warning("received signal %s; destroying sandboxed Gym host before exit", signum)
        try:
            shutdown()
        finally:
            # Re-raised even if cleanup failed, so the exit status still reports the signal --
            # swallowing it would make a cancelled job look like a clean stop.
            os.kill(os.getpid(), signum)

    for signum in TERMINATION_SIGNALS:
        try:
            signal.signal(signum, _terminate)
        except ValueError:
            # Only the main thread may install handlers, and Ray does not promise to call an
            # actor method there. atexit still covers the ordinary exit.
            LOGGER.debug("cannot install a %s handler off the main thread", signum)


def apply_sandbox_runtime_defaults(global_config: dict[str, Any]) -> dict[str, Any]:
    """Inject sandbox-local infra defaults only (not RL training knobs)."""
    from sandboxed_gym.host.entrypoint import gym_uv_cache_dir, gym_uv_venv_dir
    from sandboxed_gym.netutil import (
        DEFAULT_BROKER_PORT_RANGE_HIGH,
        DEFAULT_BROKER_PORT_RANGE_LOW,
    )

    cfg = dict(global_config)
    cfg.pop("ray_head_node_address", None)
    cfg.setdefault("default_host", "127.0.0.1")
    cfg.setdefault("port_range_low", DEFAULT_BROKER_PORT_RANGE_LOW)
    cfg.setdefault("port_range_high", DEFAULT_BROKER_PORT_RANGE_HIGH)
    cfg.setdefault("global_aiohttp_connector_limit_per_host", 16_384)
    cfg.setdefault("global_aiohttp_connector_limit", 65_536)
    cfg.setdefault("uv_cache_dir", gym_uv_cache_dir())
    cfg.setdefault("uv_venv_dir", gym_uv_venv_dir())
    # Makes Gym pass `--python <venv>/bin/python` rather than inferring a target: the training
    # image sets an absolute UV_PYTHON, which outranks the venv Gym just activated and sends the
    # install into a read-only interpreter tree. The only symptom is every Gym server dying with
    # "Process `policy_model` finished unexpectedly!".
    cfg.setdefault("uv_pip_set_python", True)
    return cfg


def collect_gym_host_egress_allows(
    *,
    configured: list[GymHostEgressRule] | tuple[GymHostEgressRule, ...],
    broker_host: str,
    broker_port: int,
    base_urls: list[str] | tuple[str, ...],
    extra: tuple[GymHostEgressRule, ...] = (),
) -> tuple[GymHostEgressRule, ...]:
    rules = list(configured)
    rules.extend(extra)
    rules.append(GymHostEgressRule(host=broker_host, port=broker_port))
    for base_url in base_urls:
        if not base_url:
            continue
        parsed = urlparse(str(base_url))
        if not parsed.hostname:
            continue
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        rules.append(GymHostEgressRule(host=parsed.hostname, port=port))
    deduped: dict[tuple[str, int], GymHostEgressRule] = {}
    for rule in rules:
        deduped.setdefault((rule.host, rule.port), rule)
    return tuple(deduped.values())


def build_gym_host_spec(
    cfg: SandboxedGymServeConfig,
    broker: BrokerEndpoint,
) -> GymHostSpec:
    sandbox = cfg.sandbox
    broker_host = broker.host
    parsed = urlparse(broker.url)
    if parsed.hostname and not parsed.hostname.replace(".", "").isdigit():
        broker_host = parsed.hostname

    dataset_path = None
    dataset_mount = None
    if sandbox.dataset_pvc_claim:
        dataset_path = sandbox.dataset_mount_path
        dataset_mount = GymHostVolumeMount(
            pvc_claim=sandbox.dataset_pvc_claim,
            sub_path=sandbox.dataset_sub_path,
            mount_path=sandbox.dataset_mount_path,
            read_only=True,
        )

    gym_global = apply_sandbox_runtime_defaults(cfg.gym_global_config)
    # Caller's variables first so they cannot displace the gym config, package-required flag,
    # broker endpoint, or mount paths. `build_bootstrap_env` also refuses reserved-key collisions.
    bootstrap_extra = {
        **cfg.host_env,
        GYM_GLOBAL_CONFIG_ENV_KEY: json.dumps(gym_global, sort_keys=True),
    }
    if cfg.environment_path is not None:
        # The mount path is ``/job/environment`` with or without a FileSet. This flag is how the
        # host distinguishes a required package from an image-bundled tree at the same path.
        bootstrap_extra[ENVIRONMENT_PACKAGE_REQUIRED_ENV_KEY] = "true"
    if cfg.environment_offline:
        bootstrap_extra[ENVIRONMENT_OFFLINE_ENV_KEY] = "true"
    bootstrap_env = build_bootstrap_env(
        cfg.job_id,
        cfg.environment_path or sandbox.env_mount_path,
        sandbox.work_mount_path,
        broker.url,
        broker.token,
        sandbox.max_request_bytes,
        sandbox.max_response_bytes,
        dataset_path=dataset_path,
        rollout_deadline_s=sandbox.rollout_timeout_s,
        extra=bootstrap_extra,
    )

    egress_allow = collect_gym_host_egress_allows(
        configured=sandbox.network_policy.egress_allow,
        broker_host=broker_host,
        broker_port=broker.port,
        base_urls=list(cfg.policy_base_urls),
        extra=cfg.egress_extra,
    )

    return GymHostSpec(
        job_id=cfg.job_id,
        runtime_image=sandbox.image,
        environment_mount=GymHostVolumeMount(
            pvc_claim=sandbox.environment_pvc_claim,
            sub_path=sandbox.environment_sub_path,
            mount_path=sandbox.env_mount_path,
            read_only=True,
        ),
        dataset_mount=dataset_mount,
        workspace_mount=GymHostVolumeMount(
            pvc_claim=sandbox.workspace_pvc_claim,
            sub_path=sandbox.workspace_sub_path,
            mount_path=sandbox.work_mount_path,
            read_only=False,
        ),
        egress_allow=egress_allow,
        bootstrap_env=bootstrap_env,
        max_request_bytes=sandbox.max_request_bytes,
        max_response_bytes=sandbox.max_response_bytes,
        ttl_s=sandbox.ttl_s,
        ready_timeout_s=sandbox.ready_timeout_s,
        resources=sandbox.resources,
        runtime_http_port=sandbox.runtime_http_port,
        allow_internet=sandbox.allow_internet,
        public_dns_allow=sandbox.network_policy.public_dns_allow,
        resolver_addresses=sandbox.network_policy.resolver_addresses,
        # Leave an omitted entrypoint provider-specific: Docker preserves the runtime image CMD,
        # while OpenSandbox must supply the standard Gym-host command explicitly because its SDK
        # otherwise starts a persistent-sandbox keepalive. Avoid resolving scripts from this
        # orchestrator container and handing those paths to a different runtime image.
        entrypoint=(tuple(sandbox.entrypoint) if sandbox.entrypoint else None),
    )


class SandboxedGymSession:
    """Live broker + host session. ``run_rollouts`` returns raw Gym host results."""

    def __init__(
        self,
        *,
        cfg: SandboxedGymServeConfig,
        broker_server: EpisodeBroker,
        broker: BrokerEndpoint,
        host_provider: SandboxedGymHostProvider,
        host: GymHostHandle,
        orchestrator_url: str | None = None,
        async_runner: _SessionAsyncRunner | None = None,
    ) -> None:
        self.cfg = cfg
        self._broker_server = broker_server
        self.broker = broker
        self._host_provider = host_provider
        self.host = host
        self.orchestrator_url = orchestrator_url
        self._async_runner = async_runner
        self._rollout_timeout_s = float(cfg.sandbox.rollout_timeout_s)
        self._max_request_bytes = cfg.sandbox.max_request_bytes
        self._max_response_bytes = cfg.sandbox.max_response_bytes
        self._chunk_size = cfg.sandbox.rollout_chunk_size
        self._max_in_flight = cfg.sandbox.rollout_max_in_flight
        self._max_attempts = cfg.sandbox.rollout_max_attempts
        self._retry_backoff_s = float(cfg.sandbox.rollout_retry_backoff_s)

    def descriptor(
        self, *, mode: str | None = None, orchestrator_url: str | None = None
    ) -> SandboxedGymSessionDescriptor:
        url = orchestrator_url if orchestrator_url is not None else self.orchestrator_url
        serve_mode = mode or self.cfg.serve_mode
        return SandboxedGymSessionDescriptor(
            job_id=self.cfg.job_id,
            mode=serve_mode,  # ty: ignore[invalid-argument-type]
            orchestrator_url=url,
            health_url=self.host.health_url,
            rollout_url=(f"{url.rstrip('/')}/rollouts/run" if url else self.host.rollout_url),
            headers=dict(self.host.headers),
            broker_url=self.broker.url,
            broker_token=self.broker.token,
            rollout_auth_token=self.cfg.rollout_auth_token,
        )

    def run_rollouts(self, examples: list[dict[str, Any]]) -> list[Any]:
        """Run ``examples`` on the host as bounded concurrent chunks.

        Results are not in input order and never have been -- Gym yields through
        ``as_completed``. Attribution travels on each result as ``_ng_task_index`` /
        ``_ng_rollout_index``; see ``_with_row_identity`` in the host runtime.
        """
        if not examples:
            raise ValueError("rollout batch must not be empty")
        return _run_coro_sync(self._post_all_chunks(examples))

    async def _post_all_chunks(self, examples: list[dict[str, Any]]) -> list[Any]:
        chunks = [examples[start : start + self._chunk_size] for start in range(0, len(examples), self._chunk_size)]
        semaphore = asyncio.Semaphore(self._max_in_flight)
        # return_exceptions so one chunk failing does not leave the others running detached.
        parts = await asyncio.gather(
            *(
                self._post_chunk_with_retries(index, chunk, len(chunks), semaphore)
                for index, chunk in enumerate(chunks)
            ),
            return_exceptions=True,
        )
        failures = [part for part in parts if isinstance(part, BaseException)]
        # A failure that is not a transport error is a bug in this process, not a report about
        # the sandbox. Re-raised as itself: wrapping it would give it a plausible retryable/origin
        # and hide the stack that explains it.
        for failure in failures:
            if not isinstance(failure, RolloutTransportError):
                raise failure
        if failures:
            whole_batch = (
                " Every chunk failed, so the sandbox itself is the likelier cause than any one request."
                if len(failures) == len(chunks) > 1
                else ""
            )
            raise RolloutTransportError(
                f"{len(failures)} of {len(chunks)} rollout chunk(s) failed for this batch of "
                f"{len(examples)} example(s).{whole_batch} First failure: {failures[0]}",
                retryable=False,
                origin=getattr(failures[0], "origin", "proxy"),
            ) from failures[0]
        return [result for part in parts for result in part]  # ty: ignore[not-iterable]

    async def _post_chunk_with_retries(
        self, index: int, chunk: list[dict[str, Any]], total: int, semaphore: asyncio.Semaphore
    ) -> list[Any]:
        label = f"chunk {index + 1}/{total}"
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with semaphore:
                    # Logged before the POST and inside the semaphore, so the line appears when the
                    # chunk actually goes out. The completion log below is reached only by a chunk
                    # that came back, which leaves a stalled batch -- the failure this transport
                    # exists to survive -- with nothing in the log at all.
                    LOGGER.info(
                        "rollout %s: POST %d example(s)%s",
                        label,
                        len(chunk),
                        "" if attempt == 1 else f" (attempt {attempt}/{self._max_attempts})",
                    )
                    started = time.monotonic()
                    results = await asyncio.to_thread(self._post_chunk, chunk)
            except RolloutTransportError as exc:
                if not exc.retryable or attempt == self._max_attempts:
                    raise RolloutTransportError(
                        f"rollout {label} ({len(chunk)} example(s)) failed after {attempt} attempt(s): {exc}",
                        retryable=exc.retryable,
                        origin=exc.origin,
                    ) from exc
                backoff = self._retry_backoff_s * attempt
                LOGGER.warning(
                    "rollout %s attempt %d/%d failed (%s); retrying in %.1fs: %s",
                    label,
                    attempt,
                    self._max_attempts,
                    exc.origin,
                    backoff,
                    exc,
                )
                # Slept outside the semaphore so a backing-off chunk frees its slot.
                await asyncio.sleep(backoff)
                continue
            LOGGER.info("rollout %s: %d result(s) in %.1fs", label, len(results), time.monotonic() - started)
            return results
        raise AssertionError("unreachable: the loop either returns or raises")

    def _post_chunk(self, chunk: list[dict[str, Any]]) -> list[Any]:
        body = json.dumps({"examples": chunk}).encode("utf-8")
        if len(body) > self._max_request_bytes:
            raise RolloutTransportError(
                f"rollout request exceeds max_request_bytes ({len(body)} > {self._max_request_bytes}); "
                f"lower sandbox.rollout_chunk_size",
                retryable=False,
                origin="client",
            )
        request = urllib.request.Request(
            self.host.rollout_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", **self.host.headers},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._rollout_timeout_s) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise self._transport_error(exc.read().decode("utf-8", errors="replace"), exc.code, chunk, started) from exc
        except urllib.error.URLError as exc:
            # Raised before any response headers arrived, so the host almost certainly never began
            # this chunk. Safe to retry.
            raise self._in_transit_error(str(exc.reason), chunk, started, retryable=True) from exc
        except (OSError, http.client.HTTPException) as exc:
            # The body died mid-read: a reset, a truncated response, a socket timeout. The heartbeat
            # holds this connection open for the length of a batch, so there is far more of it to
            # interrupt than there used to be, and none of it arrives as a URLError.
            #
            # Deliberately NOT retryable. Reaching a body means the host answered, which it only
            # does after `submit_rollouts` has started the work -- and it has no request id to
            # deduplicate against, so a retry runs the same examples a second time while the first
            # is still going. Duplicated generation and duplicated environment side effects are a
            # worse outcome than a failed chunk, and only the host can make this safe.
            raise self._in_transit_error(f"{type(exc).__name__}: {exc}", chunk, started, retryable=False) from exc
        if len(payload) > self._max_response_bytes:
            raise RolloutTransportError(
                f"rollout response exceeds max_response_bytes ({len(payload)} > {self._max_response_bytes})",
                retryable=False,
                origin="client",
            )
        return self._decode_results(payload, time.monotonic() - started)

    def _in_transit_error(
        self, detail: str, chunk: list[dict[str, Any]], started: float, *, retryable: bool
    ) -> RolloutTransportError:
        elapsed = time.monotonic() - started
        return RolloutTransportError(
            f"rollout POST to {self.host.rollout_url} failed after {elapsed:.1f}s for "
            f"{len(chunk)} example(s): {detail}; {_transit_failure_hint(elapsed)}",
            retryable=retryable,
            origin="proxy",
        )

    def _transport_error(
        self, body: str, status: int, chunk: list[dict[str, Any]], started: float
    ) -> RolloutTransportError:
        """Classify a non-2xx response by who produced it."""
        elapsed = time.monotonic() - started
        reported = _sandbox_reported_error(body)
        if reported is not None:
            # The host answered, so the environment failed: deterministic, and a retry only
            # spends generation time to reach the same place.
            return RolloutTransportError(
                f"the sandboxed environment failed this rollout after {elapsed:.1f}s for "
                f"{len(chunk)} example(s) (HTTP {status} from {self.host.rollout_url}): {reported}",
                retryable=False,
                origin="sandbox",
            )
        proxy_reported = _proxy_reported_error(body)
        if proxy_reported is not None:
            # The proxy answered with its own envelope, so it has already given up.
            return RolloutTransportError(
                f"rollout POST was rejected by the sandbox proxy with HTTP {status} after "
                f"{elapsed:.1f}s for {len(chunk)} example(s) to {self.host.rollout_url}: "
                f"{proxy_reported}; {_transit_failure_hint(elapsed)}",
                retryable=False,
                origin="proxy",
            )
        return RolloutTransportError(
            f"rollout POST was rejected in transit with HTTP {status} after {elapsed:.1f}s for "
            f"{len(chunk)} example(s) to {self.host.rollout_url}; {_transit_failure_hint(elapsed)}. "
            f"Body: {body[:512]}",
            retryable=True,
            origin="proxy",
        )

    def _decode_results(self, payload: bytes, elapsed: float | None = None) -> list[Any]:
        try:
            # Strict, and caught rather than avoided: errors="replace" would let a body with one
            # corrupt byte still parse, handing the caller U+FFFD where the host wrote data.
            body = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RolloutTransportError(
                f"the sandboxed Gym host returned a body that is not UTF-8 ({exc})",
                retryable=False,
                origin="sandbox",
            ) from exc
        if not body:
            if elapsed is not None and elapsed < MIN_PROXY_CUTOFF_S:
                raise RolloutTransportError(
                    f"the sandboxed Gym host answered and then sent nothing in {elapsed:.1f}s -- "
                    f"too fast to have been cut in transit, so it died before it could write; "
                    f"check whether the sandbox was OOMKilled or evicted",
                    retryable=False,
                    origin="sandbox",
                )
            raise RolloutTransportError(
                "the sandboxed Gym host sent no body at all"
                + (f" in {elapsed:.1f}s" if elapsed is not None else "")
                + " -- nothing was ever written. Either it died before writing anything (check the "
                "sandbox for an OOMKill or an eviction), or its image predates the rollout "
                "heartbeat and the proxy cut the silent request at its own cap (check "
                "sandbox.image). A dropped connection and a truncated one are indistinguishable "
                "to this client",
                retryable=False,
                origin="sandbox",
            )
        if not body.strip():
            raise RolloutTransportError(
                f"the sandboxed Gym host answered and then stopped without sending a result "
                f"envelope ({len(payload)} byte(s) of heartbeat, then silence); it most likely "
                f"died mid-batch -- check whether the sandbox was OOMKilled or evicted",
                retryable=False,
                origin="sandbox",
            )
        try:
            # The host heartbeats leading whitespace while a batch runs; json tolerates it.
            decoded = json.loads(body)
        except (ValueError, RecursionError) as exc:
            # A *non-empty* malformed body is a broken response contract, not a transport blip, so
            # it must not be retried into looking like one. Wider than JSONDecodeError because json
            # also refuses input outright -- the integer-digit limit, deep nesting -- and narrowing
            # this back lets those leave unclassified.
            raise RolloutTransportError(
                f"the sandboxed Gym host returned a body that is not JSON ({exc}); first 200 bytes: {body[:200]!r}",
                retryable=False,
                origin="sandbox",
            ) from exc
        if isinstance(decoded, dict) and "error" in decoded:
            # A 200 carrying an error: the host had already committed its status line when it
            # failed, so this is the only channel it had left.
            raise RolloutTransportError(
                f"the sandboxed Gym host reported {decoded['error']}",
                retryable=False,
                origin="sandbox",
            )
        if isinstance(decoded, dict) and "results" in decoded:
            results = decoded["results"]
            if not isinstance(results, list):
                raise RolloutTransportError(
                    f"unexpected rollout response shape: 'results' is {type(results).__name__}, not a list",
                    retryable=False,
                    origin="sandbox",
                )
            return results
        if isinstance(decoded, list):
            return decoded
        raise RolloutTransportError(
            f"unexpected rollout response shape: {type(decoded).__name__}",
            retryable=False,
            origin="sandbox",
        )

    def shutdown(self) -> None:
        try:
            if self._async_runner is not None:
                self._async_runner.run(self._host_provider.destroy_host(self.host))
            else:
                _run_coro_sync(self._host_provider.destroy_host(self.host))
        except Exception:
            LOGGER.exception("Failed to destroy sandboxed Gym host")
        finally:
            if self._async_runner is not None:
                self._async_runner.close()
        try:
            self._broker_server.shutdown()
        except Exception:
            LOGGER.exception("Failed to shut down episode broker")


class EpisodeBroker(Protocol):
    """All the orchestrator needs of a broker: somewhere to start it, and a way to stop it.

    Narrow on purpose -- anything added here couples the orchestrator to how a broker is hosted,
    which is the caller's choice to make.
    """

    def start(self) -> BrokerEndpoint: ...

    def shutdown(self) -> None: ...


class SandboxedGymOrchestrator:
    """Start episode broker, provision Gym host, return a live session."""

    def start(
        self,
        cfg: SandboxedGymServeConfig | Mapping[str, Any],
        *,
        broker: EpisodeBroker | None = None,
    ) -> SandboxedGymSession:
        """Bring up a session, hosting the broker in this process unless given one.

        The default hosts it on a background thread here, which suits a caller that rarely
        creates episodes -- an ordinary evaluation creates none. Pass a broker that runs
        elsewhere when they are created heavily: a SWE rollout crosses the broker on every
        ``exec``, and that traffic would otherwise contend for this process's GIL.

        A supplied broker is configured by its owner; ``cfg.episode_broker`` is not applied to it.
        """
        if not isinstance(cfg, SandboxedGymServeConfig):
            cfg = SandboxedGymServeConfig.model_validate(cfg)

        broker_server: EpisodeBroker = broker if broker is not None else EpisodeBrokerServer(cfg.broker_config())
        broker_endpoint = broker_server.start()

        # The broker is serving from here on, so every later failure has to release it --
        # including provider selection, which rejects an unknown name or a bad option. An
        # injected broker can be a Ray actor, which would otherwise outlive this process.
        try:
            host_spec = build_gym_host_spec(cfg, broker_endpoint)
            host_provider = get_host_provider(cfg.host_provider, cfg.sandbox.host_provider_options)
            # Use one loop for create, wait-ready, and destroy. A new asyncio.run() for
            # each call closes the SDK client that create_host just built.
            async_runner = _SessionAsyncRunner()
            host: GymHostHandle | None = None
            try:
                host = async_runner.run(host_provider.create_host(host_spec))
                async_runner.run(host_provider.wait_ready(host, cfg.sandbox.ready_timeout_s))
            except Exception:
                if host is not None:
                    try:
                        async_runner.run(host_provider.destroy_host(host))
                    except Exception:
                        LOGGER.exception("Failed to destroy sandboxed Gym host after startup failure")
                async_runner.close()
                raise
        except Exception:
            # A supplied broker is the caller's code. Let its failure be logged rather than
            # replace the startup error that is the reason we are here.
            try:
                broker_server.shutdown()
            except Exception:
                LOGGER.exception("Failed to shut down episode broker after startup failure")
            raise

        return SandboxedGymSession(
            cfg=cfg,
            broker_server=broker_server,
            broker=broker_endpoint,
            host_provider=host_provider,
            host=host,
            async_runner=async_runner,
        )
