# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenSandbox episode driver's translation layer.

The driver replaces NeMo-Gym's 1541-line provider with the seven operations the episode backend
actually uses. Provisioning a real sandbox needs a cluster, so what is covered here is the part
that carries judgement and would otherwise be unverified: how typed resources, exec identity,
command output and lifecycle status map onto the SDK's shapes.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from typing import cast

import pytest
from sandboxed_gym.backends._opensandbox_driver import (
    _SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY,
    _STATUS_ALIASES,
    OpenSandboxDriver,
    _exec_identity,
    _joined_output,
    _resource_limits,
    _resource_requests,
)
from sandboxed_gym.backends.base import UnsupportedEpisodeOperationError
from sandboxed_gym.sandbox_types import SandboxHandle, SandboxResources, SandboxSpec, SandboxStatus

requires_opensandbox = pytest.mark.skipif(
    importlib.util.find_spec("opensandbox") is None,
    reason="needs the OpenSandbox SDK; install the `opensandbox` extra to run",
)


def spec_with(**resources: object) -> SandboxSpec:
    return SandboxSpec(image="img:1", resources=SandboxResources(**resources))  # ty: ignore[invalid-argument-type]


def test_configured_resource_limits_are_forwarded_as_strings() -> None:
    # `resource` is the hard cap, separate from the scheduling requests. Values arrive from
    # operator YAML, so ints must survive as the strings the SDK expects.
    assert _resource_limits({"resource": {"cpu": 4, "memory": "12Gi"}}) == {"cpu": "4", "memory": "12Gi"}


def test_absent_resource_limits_are_none_rather_than_empty() -> None:
    # None lets the SDK apply its own default; an empty dict would be forwarded as a real,
    # zero-valued limit set.
    assert _resource_limits({}) is None
    assert _resource_limits({"resource": {}}) is None


def test_resources_map_onto_the_sdk_kubernetes_style_strings() -> None:
    # The SDK documents this format: cpu in millicores, memory with a binary suffix, gpu as a
    # count. Fractional CPU is the case worth pinning -- 0.5 cores is 500m, not "0.5".
    requests = _resource_requests(spec_with(cpu=0.5, memory_mib=2048, disk_gib=10, gpu=2))

    assert requests == {"cpu": "500m", "memory": "2048Mi", "ephemeral-storage": "10Gi", "gpu": "2"}


def test_unset_resources_are_omitted_rather_than_sent_as_zero() -> None:
    # A zero request is not the same as no request: it would pin the episode to nothing.
    assert _resource_requests(spec_with()) == {}


def test_a_gpu_type_request_is_refused_rather_than_dropped() -> None:
    """There is no documented request key for a device model.

    Dropping it silently would grade the episode on whatever GPU it happened to land on, which is
    the downgrade the backend contract forbids.
    """
    with pytest.raises(UnsupportedEpisodeOperationError, match="GPU type"):
        _resource_requests(spec_with(gpu=1, gpu_type="a100"))


@pytest.mark.parametrize(("user", "expected"), [(None, {}), (1000, {"uid": 1000}), ("1000", {"uid": 1000})])
def test_numeric_users_reach_the_sdk_as_a_uid(user: str | int | None, expected: dict[str, int]) -> None:
    assert _exec_identity(user) == expected


def test_a_named_user_is_refused_rather_than_guessed() -> None:
    # Resolving a name needs the image's passwd database; running as the wrong id is a silent
    # privilege change either way.
    with pytest.raises(UnsupportedEpisodeOperationError, match="numeric uid"):
        _exec_identity("root")


def _sdk_messages(*texts: str) -> list[object]:
    OutputMessage = getattr(importlib.import_module("opensandbox.models.execd"), "OutputMessage")
    return [OutputMessage(text=text, timestamp=0) for text in texts]


def _driver_with_sandbox(sandbox: object) -> tuple[OpenSandboxDriver, SandboxHandle]:
    return OpenSandboxDriver(), SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=sandbox)


@requires_opensandbox
@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("b23e3df7bafd4529a159b7429c8a220e",), "b23e3df7bafd4529a159b7429c8a220e"),
        (("a", "b"), "a\nb"),
        (("a", "\n", "b"), "a\n\nb"),
        (("\n",), ""),
    ],
    ids=["single-line", "multiple-lines", "blank-line", "only-a-blank-line"],
)
def test_output_is_the_message_text_rejoined_into_lines(texts: tuple[str, ...], expected: str) -> None:
    """execd sends one message per line with its terminator stripped, and a blank line as ``"\\n"``.

    These are the shapes a live OpenSandbox 0.1.16 server returned for ``echo <nonce>``,
    ``printf 'a\\nb\\n'``, ``printf 'a\\n\\nb\\n'`` and a bare ``echo``. Reading any field but
    ``text`` leaks the message model's repr, timestamp included, into the stdout callers compare
    against. A bare ``echo`` did write output, so it is ``""``, not ``None``.
    """
    assert _joined_output(_sdk_messages(*texts)) == expected  # ty: ignore[invalid-argument-type]


@requires_opensandbox
@pytest.mark.parametrize("empty", [None, [], [""]], ids=["absent", "no-messages", "empty-message"])
def test_absent_output_is_none_not_an_empty_string(empty: list[str] | None) -> None:
    # The contract's `stdout`/`stderr` are optional, and callers distinguish "no output" from
    # "empty output" when deciding whether a command said anything.
    messages = None if empty is None else _sdk_messages(*empty)
    assert _joined_output(messages) is None


@requires_opensandbox
async def test_exec_returns_stdout_and_stderr_text_from_the_sdk_execution() -> None:
    from types import SimpleNamespace

    execd = importlib.import_module("opensandbox.models.execd")
    execution = execd.Execution(
        logs=execd.ExecutionLogs(stdout=_sdk_messages("out-1", "out-2"), stderr=_sdk_messages("err")),
        exit_code=0,
    )

    async def run(command: str, opts: object) -> object:
        return execution

    driver, handle = _driver_with_sandbox(SimpleNamespace(commands=SimpleNamespace(run=run)))

    result = await driver.exec(handle, "anything")

    assert (result.stdout, result.stderr, result.return_code) == ("out-1\nout-2", "err", 0)


@requires_opensandbox
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("Pending", SandboxStatus.STARTING),
        ("Running", SandboxStatus.RUNNING),
        ("Terminated", SandboxStatus.STOPPED),
        ("Failed", SandboxStatus.ERROR),
        ("Paused", SandboxStatus.UNKNOWN),
    ],
)
async def test_status_maps_the_sdk_lifecycle_state(state: str, expected: SandboxStatus) -> None:
    """The SDK reports lifecycle as ``SandboxInfo.status.state``, capitalised (``"Running"``).

    ``status`` itself is a model; matching its repr, or its free-form ``reason``, reads a live
    sandbox as ``unknown``.
    """
    from types import SimpleNamespace

    SdkSandboxStatus = getattr(importlib.import_module("opensandbox.models.sandboxes"), "SandboxStatus")

    async def get_info() -> object:
        return SimpleNamespace(status=SdkSandboxStatus(state=state, reason="SOME_REASON"))

    driver, handle = _driver_with_sandbox(SimpleNamespace(get_info=get_info))

    assert await driver.status(handle) == expected


def test_every_status_alias_maps_onto_a_real_contract_status() -> None:
    # The alias table exists because the SDK's vocabulary is wider than the contract's. A typo
    # here would resolve to UNKNOWN at runtime and look like a dead sandbox.
    assert all(isinstance(value, SandboxStatus) for value in _STATUS_ALIASES.values())
    assert "running" not in _STATUS_ALIASES, "statuses that already match must not be aliased"


@requires_opensandbox
async def test_destroy_sandboxes_matching_lists_every_page_and_kills_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    SandboxManager = getattr(importlib.import_module("opensandbox.manager"), "SandboxManager")

    manager = SimpleNamespace(
        filters=[],
        killed=[],
        closed=False,
    )

    async def list_sandbox_infos(sandbox_filter: object) -> object:
        manager.filters.append(sandbox_filter)
        page = getattr(sandbox_filter, "page")
        return SimpleNamespace(
            sandbox_infos=[SimpleNamespace(id=f"orphan-{page}")],
            pagination=SimpleNamespace(has_next_page=page == 1),
        )

    async def kill_sandbox(sandbox_id: str) -> None:
        manager.killed.append(sandbox_id)

    async def close() -> None:
        manager.closed = True

    manager.list_sandbox_infos = list_sandbox_infos
    manager.kill_sandbox = kill_sandbox
    manager.close = close

    async def create_manager(*, connection_config: object) -> object:
        return manager

    monkeypatch.setattr(SandboxManager, "create", create_manager)

    metadata = {_SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY: "create-1"}
    removed = await OpenSandboxDriver().destroy_sandboxes_matching(metadata)

    assert removed == ("orphan-1", "orphan-2")
    assert manager.killed == ["orphan-1", "orphan-2"]
    assert [sandbox_filter.metadata for sandbox_filter in manager.filters] == [
        metadata,
        metadata,
    ]
    assert manager.closed is True


async def test_destroy_sandboxes_matching_refuses_an_empty_selector() -> None:
    with pytest.raises(ValueError, match="at least one metadata selector"):
        await OpenSandboxDriver().destroy_sandboxes_matching({})


@requires_opensandbox
async def test_create_reclaims_only_its_creation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    Sandbox = getattr(importlib.import_module("opensandbox"), "Sandbox")

    driver = OpenSandboxDriver()
    captured_metadata: dict[str, str] = {}
    cleanup_filters: list[dict[str, str]] = []

    async def fail_create(image: str, **kwargs: object) -> object:
        captured_metadata.update(cast(Mapping[str, str], kwargs["metadata"]))
        raise RuntimeError("create failed")

    async def cleanup(metadata: Mapping[str, str]) -> tuple[str, ...]:
        cleanup_filters.append(dict(metadata))
        return ("orphan-1",)

    monkeypatch.setattr(Sandbox, "create", staticmethod(fail_create))
    monkeypatch.setattr(driver, "destroy_sandboxes_matching", cleanup)

    with pytest.raises(RuntimeError, match="create failed"):
        await driver.create(SandboxSpec(image="img:1", metadata={"existing": "value"}))

    create_attempt_id = captured_metadata[_SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY]
    assert len(create_attempt_id) == 32
    assert captured_metadata["existing"] == "value"
    assert cleanup_filters == [{_SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY: create_attempt_id}]


@requires_opensandbox
async def test_cleanup_failure_does_not_mask_create_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    Sandbox = getattr(importlib.import_module("opensandbox"), "Sandbox")

    driver = OpenSandboxDriver()

    async def fail_create(image: str, **kwargs: object) -> object:
        raise RuntimeError("create failed")

    async def fail_cleanup(metadata: Mapping[str, str]) -> tuple[str, ...]:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(Sandbox, "create", staticmethod(fail_create))
    monkeypatch.setattr(driver, "destroy_sandboxes_matching", fail_cleanup)

    with caplog.at_level("ERROR", logger="sandboxed_gym.backends._opensandbox_driver"):
        with pytest.raises(RuntimeError, match="create failed"):
            await driver.create(SandboxSpec(image="img:1"))

    assert "failed to reconcile sandbox create attempt" in caplog.text


@requires_opensandbox
async def test_create_passes_the_configured_cap_and_the_episode_requests_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What this guards is the wiring, not the translation `_resource_limits` already covers.
    # `resource` is the operator's hard cap and `resource_requests` is what the episode is
    # scheduled with; they are separate SDK arguments. Pinning only the cap would still pass if
    # the two were collapsed back into one, which is the shape the original bug took.
    from types import SimpleNamespace

    Sandbox = getattr(importlib.import_module("opensandbox"), "Sandbox")

    driver = OpenSandboxDriver(create={"resource": {"cpu": 4, "memory": "12Gi"}})
    captured: dict[str, object] = {}

    async def capture_create(image: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(sandbox_id="sandbox-1")

    monkeypatch.setattr(Sandbox, "create", staticmethod(capture_create))

    handle = await driver.create(spec_with(cpu=0.5))

    assert handle.sandbox_id == "sandbox-1"
    assert captured["resource"] == {"cpu": "4", "memory": "12Gi"}
    assert captured["resource_requests"] == {"cpu": "500m"}
