# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox provider, against a fake server. The SDK calls themselves are checked on a cluster."""

from __future__ import annotations

import base64
import logging
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest
from kubernetes import client as k8s
from nemo_builder_plugin.run import opensandbox_sandbox
from nemo_builder_plugin.run.opensandbox_sandbox import (
    EXECD_TMPDIR,
    KEEPALIVE,
    CommandResult,
    OpenSandboxProvider,
)
from nemo_builder_plugin.run.sandbox import (
    JOB_LABEL,
    SANDBOX_DEADLINE_SECONDS,
    SANDBOX_LABEL,
    Mount,
    job_key,
    mounts,
)
from nemo_builder_plugin.run.supervise import _build_group, _read_api_key
from nemo_builder_plugin.steps import DEFAULT_EGRESS_ALLOW, ContextSource, SandboxGroup, SandboxImage, SandboxSpec
from nhx_sandbox import egress
from nhx_sandbox.egress import EgressPolicy
from nhx_sandbox.opensandbox_policy import EgressVerificationError

JOB_SUB_PATH = "jobs/default/abc"

#: What the fake sidecar reports: the policy it was given, unless a test says otherwise.
APPLIED_AS_GIVEN = object()


@pytest.fixture(autouse=True)
def _no_local_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this machine's nameservers out of the policies these tests build."""
    monkeypatch.setattr(egress, "local_resolver_addresses", lambda path=None: ())


class FakeSandbox:
    def __init__(
        self, results: Mapping[str, int | None], *, fail_on: str | None, destroy_fails: bool, applied: object
    ) -> None:
        self.commands: list[str] = []
        self.destroyed = False
        self.given: EgressPolicy | None = None
        self._results, self._fail_on, self._destroy_fails = results, fail_on, destroy_fails
        self._applied = applied

    @property
    def id(self) -> str:
        return "sbx-1"

    def run(self, command: str, *, timeout_seconds: int) -> CommandResult:
        assert timeout_seconds > 0
        self.commands.append(command)
        if self._fail_on and self._fail_on in command:
            raise ConnectionError("lost the sandbox")
        if command.startswith("rm -rf"):
            image = next((name for name in self._results if f"/out/{name}" in command), "")
            return CommandResult(exit_code=1 if image == "unclearable" else 0, output=[])
        image = command.split("--oci-layout-path=/nhx-work/out/")[1].split()[0]
        return CommandResult(exit_code=self._results.get(image), output=[f"building {image}\nstep 2"])

    def applied_egress(self) -> object:
        if isinstance(self._applied, Exception):
            raise self._applied
        if self._applied is not APPLIED_AS_GIVEN:
            return self._applied
        assert self.given is not None
        return SimpleNamespace(default_action=self.given.default_action, egress=list(self.given.rules))

    def destroy(self) -> None:
        self.destroyed = True
        if self._destroy_fails:
            raise ConnectionError("the server is gone")


class FakeApi:
    def __init__(
        self,
        results: Mapping[str, int | None] | None = None,
        *,
        create_fails: bool = False,
        fail_on: str | None = None,
        destroy_fails: bool = False,
        existing: list[str] | None = None,
        applied: object = APPLIED_AS_GIVEN,
    ) -> None:
        self.created: list[dict[str, Any]] = []
        self.killed_by: list[dict[str, str]] = []
        self.sandbox = FakeSandbox(results or {}, fail_on=fail_on, destroy_fails=destroy_fails, applied=applied)
        self._create_fails = create_fails
        self._existing = existing or []

    def create(self, **kwargs: Any) -> FakeSandbox:
        self.created.append(kwargs)
        if self._create_fails:
            raise TimeoutError("admission refused the pod; the SDK only times out")
        self.sandbox.given = kwargs["egress"]
        return self.sandbox

    def kill_labelled(self, labels: Mapping[str, str]) -> list[str]:
        self.killed_by.append(dict(labels))
        return list(self._existing)


def _sandbox(**overrides: object) -> SandboxSpec:
    return SandboxSpec.model_validate(
        {
            "image": "nhx-kaniko:test",
            "provider": "opensandbox",
            "opensandbox": {"domain": "opensandbox-server.opensandbox-system.svc.cluster.local"},
            "work_pvc": "nhx-build-work",
            "node_selector": {},
            "dns_nameservers": ["8.8.8.8"],
            "cpu": "2",
            "memory": "8Gi",
            **overrides,
        }
    )


def _group(*names: str) -> SandboxGroup:
    return SandboxGroup(
        source=ContextSource(fileset="ws/fs-a"),
        images=[SandboxImage(image=name, platform="linux/amd64", dockerfile="Dockerfile") for name in names],
    )


def _provider(api: FakeApi, **sandbox: object) -> OpenSandboxProvider:
    return OpenSandboxProvider(
        api, sandbox=_sandbox(**sandbox), workspace="default", job_id="abc", job_sub_path=JOB_SUB_PATH
    )


class TestTheSandbox:
    def _created(self) -> dict[str, Any]:
        api = FakeApi({"img-a": 0})
        _provider(api).build(0, _group("img-a"))
        return api.created[0]

    def test_it_runs_the_kaniko_image_kept_up_between_commands(self) -> None:
        created = self._created()
        assert (created["image"], created["entrypoint"]) == ("nhx-kaniko:test", KEEPALIVE)

    def test_execd_writes_its_logs_where_kaniko_does_not_snapshot(self) -> None:
        """In /tmp, kaniko's own log would land in the image, and a multi-stage build would delete it."""
        assert self._created()["env"] == {"TMPDIR": EXECD_TMPDIR}

    def test_it_mounts_what_the_plain_pod_mounts(self) -> None:
        created = self._created()
        assert created["claim"] == "nhx-build-work"
        assert created["mounts"] == mounts(_group("img-a"), JOB_SUB_PATH)
        assert created["mounts"][0] == Mount(f"{JOB_SUB_PATH}/context/ws/fs-a", "/nhx-work/context/ws/fs-a", True)

    def test_it_carries_the_labels_a_policy_and_a_sweep_select_on(self) -> None:
        labels = self._created()["labels"]
        assert labels[SANDBOX_LABEL] == "true"
        assert labels[JOB_LABEL] == job_key("default", "abc")

    def test_it_ends_even_if_this_step_does_not_end_it(self) -> None:
        assert self._created()["ttl_seconds"] == SANDBOX_DEADLINE_SECONDS

    def test_it_is_sized_by_the_deployment(self) -> None:
        created = self._created()
        assert (created["cpu"], created["memory"]) == ("2", "8Gi")


class TestItsNetwork:
    """Each sandbox is created denying egress by default, and checked for it before anything runs in it."""

    def _policy(self, **sandbox: object) -> EgressPolicy:
        api = FakeApi({"img-a": 0})
        _provider(api, **sandbox).build(0, _group("img-a"))
        return api.created[0]["egress"]

    def test_it_denies_by_default_and_allows_the_deployments_list(self) -> None:
        policy = self._policy()
        assert policy.default_action == "deny"
        assert [rule.target for rule in policy.rules if rule.action == "allow"] == DEFAULT_EGRESS_ALLOW

    def test_it_denies_the_cluster_and_the_metadata_server(self) -> None:
        denied = {rule.target for rule in self._policy().rules if rule.action == "deny"}
        assert {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "fc00::/7"} <= denied

    def test_the_deployment_names_what_it_may_reach(self) -> None:
        policy = self._policy(egress_allow=["*.example.com", "10.1.2.3"])
        assert [rule.target for rule in policy.rules if rule.action == "allow"] == ["*.example.com", "10.1.2.3"]
        # An allowed address is carved out of the range that would deny it.
        assert "10.0.0.0/8" not in {rule.target for rule in policy.rules if rule.action == "deny"}

    def test_a_sandbox_that_allows_by_default_runs_nothing(self) -> None:
        api = FakeApi({"img-a": 0}, applied=SimpleNamespace(default_action="allow", egress=[]))
        with pytest.raises(EgressVerificationError, match="default_action='allow'"):
            _provider(api).build(0, _group("img-a"))
        assert api.sandbox.commands == []
        assert api.sandbox.destroyed

    def test_a_sandbox_whose_policy_cannot_be_read_runs_nothing(self) -> None:
        api = FakeApi({"img-a": 0}, applied=ConnectionError("no egress sidecar"))
        with pytest.raises(ConnectionError):
            _provider(api).build(0, _group("img-a"))
        assert api.sandbox.commands == []
        assert api.sandbox.destroyed

    def test_it_fails_only_its_own_images(self) -> None:
        api = FakeApi({"img-a": 0}, applied=SimpleNamespace(default_action="allow", egress=[]))
        assert _build_group(_provider(api), 0, _group("img-a", "img-b")) == 2


class TestBuilding:
    def test_each_image_is_its_own_command_after_its_output_is_emptied(self) -> None:
        api = FakeApi({"img-a": 0, "img-b": 0})
        _provider(api).build(0, _group("img-a", "img-b"))
        commands = api.sandbox.commands
        assert [c.split()[0] for c in commands] == ["rm", "/kaniko/executor", "rm", "/kaniko/executor"]
        assert "/nhx-work/out/img-a" in commands[0] and "--oci-layout-path=/nhx-work/out/img-a" in commands[1]
        assert all("--no-push" in c for c in commands[1::2])

    def test_each_exit_code_is_the_result(self) -> None:
        api = FakeApi({"img-a": 0, "img-b": 7})
        assert _provider(api).build(0, _group("img-a", "img-b")) == {"img-a": 0, "img-b": 7}

    def test_an_image_without_an_exit_code_has_no_result(self) -> None:
        api = FakeApi({"img-a": None, "img-b": 0})
        assert _provider(api).build(0, _group("img-a", "img-b")) == {"img-b": 0}

    def test_an_output_that_cannot_be_emptied_is_not_built_into(self) -> None:
        """Else a failed build would leave an earlier attempt's layout for `push`."""
        api = FakeApi({"unclearable": 0, "img-b": 0})
        assert _provider(api).build(0, _group("unclearable", "img-b")) == {"img-b": 0}
        assert not any("--oci-layout-path=/nhx-work/out/unclearable" in c for c in api.sandbox.commands)

    def test_output_is_logged_a_line_at_a_time(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=opensandbox_sandbox.__name__):
            _provider(FakeApi({"img-a": 0})).build(0, _group("img-a"))
        messages = [record.getMessage() for record in caplog.records]
        assert "  building img-a" in messages and "  step 2" in messages


class TestTheSandboxAlwaysGoes:
    def test_after_building(self) -> None:
        api = FakeApi({"img-a": 0})
        _provider(api).build(0, _group("img-a"))
        assert api.sandbox.destroyed

    def test_after_a_command_fails(self) -> None:
        api = FakeApi({"img-a": 0}, fail_on="/kaniko/executor")
        with pytest.raises(ConnectionError):
            _provider(api).build(0, _group("img-a"))
        assert api.sandbox.destroyed

    def test_a_sandbox_that_cannot_be_deleted_keeps_what_it_built(self) -> None:
        api = FakeApi({"img-a": 0}, destroy_fails=True)
        assert _provider(api).build(0, _group("img-a")) == {"img-a": 0}


class TestEachSandboxFailsAlone:
    def test_a_sandbox_that_cannot_be_created_fails_only_its_own_images(self) -> None:
        api = FakeApi(create_fails=True)
        assert _build_group(_provider(api), 0, _group("img-a", "img-b")) == 2

    def test_a_built_group_counts_its_failures(self) -> None:
        api = FakeApi({"img-a": 0, "img-b": 1})
        assert _build_group(_provider(api), 0, _group("img-a", "img-b", "img-c")) == 2


class TestTheSweep:
    def test_it_kills_what_this_job_left_and_nothing_else(self) -> None:
        api = FakeApi(existing=["sbx-old"])
        _provider(api).sweep()
        assert api.killed_by == [{JOB_LABEL: job_key("default", "abc")}]


class _Secrets:
    def __init__(self, data: dict[str, str] | None) -> None:
        self.read: list[tuple[str, str]] = []
        self._data = data

    def read_namespaced_secret(self, *, name: str, namespace: str) -> SimpleNamespace:
        self.read.append((name, namespace))
        return SimpleNamespace(data=self._data)


class TestTheKey:
    """Read by this step's ServiceAccount, so it reaches no other pod, and never through the step's env."""

    def test_it_is_read_from_the_named_secret_in_this_namespace(self) -> None:
        api = _Secrets({"api-key": base64.b64encode(b"tenant-key\n").decode()})
        key = _read_api_key(cast(k8s.CoreV1Api, api), name="opensandbox-builder-api-key", namespace="nhx-builds")
        assert key == "tenant-key"
        assert api.read == [("opensandbox-builder-api-key", "nhx-builds")]

    @pytest.mark.parametrize("data", [None, {}, {"other": "eA=="}])
    def test_a_secret_without_the_key_is_refused_by_name(self, data: dict[str, str] | None) -> None:
        with pytest.raises(RuntimeError, match="opensandbox-builder-api-key in nhx-builds has no api-key"):
            _read_api_key(
                cast(k8s.CoreV1Api, _Secrets(data)), name="opensandbox-builder-api-key", namespace="nhx-builds"
            )
