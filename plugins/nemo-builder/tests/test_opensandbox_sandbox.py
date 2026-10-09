# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox provider, against a fake server. ``test_opensandbox_sdk.py`` checks the SDK calls behind it."""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest
from kubernetes import client as k8s
from nemo_builder_plugin.run import opensandbox_sandbox
from nemo_builder_plugin.run.opensandbox_sandbox import (
    DROPCAPS,
    ENFORCEMENT_MODE,
    EXECD_TMPDIR,
    KEEPALIVE,
    AppliedEgress,
    OpenSandboxProvider,
)
from nemo_builder_plugin.run.sandbox import (
    JOB_LABEL,
    KANIKO_FEATURE_FLAGS,
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

JOB_SUB_PATH = "jobs/default/abc"

#: What a fake egress sidecar reports, from the policy its sandbox was created with.
Readback = Callable[[EgressPolicy], AppliedEgress]


def as_given(policy: EgressPolicy, *, mode: str | None = ENFORCEMENT_MODE) -> AppliedEgress:
    return AppliedEgress(mode, SimpleNamespace(default_action=policy.default_action, egress=list(policy.rules)))


class FakeSandbox:
    def __init__(
        self,
        sandbox_id: str,
        results: Mapping[str, int | None],
        *,
        given: EgressPolicy,
        fail_on: str | None,
        destroy_fails: bool,
        applied: Readback | Exception,
    ) -> None:
        self.id = sandbox_id
        self.commands: list[str] = []
        self.destroyed = False
        self._results, self._given, self._fail_on, self._destroy_fails = results, given, fail_on, destroy_fails
        self._applied = applied

    def run(self, command: str, *, timeout_seconds: int, on_output: Callable[[str], None]) -> int | None:
        assert timeout_seconds > 0
        self.commands.append(command)
        if self._fail_on and self._fail_on in command:
            raise ConnectionError("lost the sandbox")
        if command.startswith("rm -rf"):
            image = next((name for name in self._results if f"/out/{name}" in command), "")
            return 1 if image == "unclearable" else 0
        image = command.split("--oci-layout-path=/nhx-work/out/")[1].split()[0]
        on_output(f"building {image}\nstep 2")
        return self._results.get(image)

    def applied_egress(self) -> AppliedEgress:
        if isinstance(self._applied, Exception):
            raise self._applied
        return self._applied(self._given)

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
        existing: Mapping[str, str] | None = None,
        kill_fails: bool = False,
        killed_stay: bool = False,
        applied: Readback | Exception = as_given,
    ) -> None:
        self.created: list[dict[str, Any]] = []
        self.sandboxes: list[FakeSandbox] = []
        self.killed: list[str] = []
        self._results, self._create_fails, self._fail_on = results or {}, create_fails, fail_on
        self._destroy_fails, self._applied = destroy_fails, applied
        #: Each existing sandbox's ID, and the job label it carries.
        self._existing = dict(existing or {})
        self._kill_fails, self._killed_stay = kill_fails, killed_stay

    @property
    def commands(self) -> list[str]:
        return [command for sandbox in self.sandboxes for command in sandbox.commands]

    def create(self, **kwargs: Any) -> FakeSandbox:
        self.created.append(kwargs)
        if self._create_fails:
            raise TimeoutError("admission refused the pod; the SDK only times out")
        sandbox = FakeSandbox(
            f"sbx-{len(self.sandboxes)}",
            self._results,
            given=kwargs["egress"],
            fail_on=self._fail_on,
            destroy_fails=self._destroy_fails,
            applied=self._applied,
        )
        self.sandboxes.append(sandbox)
        return sandbox

    def labelled(self, labels: Mapping[str, str]) -> list[str]:
        assert list(labels) == [JOB_LABEL]
        return [sandbox_id for sandbox_id, job in self._existing.items() if job == labels[JOB_LABEL]]

    def kill(self, sandbox_id: str) -> None:
        self.killed.append(sandbox_id)
        if self._kill_fails:
            raise ConnectionError("the server answered 404")
        if not self._killed_stay:
            del self._existing[sandbox_id]


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

    def test_each_image_has_a_sandbox_of_its_own(self) -> None:
        """In a shared one, an earlier image's `RUN` could replace what starts a later image's build."""
        api = FakeApi({"img-a": 0, "img-b": 0})
        _provider(api).build(0, _group("img-a", "img-b"))
        assert [sandbox.commands[-1].split("--oci-layout-path=")[1].split()[0] for sandbox in api.sandboxes] == [
            "/nhx-work/out/img-a",
            "/nhx-work/out/img-b",
        ]

    def test_it_mounts_its_context_and_only_its_own_output(self) -> None:
        api = FakeApi({"img-a": 0, "img-b": 0})
        _provider(api).build(0, _group("img-a", "img-b"))
        created = api.created[1]
        assert created["claim"] == "nhx-build-work"
        assert created["mounts"] == mounts(_group("img-b"), JOB_SUB_PATH)
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

    def _refused(self, caplog: pytest.LogCaptureFixture, applied: Readback | Exception) -> str:
        """Build with a sidecar reporting ``applied``, check nothing ran, and return what was logged."""
        api = FakeApi({"img-a": 0}, applied=applied)
        with caplog.at_level(logging.ERROR, logger=opensandbox_sandbox.__name__):
            assert _provider(api).build(0, _group("img-a")) == {}
        assert api.commands == []
        assert api.sandboxes[0].destroyed
        return caplog.text

    def test_it_denies_by_default_and_allows_the_deployments_list(self) -> None:
        policy = self._policy()
        assert policy.default_action == "deny"
        assert [rule.target for rule in policy.rules if rule.action == "allow"] == DEFAULT_EGRESS_ALLOW

    def test_it_denies_the_cluster_and_the_metadata_server(self) -> None:
        denied = {rule.target for rule in self._policy().rules if rule.action == "deny"}
        assert {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "fc00::/7"} <= denied

    def test_this_steps_resolver_is_denied_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On GKE with Cloud DNS, the resolver is the metadata server. The sidecar reaches it on its own."""
        monkeypatch.setattr(egress, "local_resolver_addresses", lambda path=None: ("169.254.169.254", "10.96.0.10"))
        policy = self._policy()
        assert [rule.target for rule in policy.rules if rule.action == "allow"] == DEFAULT_EGRESS_ALLOW
        assert {"10.0.0.0/8", "169.254.0.0/16"} <= {rule.target for rule in policy.rules if rule.action == "deny"}

    def test_the_deployment_names_what_it_may_reach(self) -> None:
        policy = self._policy(egress_allow=["*.example.com", "10.1.2.3"])
        assert [rule.target for rule in policy.rules if rule.action == "allow"] == ["*.example.com", "10.1.2.3"]
        # An allowed address is carved out of the range that would deny it.
        assert "10.0.0.0/8" not in {rule.target for rule in policy.rules if rule.action == "deny"}

    def test_a_sandbox_that_allows_by_default_runs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        applied = AppliedEgress(ENFORCEMENT_MODE, SimpleNamespace(default_action="allow", egress=[]))
        assert "default_action='allow'" in self._refused(caplog, lambda given: applied)

    def test_a_sidecar_that_filters_names_only_runs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """In `dns` mode, the server's default, it enforces nothing on addresses."""
        assert "enforces 'dns', not 'dns+nft'" in self._refused(caplog, lambda given: as_given(given, mode="dns"))

    def test_a_sidecar_that_reports_no_mode_runs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        assert "enforces None" in self._refused(caplog, lambda given: as_given(given, mode=None))

    def test_a_sandbox_missing_a_denied_range_runs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """The sidecar reports the policy it was given, so a rule it doesn't report is one it doesn't enforce."""

        def without_the_cluster(given: EgressPolicy) -> AppliedEgress:
            rules = [rule for rule in given.rules if rule.target != "10.0.0.0/8"]
            return AppliedEgress(ENFORCEMENT_MODE, SimpleNamespace(default_action="deny", egress=rules))

        assert "did not apply 1 requested egress rule(s)" in self._refused(caplog, without_the_cluster)

    def test_a_sandbox_whose_policy_cannot_be_read_runs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        assert "no egress sidecar" in self._refused(caplog, ConnectionError("no egress sidecar"))

    def test_it_fails_only_its_own_images(self) -> None:
        applied = AppliedEgress(ENFORCEMENT_MODE, SimpleNamespace(default_action="allow", egress=[]))
        api = FakeApi({"img-a": 0}, applied=lambda given: applied)
        assert _build_group(_provider(api), 0, _group("img-a", "img-b")) == 2


class TestBuilding:
    def test_each_image_is_built_through_nhx_dropcaps_after_its_output_is_emptied(self) -> None:
        api = FakeApi({"img-a": 0, "img-b": 0})
        _provider(api).build(0, _group("img-a", "img-b"))
        flags = [f"{name}={value}" for name, value in KANIKO_FEATURE_FLAGS.items()]
        for sandbox, image in zip(api.sandboxes, ["img-a", "img-b"], strict=True):
            clear, build = sandbox.commands
            assert clear.startswith("rm -rf") and f"/nhx-work/out/{image}" in clear
            assert build.split()[: len(flags) + 2] == [*flags, DROPCAPS, "/kaniko/executor"]
            assert f"--oci-layout-path=/nhx-work/out/{image}" in build and "--no-push" in build

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
        assert not any("--oci-layout-path=/nhx-work/out/unclearable" in c for c in api.commands)

    def test_a_command_that_fails_costs_only_its_own_image(self) -> None:
        api = FakeApi({"img-a": 0, "img-b": 0}, fail_on="--oci-layout-path=/nhx-work/out/img-b")
        assert _provider(api).build(0, _group("img-a", "img-b")) == {"img-a": 0}

    def test_no_sandbox_is_created_once_the_groups_hour_is_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(opensandbox_sandbox, "BUILD_TIMEOUT_SECONDS", 0)
        api = FakeApi({"img-a": 0})
        assert _provider(api).build(0, _group("img-a")) == {}
        assert api.created == []

    def test_output_is_logged_a_line_at_a_time(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=opensandbox_sandbox.__name__):
            _provider(FakeApi({"img-a": 0})).build(0, _group("img-a"))
        messages = [record.getMessage() for record in caplog.records]
        assert "  building img-a" in messages and "  step 2" in messages


class TestTheSandboxAlwaysGoes:
    def test_after_building(self) -> None:
        api = FakeApi({"img-a": 0, "img-b": 0})
        _provider(api).build(0, _group("img-a", "img-b"))
        assert [sandbox.destroyed for sandbox in api.sandboxes] == [True, True]

    def test_after_a_command_fails(self) -> None:
        api = FakeApi({"img-a": 0}, fail_on="/kaniko/executor")
        assert _provider(api).build(0, _group("img-a")) == {}
        assert api.sandboxes[0].destroyed

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
    """An earlier attempt's sandboxes build into this attempt's outputs, so it must not start until those are gone."""

    def test_it_kills_what_this_job_left(self) -> None:
        api = FakeApi(existing={"sbx-old": job_key("default", "abc"), "sbx-other": job_key("default", "other")})
        _provider(api).sweep()
        assert api.killed == ["sbx-old"]

    def test_nothing_to_sweep_kills_nothing(self) -> None:
        api = FakeApi()
        _provider(api).sweep()
        assert api.killed == []

    def test_a_sandbox_that_cannot_be_killed_but_is_gone_does_not_stop_the_step(self) -> None:
        """It may have ended at its deadline between being listed and killed."""

        class GoneBeforeItsKill(FakeApi):
            def kill(self, sandbox_id: str) -> None:
                del self._existing[sandbox_id]
                super().kill(sandbox_id)

        api = GoneBeforeItsKill(existing={"sbx-old": job_key("default", "abc")}, kill_fails=True)
        _provider(api).sweep()
        assert api.killed == ["sbx-old"]

    def test_a_sandbox_that_will_not_go_stops_the_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = FakeApi(existing={"sbx-old": job_key("default", "abc")}, killed_stay=True)
        monkeypatch.setattr(opensandbox_sandbox, "SWEEP_TIMEOUT_SECONDS", 0)
        with pytest.raises(RuntimeError, match="still there: sbx-old"):
            _provider(api).sweep()


class _Secrets:
    def __init__(self, data: dict[str, str] | None) -> None:
        self.read: list[tuple[str, str]] = []
        self._data = data

    def read_namespaced_secret(self, *, name: str, namespace: str) -> SimpleNamespace:
        self.read.append((name, namespace))
        return SimpleNamespace(data=self._data)


class TestTheKey:
    """Read with this step's ServiceAccount, never through the step's env."""

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
