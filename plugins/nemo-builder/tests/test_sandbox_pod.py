# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sandbox pod spec, asserted as a document."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from kubernetes import client as k8s
from kubernetes.client.exceptions import ApiException
from nemo_builder_plugin.run import pod_sandbox, sandbox
from nemo_builder_plugin.run.pod_sandbox import (
    KANIKO_CAPABILITIES,
    RESULT_MARKER,
    KubernetesPodProvider,
    _build_script,
    _pod_manifest,
    _results_from_log,
    sandbox_pod_name,
)
from nemo_builder_plugin.run.sandbox import BUILD_TIMEOUT_SECONDS, JOB_LABEL, SANDBOX_ROOT, job_key, sandbox_labels
from nemo_builder_plugin.run.supervise import _build_group, _exit_code
from nemo_builder_plugin.steps import ContextSource, SandboxGroup, SandboxImage, SandboxSpec, WorkLayout


def _sandbox(**overrides: object) -> SandboxSpec:
    base: dict[str, object] = {
        "image": "nhx-kaniko:test",
        "work_pvc": "nhx-build-work",
        "node_selector": {"nhx.nvidia.com/build-node": "true"},
        "dns_nameservers": ["8.8.8.8", "1.1.1.1"],
        "cpu": "2",
        "memory": "8Gi",
    }
    base.update(overrides)
    return SandboxSpec.model_validate(base)


def _group(n: int = 2, *, source: ContextSource | None = None) -> SandboxGroup:
    return SandboxGroup(
        source=source or ContextSource(fileset="ws/fs-a"),
        images=[SandboxImage(image=f"demo-1-{i}", platform="linux/amd64", dockerfile="Dockerfile") for i in range(n)],
    )


def _pod():
    return _pod_manifest(
        name="nhx-sbx-abc-g0",
        namespace="nhx-builds",
        group=_group(),
        sandbox=_sandbox(),
        labels=sandbox_labels("default", "abc"),
        job_sub_path="jobs/default/abc",
    )


class TestTheSandboxHoldsNothing:
    def test_no_service_account_token(self) -> None:
        assert _pod().spec.automount_service_account_token is False

    def test_no_environment_and_no_secret_volume(self) -> None:
        container = _pod().spec.containers[0]
        assert not container.env
        assert not container.env_from
        # Nor the `<SERVICE>_SERVICE_HOST` variables the kubelet injects by default.
        assert _pod().spec.enable_service_links is False
        assert [v.name for v in _pod().spec.volumes] == ["work"]
        assert _pod().spec.volumes[0].persistent_volume_claim is not None

    def test_it_ends_even_if_its_supervisor_does_not(self) -> None:
        assert _pod().spec.active_deadline_seconds > BUILD_TIMEOUT_SECONDS

    def test_it_wears_the_sandbox_label(self) -> None:
        """What a NetworkPolicy selects on; without it the policy silently does not apply."""
        assert _pod().metadata.labels["nhx.nvidia.com/sandbox"] == "true"

    def test_it_names_its_job_for_a_later_attempt_to_find(self) -> None:
        assert _pod().metadata.labels[JOB_LABEL] == job_key("default", "abc")

    def test_the_same_job_name_in_two_workspaces_is_two_jobs(self) -> None:
        assert job_key("team-a", "demo-1") != job_key("team-b", "demo-1")


class TestThePostureTheNamespaceAdmits:
    def test_seccomp_is_runtimedefault_not_unconfined(self) -> None:
        ctx = _pod().spec.containers[0].security_context
        assert ctx.seccomp_profile.type == "RuntimeDefault"
        assert ctx.allow_privilege_escalation is False

    def test_root_but_not_the_stock_root(self) -> None:
        ctx = _pod().spec.containers[0].security_context
        assert ctx.run_as_user == 0
        assert ctx.capabilities.drop == ["ALL"]
        assert ctx.capabilities.add == KANIKO_CAPABILITIES
        assert "NET_RAW" not in ctx.capabilities.add
        assert "MKNOD" not in ctx.capabilities.add


class TestMounts:
    def test_context_is_read_only_and_outputs_are_not(self) -> None:
        mounts = {m.mount_path: m for m in _pod().spec.containers[0].volume_mounts}
        assert mounts["/nhx-work/context/ws/fs-a"].read_only is True
        assert not mounts["/nhx-work/out/demo-1-0"].read_only
        assert not mounts["/nhx-work/out/demo-1-1"].read_only

    def test_subpaths_are_scoped_to_this_job_and_this_group(self) -> None:
        mounts = {m.mount_path: m for m in _pod().spec.containers[0].volume_mounts}
        assert mounts["/nhx-work/context/ws/fs-a"].sub_path == "jobs/default/abc/context/ws/fs-a"
        assert mounts["/nhx-work/out/demo-1-0"].sub_path == "jobs/default/abc/out/demo-1-0"

    def test_only_this_groups_outputs_are_mounted(self) -> None:
        mounts = {m.mount_path for m in _pod().spec.containers[0].volume_mounts}
        assert "/nhx-work/out" not in mounts
        assert {m for m in mounts if m.startswith("/nhx-work/out/")} == {
            "/nhx-work/out/demo-1-0",
            "/nhx-work/out/demo-1-1",
        }

    def test_each_mount_is_the_same_layout_path_under_the_sandbox_root(self) -> None:
        """A subtree context is where a second path scheme would show."""
        pod = _pod_manifest(
            name="nhx-sbx-abc-g0",
            namespace="nhx-builds",
            group=_group(source=ContextSource(fileset="ws/fs-a", context_path="env/tests")),
            sandbox=_sandbox(),
            labels=sandbox_labels("default", "abc"),
            job_sub_path="jobs/default/abc",
        )
        for mount in pod.spec.containers[0].volume_mounts:
            relative = PurePosixPath(mount.sub_path).relative_to("jobs/default/abc")
            assert PurePosixPath(mount.mount_path) == SANDBOX_ROOT / relative

    def test_the_script_builds_from_and_writes_to_the_mounted_paths(self) -> None:
        script = _build_script(_group(1))
        mounts = {m.mount_path for m in _pod().spec.containers[0].volume_mounts}
        assert f"--context=dir://{WorkLayout(SANDBOX_ROOT).context(ContextSource(fileset='ws/fs-a'))}" in script
        assert "--oci-layout-path=/nhx-work/out/demo-1-0" in script
        assert {"/nhx-work/context/ws/fs-a", "/nhx-work/out/demo-1-0", "/nhx-work/out/demo-1-1"} == mounts


class TestPodNames:
    def test_the_same_job_name_in_two_workspaces_is_two_pods(self) -> None:
        assert sandbox_pod_name("team-a", "demo-1", 0) != sandbox_pod_name("team-b", "demo-1", 0)

    def test_a_long_job_name_still_makes_a_legal_pod_name(self) -> None:
        name = sandbox_pod_name("default", "a" * 60 + "-1", 12)
        assert len(name) <= 63
        assert name.endswith("-g12") and "--" not in name


class TestDns:
    def test_public_resolvers_not_cluster_dns(self) -> None:
        spec = _pod().spec
        assert spec.dns_policy == "None"
        assert spec.dns_config.nameservers == ["8.8.8.8", "1.1.1.1"]


class TestTheBuildScript:
    def test_it_never_pushes(self) -> None:
        script = _build_script(_group())
        assert "--no-push" in script
        assert "crane" not in script and "cosign" not in script

    def test_one_invocation_per_image_with_cleanup_between(self) -> None:
        script = _build_script(_group(3))
        assert script.count("/kaniko/executor") == 3
        assert script.count("--cleanup") == 3


class TestTheBuildScriptRuns:
    @staticmethod
    def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dockerfiles: list[str]) -> dict[str, int]:
        executor = tmp_path / "executor"
        # Exits 3 for a "broken*" Dockerfile, else 0: not 1, so only kaniko's own status matches. For an
        # "unterminated*" one, its output ends without a newline, as a `RUN printf done` does.
        executor.write_text(
            '#!/bin/sh\nfor arg in "$@"; do case "$arg" in\n'
            "  --dockerfile=broken*) exit 3;;\n"
            "  --dockerfile=unterminated*) printf done; exit 0;;\n"
            "esac; done\nexit 0\n"
        )
        executor.chmod(0o755)
        monkeypatch.setattr(sandbox, "KANIKO_EXECUTOR", str(executor))
        # The script empties each image's output directory; point it somewhere of the test's own.
        monkeypatch.setattr(sandbox, "SANDBOX_ROOT", PurePosixPath(tmp_path))

        group = SandboxGroup(
            source=ContextSource(fileset="ws/fs-a"),
            images=[
                SandboxImage(image=f"demo-1-{i}", platform="linux/amd64", dockerfile=dockerfile)
                for i, dockerfile in enumerate(dockerfiles)
            ],
        )
        result = subprocess.run(["sh", "-c", _build_script(group)], capture_output=True, text=True, check=False)
        return _results_from_log(result.stdout)

    def test_each_marker_records_kanikos_own_exit_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        results = self._run(tmp_path, monkeypatch, ["Dockerfile", "broken.Dockerfile", "Dockerfile"])
        assert results == {"demo-1-0": 0, "demo-1-1": 3, "demo-1-2": 0}

    def test_an_earlier_attempts_layout_is_gone_before_the_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = tmp_path / "out" / "demo-1-0"
        stale.mkdir(parents=True)
        for name in ("index.json", "oci-layout", ".hidden"):
            (stale / name).write_text("{}")
        (stale / "blobs").mkdir()
        results = self._run(tmp_path, monkeypatch, ["broken.Dockerfile"])
        assert results == {"demo-1-0": 3}
        assert stale.is_dir() and list(stale.iterdir()) == []

    def test_a_failing_image_does_not_abort_the_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        results = self._run(tmp_path, monkeypatch, ["broken.Dockerfile", "Dockerfile"])
        assert results == {"demo-1-0": 3, "demo-1-1": 0}

    def test_output_without_a_final_newline_does_not_hide_the_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = self._run(tmp_path, monkeypatch, ["unterminated.Dockerfile", "unterminated.Dockerfile"])
        assert results == {"demo-1-0": 0, "demo-1-1": 0}


class _Api:
    """The pod calls a sandbox makes, failing where told to. Deleted pods are gone at once."""

    def __init__(
        self,
        *,
        create: Exception | None = None,
        delete: Exception | None = None,
        existing: dict[str, str] | None = None,
    ) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []
        self._create, self._delete = create, delete
        #: Pod name -> its job label.
        self.pods = dict(existing or {})

    def create_namespaced_pod(self, *, namespace: str, body: k8s.V1Pod) -> None:
        if self._create:
            raise self._create
        self.created.append(body.metadata.name)

    def delete_namespaced_pod(self, *, name: str, namespace: str, grace_period_seconds: int) -> None:
        self.deleted.append(name)
        if self._delete:
            raise self._delete
        self.pods.pop(name, None)

    def list_namespaced_pod(self, *, namespace: str, label_selector: str) -> k8s.V1PodList:
        key, _, value = label_selector.partition("=")
        assert key == JOB_LABEL
        items = [k8s.V1Pod(metadata=k8s.V1ObjectMeta(name=n)) for n, job in self.pods.items() if job == value]
        return k8s.V1PodList(items=items)


def _provider(api: _Api) -> KubernetesPodProvider:
    return KubernetesPodProvider(
        cast(k8s.CoreV1Api, api),
        namespace="nhx-builds",
        sandbox=_sandbox(),
        workspace="default",
        job_id="abc",
        job_sub_path="jobs/default/abc",
    )


class TestEachSandboxFailsAlone:
    NAME = sandbox_pod_name("default", "abc", 0)
    BUILT = f"{RESULT_MARKER} demo-1-0 0\n{RESULT_MARKER} demo-1-1 0\n"

    def _build(
        self, monkeypatch: pytest.MonkeyPatch, api: _Api, *, log: str = BUILT, read: Exception | None = None
    ) -> int:
        def read_log(api: object, *, name: str, namespace: str) -> str:
            if read:
                raise read
            return log

        monkeypatch.setattr(pod_sandbox, "_await_pod", lambda api, *, name, namespace: "Succeeded")
        monkeypatch.setattr(pod_sandbox, "_read_pod_log", read_log)
        return _build_group(_provider(api), 0, _group(2))

    def test_a_sandbox_that_cannot_be_created_fails_only_its_own_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = _Api(create=ApiException(status=403, reason="exceeded quota"))
        assert self._build(monkeypatch, api) == 2
        assert api.deleted == []

    def test_a_log_that_cannot_be_read_fails_the_group_and_the_sandbox_is_still_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = _Api()
        assert self._build(monkeypatch, api, read=ApiException(status=400, reason="container is waiting")) == 2
        assert api.deleted == [self.NAME]

    def test_a_sandbox_that_cannot_be_deleted_keeps_what_it_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = _Api(delete=ApiException(status=404, reason="Not Found"))
        assert self._build(monkeypatch, api) == 0
        assert api.deleted == [self.NAME]

    def test_the_log_is_logged_a_line_at_a_time(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=pod_sandbox.__name__):
            self._build(monkeypatch, _Api(), log="STEP 1/2\nSTEP 2/2\n" + self.BUILT)
        messages = [record.getMessage() for record in caplog.records]
        assert "  STEP 1/2" in messages and "  STEP 2/2" in messages


class TestTheSweep:
    """A retried step's pods have the earlier attempt's names, so it must not start until those are gone."""

    def test_an_earlier_attempts_pods_are_deleted(self) -> None:
        api = _Api(existing={"left-g0": job_key("default", "abc"), "other-g0": job_key("default", "other")})
        _provider(api).sweep()
        assert api.deleted == ["left-g0"]
        assert list(api.pods) == ["other-g0"]

    def test_nothing_to_sweep_deletes_nothing(self) -> None:
        api = _Api()
        _provider(api).sweep()
        assert api.deleted == []

    def test_a_pod_that_will_not_go_stops_the_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = _Api(existing={"left-g0": job_key("default", "abc")}, delete=ApiException(status=500))
        monkeypatch.setattr(pod_sandbox, "SWEEP_TIMEOUT_SECONDS", 0)
        with pytest.raises(RuntimeError, match="still there: left-g0"):
            _provider(api).sweep()


class TestResultParsing:
    def test_reads_per_image_exit_codes_from_the_log(self) -> None:
        log = f"noise\n{RESULT_MARKER} demo-1-0 0\nmore noise\n{RESULT_MARKER} demo-1-1 1\n"
        assert _results_from_log(log) == {"demo-1-0": 0, "demo-1-1": 1}

    def test_reads_a_bytes_log_too(self) -> None:
        log = f"\x1b[36mINFO\x1b[0m noise\n{RESULT_MARKER} demo-1-0 0\n".encode()
        assert _results_from_log(log) == {"demo-1-0": 0}

    def test_undecodable_bytes_do_not_lose_the_verdict(self) -> None:
        log = b"\xff\xfe garbage\n" + f"{RESULT_MARKER} demo-1-0 0\n".encode()
        assert _results_from_log(log) == {"demo-1-0": 0}

    def test_a_bytes_repr_string_finds_nothing_which_is_why_the_source_is_fixed(self) -> None:
        """The client's `str` for a non-UTF-8 log is a bytes repr; `_read_pod_log` avoids it."""
        broken = "b'" + f"noise\\n{RESULT_MARKER} demo-1-0 0\\n" + "'"
        assert "\n" not in broken.replace("\\n", "")
        assert _results_from_log(broken) == {}

    def test_ignores_lines_that_only_look_like_markers(self) -> None:
        log = f"{RESULT_MARKER} demo-1-0 notanumber\n{RESULT_MARKER} oops\n"
        assert _results_from_log(log) == {}


class TestExitCode:
    def test_a_partial_failure_still_lets_push_run(self) -> None:
        assert _exit_code(failures=1, total=10) == 0

    def test_a_clean_set_succeeds(self) -> None:
        assert _exit_code(failures=0, total=3) == 0

    def test_a_set_where_nothing_built_fails_the_step(self) -> None:
        assert _exit_code(failures=3, total=3) == 1

    def test_a_single_image_set_that_failed_fails_the_step(self) -> None:
        assert _exit_code(failures=1, total=1) == 1
