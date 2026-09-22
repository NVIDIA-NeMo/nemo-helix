# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sandbox pod spec, asserted as a document.

`deploy/sandbox-egress-probe.sh` proves the NetworkPolicy closes what it should, with packets.
This file proves the pod composed at runtime is the one that policy applies to, and carries the
posture the namespace admits -- neither of which a packet can tell you.
"""

from __future__ import annotations

from nemo_builder_plugin.run.supervise import (
    KANIKO_CAPABILITIES,
    RESULT_MARKER,
    _build_script,
    _exit_code,
    _pod_manifest,
    _results_from_log,
)
from nemo_builder_plugin.steps import SandboxGroup, SandboxImage, SandboxSpec


def _sandbox(**overrides: object) -> SandboxSpec:
    base: dict[str, object] = {
        "image": "gcr.io/kaniko-project/executor:debug",
        "namespace": "nmp-builds",
        "work_pvc": "nmp-build-work",
        "node_selector": {"nmp.nvidia.com/build-node": "true"},
    }
    base.update(overrides)
    return SandboxSpec.model_validate(base)


def _group(n: int = 2) -> SandboxGroup:
    return SandboxGroup(
        context_sub_path="context/fs-a",
        output_sub_path="out",
        images=[
            SandboxImage(
                image=f"demo-1-{i}",
                platform="linux/amd64",
                context="/ctx",
                dockerfile="Dockerfile",
                layout=f"/out/demo-1-{i}",
            )
            for i in range(n)
        ],
    )


def _pod():
    return _pod_manifest(
        name="nmp-sbx-abc-g0",
        group=_group(),
        sandbox=_sandbox(),
        pvc="nmp-build-work",
        job_sub_path="jobs/default/abc",
    )


class TestTheSandboxHoldsNothing:
    def test_no_service_account_token(self) -> None:
        """Closes threat-model path C: even if it reached Files or Jobs it presents no identity."""
        assert _pod().spec.automount_service_account_token is False

    def test_no_environment_and_no_secret_volume(self) -> None:
        """`RUN cat` has nothing to find. The credential is gone, not merely unreadable."""
        container = _pod().spec.containers[0]
        assert not container.env
        assert not container.env_from
        assert [v.name for v in _pod().spec.volumes] == ["work"]
        assert _pod().spec.volumes[0].persistent_volume_claim is not None

    def test_it_wears_the_label_the_networkpolicy_selects_on(self) -> None:
        """Without this label the policy does not apply and the pod is unconstrained -- which is
        a silent failure, because the build still succeeds."""
        assert _pod().metadata.labels["nmp.nvidia.com/sandbox"] == "true"


class TestThePostureTheNamespaceAdmits:
    def test_seccomp_is_runtimedefault_not_unconfined(self) -> None:
        """`Unconfined` is exactly what deploy/negative-control.sh proves is refused."""
        ctx = _pod().spec.containers[0].security_context
        assert ctx.seccomp_profile.type == "RuntimeDefault"
        assert ctx.allow_privilege_escalation is False

    def test_root_but_not_the_stock_root(self) -> None:
        """uid 0 is load-bearing -- unpacking layers means owning files across many uids -- but
        nine of containerd's fourteen capabilities are dropped, including NET_RAW and MKNOD."""
        ctx = _pod().spec.containers[0].security_context
        assert ctx.run_as_user == 0
        assert ctx.capabilities.drop == ["ALL"]
        assert ctx.capabilities.add == KANIKO_CAPABILITIES
        assert "NET_RAW" not in ctx.capabilities.add
        assert "MKNOD" not in ctx.capabilities.add


class TestMounts:
    def test_context_is_read_only_and_output_is_not(self) -> None:
        mounts = {m.mount_path: m for m in _pod().spec.containers[0].volume_mounts}
        assert mounts["/ctx"].read_only is True
        assert not mounts["/out"].read_only

    def test_subpaths_are_scoped_to_this_job_and_this_group(self) -> None:
        """A Dockerfile sees its own context and no other source in the set, and no other job."""
        mounts = {m.mount_path: m for m in _pod().spec.containers[0].volume_mounts}
        assert mounts["/ctx"].sub_path == "jobs/default/abc/context/fs-a"
        assert mounts["/out"].sub_path == "jobs/default/abc/out"


class TestDns:
    def test_public_resolvers_not_cluster_dns(self) -> None:
        """Cluster DNS here answers on a link-local address the egress policy denies, and
        re-allowing it reopens the Pod CIDR. Measured; see deploy/README.md."""
        spec = _pod().spec
        assert spec.dns_policy == "None"
        assert spec.dns_config.nameservers == ["8.8.8.8", "1.1.1.1"]


class TestTheBuildScript:
    def test_it_never_pushes(self) -> None:
        """What makes the sandbox credential-free: it writes a layout, it does not publish."""
        script = _build_script(_group(), _sandbox())
        assert "--no-push" in script
        assert "crane" not in script and "cosign" not in script

    def test_one_invocation_per_image_with_cleanup_between(self) -> None:
        """They share one container root and measurably cannot overlap."""
        script = _build_script(_group(3), _sandbox())
        assert script.count("/kaniko/executor") == 3
        assert script.count("--cleanup") == 3

    def test_a_failing_image_does_not_abort_the_set(self) -> None:
        """One broken Dockerfile in a set of ten must not cost the other nine -- which is also
        why the reconciler asks the registry even when the job exited non-zero."""
        script = _build_script(_group(2), _sandbox())
        assert script.count("|| true") == 2
        assert script.count(f'echo "{RESULT_MARKER}') == 2

    def test_a_mirror_disables_fallback_past_it(self) -> None:
        """Without this, a mirror is decorative: with public egress available the fallback would
        succeed silently and no error would appear anywhere."""
        script = _build_script(_group(1), _sandbox(registry_mirror="mirror.nmp-builds.svc"))
        assert "--registry-mirror=mirror.nmp-builds.svc" in script
        assert "--skip-default-registry-fallback" in script

    def test_no_mirror_means_no_fallback_flag(self) -> None:
        assert "--skip-default-registry-fallback" not in _build_script(_group(1), _sandbox())


class TestResultParsing:
    def test_reads_per_image_exit_codes_from_the_log(self) -> None:
        """The log is all `supervise` gets: it mounts no volume, so it cannot look at the output."""
        log = f"noise\n{RESULT_MARKER} demo-1-0 0\nmore noise\n{RESULT_MARKER} demo-1-1 1\n"
        assert _results_from_log(log) == {"demo-1-0": 0, "demo-1-1": 1}

    def test_reads_a_bytes_log_too(self) -> None:
        """The Kubernetes client returns BYTES when the log is not valid UTF-8, and kaniko's
        output never is -- it is ANSI-coloured. A `str` marker compared against `bytes` lines
        matches nothing and raises nothing, so every image reported "no result recorded" while
        every build had actually succeeded. Found end-to-end; unit tests fed it a `str`."""
        log = f"\x1b[36mINFO\x1b[0m noise\n{RESULT_MARKER} demo-1-0 0\n".encode()
        assert _results_from_log(log) == {"demo-1-0": 0}

    def test_undecodable_bytes_do_not_lose_the_verdict(self) -> None:
        log = b"\xff\xfe garbage\n" + f"{RESULT_MARKER} demo-1-0 0\n".encode()
        assert _results_from_log(log) == {"demo-1-0": 0}

    def test_a_bytes_repr_string_finds_nothing_which_is_why_the_source_is_fixed(self) -> None:
        """This is the shape that broke it, kept as a regression witness.

        When the Kubernetes client deserializes a non-UTF-8 log body into its declared `str`
        return type, the result is the REPR of a bytes object: newlines are literal backslash-n.
        `splitlines()` yields one line and every marker disappears -- silently, because nothing
        raises. `_read_pod_log` avoids producing this at all by decoding the raw body itself.
        """
        broken = "b'" + f"noise\\n{RESULT_MARKER} demo-1-0 0\\n" + "'"
        assert "\n" not in broken.replace("\\n", "")  # the newlines really are escaped
        assert _results_from_log(broken) == {}

    def test_ignores_lines_that_only_look_like_markers(self) -> None:
        """A Dockerfile can print anything it likes into this log."""
        log = f"{RESULT_MARKER} demo-1-0 notanumber\n{RESULT_MARKER} oops\n"
        assert _results_from_log(log) == {}


class TestExitCode:
    """The exit code is a scheduling decision, not a report.

    The Jobs dispatcher schedules the next step only when this one is COMPLETED, so a non-zero
    exit here means `push` never runs for ANY image in the set. Found by adversarial review: the
    original returned 1 on any failure, so one broken Dockerfile published nothing at all.
    """

    def test_a_partial_failure_still_lets_push_run(self) -> None:
        assert _exit_code(failures=1, total=10) == 0

    def test_a_clean_set_succeeds(self) -> None:
        assert _exit_code(failures=0, total=3) == 0

    def test_a_set_where_nothing_built_fails_the_step(self) -> None:
        """Nothing to publish, so there is no reason to run push -- and the job should say so."""
        assert _exit_code(failures=3, total=3) == 1

    def test_a_single_image_set_that_failed_fails_the_step(self) -> None:
        assert _exit_code(failures=1, total=1) == 1
