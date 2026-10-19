# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PATCH = Path(__file__).parents[1] / "harbor-patches/patch_sandbox_k8s_root.py"
UPSTREAM_VALIDATOR = (
    '    @field_validator("run_as_user")\n'
    "    @classmethod\n"
    "    def _reject_root_user(cls, v: int | None) -> int | None:\n"
    "        if v is not None and v == 0:\n"
    '            raise ValueError("run_as_user must not be 0 (root). Sandboxes must not '
    'run as root; use a non-zero UID (e.g. 1000).")\n'
    "        return v\n"
)
CLIENT_SOURCE = (
    """from typing import Any
import os
import shlex
import time

class K8sSandboxClient:
    def __init__(
        self,
        kubeconfig_path: str | None = None,
        context: str | None = None,
        in_cluster: bool = False,
        verify_ssl: bool = True,
        astra_chamber: str | None = None,
    ) -> None:
        self._astra_chamber: str | None = normalize_astra_chamber(astra_chamber)
        self._api_client = self._load_config(kubeconfig_path, context, in_cluster, verify_ssl)

    @staticmethod
    def _drain_exec_stream(resp: Any, timeout: float | None) -> tuple[str, str, bool]:
        stdout_data = ""
        stderr_data = ""
        poll_sec = 1.0
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        timed_out = False

        while resp.is_open():
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            resp.update(timeout=poll_sec)

        return stdout_data, stderr_data, timed_out

    def wait_for_sandbox_ready(self, name, namespace="default"):
        while True:
            try:
                for event in w.stream():
                    obj = event.get("object", {})
                    event_type = event.get("type", "UNKNOWN")

                    if event_type == "DELETED":
                        raise SandboxCreationError("deleted")

                    if event_type in ("ADDED", "MODIFIED"):
                        info = self._parse_sandbox_response(obj)
                        result = self._check_sandbox_status(info, namespace, name)
                        if result is not None:
                            w.stop()
                            return result

            except ApiException as e:
                raise

    def _exec_in_pod(
        self, pod_name, command, namespace="default", container=None,
        workdir=None, timeout=None, stdin_data=None
    ):
        exec_command = self._prepare_exec_command(command, workdir)
        try:
            pass
        except ApiException as e:
            if e.status == 404:
                raise SandboxNotFoundError(
"""
    '                    f"Pod {namespace}/{pod_name} not found '
    '(sandbox may have been deleted or timed out)",\n'
    """                    **_api_exc_fields(e),
                ) from e
            logger.error("Failed to exec command in %s/%s: %s", namespace, pod_name, e)
            raise SandboxExecutionError(
                f"Failed to exec command in {namespace}/{pod_name}: {e.reason}",
                **_api_exc_fields(e),
            ) from e

    @staticmethod
    def _build_security_context(spec: Any) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        }
        return ctx

    @staticmethod
    def _build_container_manifest(container: Any) -> dict[str, Any]:
        return {
            "securityContext": {
                "capabilities": {"drop": ["ALL"]},
                "readOnlyRootFilesystem": True,
            }
        }

    def _build_pod_spec_from_high_level(self, spec: Any) -> dict[str, Any]:
        pod_spec: dict[str, Any] = {
            "containers": [
                self._build_container_manifest(container)
                for container in [spec.container, *spec.sidecars]
            ],
            "serviceAccountName": spec.service_account_name,
            "automountServiceAccountToken": False,
            "securityContext": self._build_security_context(spec),
        }
        if spec.runtime_class_name:
            pod_spec["runtimeClassName"] = spec.runtime_class_name
        return pod_spec
"""
)
CONTAINER_SOURCE = """class ContainerSpec:
    volume_mounts: list[VolumeMount] = Field(
        default_factory=list,
        description="Volume mounts for this container",
    )

"""
SANDBOX_SOURCE = """import os

class CRDBackend:
    def __init__(
        self,
        *,
        network_policy: NetworkPolicy | dict[str, Any] | None = None,
        image_pull_policy: ImagePullPolicy = "IfNotPresent",
        run_as_user=None,
    ) -> None:
        self._image_pull_policy = image_pull_policy

    async def create(self):
        container = ContainerSpec(
                resources=self._resources,
        )

class K8sSandbox:
    def __init__(
        self,
        *,
        # Common settings
        image: str = DEFAULT_IMAGE,
        image_pull_policy: ImagePullPolicy = "IfNotPresent",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._backend = CRDBackend(
                image_pull_policy=image_pull_policy,
        )
"""


def _write_upstream_package(package_dir: Path, *, validator: str = UPSTREAM_VALIDATOR) -> None:
    (package_dir / "types.py").write_text(
        "from enum import Enum\n\n"
        f"{CONTAINER_SOURCE}class SandboxSpec:\n{validator}\n"
        f"class SandboxTemplateSpec:\n{validator}",
        encoding="utf-8",
    )
    (package_dir / "client.py").write_text(CLIENT_SOURCE, encoding="utf-8")
    (package_dir / "sandbox.py").write_text(SANDBOX_SOURCE, encoding="utf-8")


def _load_patch() -> ModuleType:
    spec = importlib.util.spec_from_file_location("patch_sandbox_k8s_root", PATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_patch_matches_published_sandbox_k8s_0_1_13(tmp_path: Path) -> None:
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    _write_upstream_package(package_dir)

    _load_patch().patch(package_dir)

    types_source = (package_dir / "types.py").read_text(encoding="utf-8")
    client_source = (package_dir / "client.py").read_text(encoding="utf-8")
    sandbox_source = (package_dir / "sandbox.py").read_text(encoding="utf-8")
    compile(client_source, str(package_dir / "client.py"), "exec")
    compile(sandbox_source, str(package_dir / "sandbox.py"), "exec")
    assert types_source.count("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED") == 2
    assert 'runAsNonRoot": not root_authorized' in client_source
    assert '"readOnlyRootFilesystem": container.read_only_root_filesystem' in client_source
    assert '"CHOWN"' in client_source
    assert '"DAC_OVERRIDE"' in client_source
    assert '"FOWNER"' in client_source
    assert '"KILL"' in client_source
    assert '"SETGID"' in client_source
    assert '"SETUID"' in client_source
    # A multi-service task addresses its services by their compose names, and
    # every container in the pod shares one network namespace, so each sidecar
    # name has to resolve to loopback or a service waiting on a sibling hangs.
    assert '"hostnames": [sidecar.name for sidecar in spec.sidecars]' in client_source
    assert "self._read_only_root_filesystem = read_only_root_filesystem" in sandbox_source
    sandbox_tree = ast.parse(sandbox_source)
    public_sandbox = next(
        node for node in sandbox_tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandbox"
    )
    constructor = next(
        node for node in public_sandbox.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    keyword_only_args = {argument.arg for argument in constructor.args.kwonlyargs}
    assert "read_only_root_filesystem" in keyword_only_args
    assert "_scaled_evals_writable_root_authorized" in keyword_only_args
    backend_call = next(
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CRDBackend"
    )
    forwarded_keywords = {keyword.arg for keyword in backend_call.keywords}
    assert "read_only_root_filesystem" in forwarded_keywords
    assert "_scaled_evals_writable_root_authorized" in forwarded_keywords
    assert "def _exec_in_pod_with_kubectl(" in client_source
    assert "used kubectl fallback" in client_source
    assert "def _prefer_kubectl_exec(" in client_source
    assert "self._kubeconfig_path = kubeconfig_path" in client_source
    assert "self._context = context" in client_source
    assert "SANDBOX_K8S_PREFER_NATIVE_EXEC" in client_source
    assert "SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS" in client_source
    assert '"name": "net.ipv4.ip_unprivileged_port_start"' in client_source
    assert '"value": "0"' in client_source
    assert "SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK" in client_source
    for name in (
        "net.ipv6.conf.all.disable_ipv6",
        "net.ipv6.conf.default.disable_ipv6",
        "net.ipv6.conf.lo.disable_ipv6",
    ):
        assert f'"name": "{name}"' in client_source
    assert "selected_context = context or os.environ.get" in client_source
    assert "self._prefer_kubectl_exec(self._context)" in client_source
    assert "Using native kubectl exec" in client_source
    assert "info = self.get_sandbox(name, namespace)" in client_source
    assert "attempts = 10" in client_source
    assert "import shutil" in client_source
    assert "import subprocess" in client_source
    assert "from websocket import WebSocketConnectionClosedException" in client_source
    assert 'resp.sock.ping("scaled-evals-keepalive")' in client_source
    assert "keepalive_sec = 30.0" in client_source
    assert "except WebSocketConnectionClosedException as e:" in client_source
    assert "raise SandboxExecutionError(" in client_source
    assert "Kubernetes exec stream disconnected" in client_source


def test_capability_upgrade_matches_previous_root_overlay(tmp_path: Path) -> None:
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    (package_dir / "client.py").write_text(CLIENT_SOURCE, encoding="utf-8")

    _load_patch().patch_capabilities(package_dir)
    first_patch = (package_dir / "client.py").read_text(encoding="utf-8")
    _load_patch().patch_capabilities(package_dir)

    client_source = (package_dir / "client.py").read_text(encoding="utf-8")
    assert client_source == first_patch
    compile(client_source, str(package_dir / "client.py"), "exec")
    assert '"drop": ["ALL"]' in client_source
    assert '"add": [' in client_source
    assert '"DAC_OVERRIDE"' in client_source
    assert '"KILL"' in client_source
    assert 'container.name == "sandbox"' in client_source
    assert "not container.read_only_root_filesystem" in client_source
    assert "SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED" in client_source

    # A base image built before KILL joined the grant must be upgraded in place,
    # not rejected: the pristine manifest it was patched from is long gone.
    inherited = package_dir / "inherited"
    inherited.mkdir()
    (inherited / "client.py").write_text(
        client_source.replace('                                "KILL",\n', ""),
        encoding="utf-8",
    )
    _load_patch().patch_capabilities(inherited)
    upgraded = (inherited / "client.py").read_text(encoding="utf-8")
    assert upgraded == client_source
    _load_patch().patch_capabilities(inherited)
    assert (inherited / "client.py").read_text(encoding="utf-8") == client_source


def test_sysctl_upgrade_reaches_both_inherited_runner_shapes(tmp_path: Path) -> None:
    # `conf.all` is a broadcast write and runc applies the sysctl map in
    # randomised order, so a runner emitting `lo=0` alone restores `::1` only
    # sometimes. Both inherited shapes must reach the full grant: a base image
    # predating the IPv6 gate, and one carrying the earlier `lo`-only version.
    # Missing the second leaves a runner silently emitting the racy single key.
    module = _load_patch()
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    _write_upstream_package(package_dir)
    module.patch(package_dir)
    patched = (package_dir / "client.py").read_text(encoding="utf-8")
    assert module._SYSCTLS_GRANT in patched

    for label, stale in (
        ("pre-ipv6", module._SYSCTLS_LOW_PORTS_ONLY),
        ("lo-only", module._SYSCTLS_LO_ONLY),
    ):
        inherited = tmp_path / f"inherited-{label}"
        inherited.mkdir()
        (inherited / "client.py").write_text(patched.replace(module._SYSCTLS_GRANT, stale, 1), encoding="utf-8")
        module.patch_sysctls(inherited)
        upgraded = (inherited / "client.py").read_text(encoding="utf-8")
        assert upgraded == patched, label
        compile(upgraded, str(inherited / "client.py"), "exec")
        module.patch_sysctls(inherited)
        assert (inherited / "client.py").read_text(encoding="utf-8") == patched, label


def test_sysctl_upgrade_rejects_an_unrecognised_manifest(tmp_path: Path) -> None:
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    (package_dir / "client.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected inherited sandbox-k8s sysctl manifest"):
        _load_patch().patch_sysctls(package_dir)


def test_sidecar_host_alias_upgrade_reaches_an_inherited_runner(tmp_path: Path) -> None:
    # The image build patches a runner that inherited an already-patched
    # sandbox_k8s from the base image, so a step added after that base was built
    # has to apply here too -- and has to no-op once it has.
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    (package_dir / "client.py").write_text(CLIENT_SOURCE, encoding="utf-8")

    _load_patch().patch_sidecar_host_aliases(package_dir)
    first_patch = (package_dir / "client.py").read_text(encoding="utf-8")
    _load_patch().patch_sidecar_host_aliases(package_dir)

    client_source = (package_dir / "client.py").read_text(encoding="utf-8")
    assert client_source == first_patch
    compile(client_source, str(package_dir / "client.py"), "exec")
    assert '"hostnames": [sidecar.name for sidecar in spec.sidecars]' in client_source


def test_sidecar_capability_upgrade_reaches_an_inherited_runner(tmp_path: Path) -> None:
    # Same upgrade path as the host aliases: the image build patches a runner
    # that already carries an earlier version of the grant, so widening it to
    # sidecars has to work against that shape and no-op on a second pass.
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    _write_upstream_package(package_dir)
    module = _load_patch()
    module.patch_capabilities(package_dir)
    assert 'container.name == "sandbox"' in (package_dir / "client.py").read_text()

    module.patch_sidecar_capabilities(package_dir)
    widened = (package_dir / "client.py").read_text(encoding="utf-8")
    module.patch_sidecar_capabilities(package_dir)

    assert (package_dir / "client.py").read_text(encoding="utf-8") == widened
    compile(widened, str(package_dir / "client.py"), "exec")
    assert 'container.name == "sandbox"' not in widened
    assert "not container.read_only_root_filesystem" in widened


def test_root_patch_grants_capabilities_only_to_authorized_writable_root_container(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    _write_upstream_package(package_dir)
    _load_patch().patch(package_dir)

    tree = ast.parse((package_dir / "client.py").read_text(encoding="utf-8"))
    client = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    build_manifest = next(
        node for node in client.body if isinstance(node, ast.FunctionDef) and node.name == "_build_container_manifest"
    )
    isolated = ast.ClassDef(
        name="K8sSandboxClient",
        bases=[],
        keywords=[],
        body=[build_manifest],
        decorator_list=[],
        type_params=[],
    )
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {"Any": Any, "os": os}
    module = ast.Module(body=[isolated], type_ignores=[])
    exec(compile(module, "<patched-client>", "exec"), namespace)

    class Container:
        def __init__(self, *, name: str, read_only_root_filesystem: bool) -> None:
            self.name = name
            self.read_only_root_filesystem = read_only_root_filesystem

    read_only = namespace["K8sSandboxClient"]._build_container_manifest(
        Container(name="sandbox", read_only_root_filesystem=True)
    )
    unauthorized_writable = namespace["K8sSandboxClient"]._build_container_manifest(
        Container(name="sandbox", read_only_root_filesystem=False)
    )
    monkeypatch.setenv("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED", "true")
    writable_root = namespace["K8sSandboxClient"]._build_container_manifest(
        Container(name="sandbox", read_only_root_filesystem=False)
    )
    writable_sidecar = namespace["K8sSandboxClient"]._build_container_manifest(
        Container(name="postgres", read_only_root_filesystem=False)
    )
    read_only_sidecar = namespace["K8sSandboxClient"]._build_container_manifest(
        Container(name="postgres", read_only_root_filesystem=True)
    )

    grant = {
        "drop": ["ALL"],
        "add": ["CHOWN", "DAC_OVERRIDE", "FOWNER", "KILL", "SETGID", "SETUID"],
    }
    assert read_only["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert unauthorized_writable["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert writable_root["securityContext"]["capabilities"] == grant
    # A sidecar is held to the same two conditions as the task container and no
    # others: database images hand off with gosu and need SETUID/SETGID, so a
    # writable-root sidecar carries the grant, and a read-only one still cannot.
    assert writable_sidecar["securityContext"]["capabilities"] == grant
    assert read_only_sidecar["securityContext"]["capabilities"] == {"drop": ["ALL"]}


def test_root_patch_rejects_partial_upstream_shape(tmp_path: Path) -> None:
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    (package_dir / "types.py").write_text(
        f"from enum import Enum\n\nclass SandboxSpec:\n{UPSTREAM_VALIDATOR}",
        encoding="utf-8",
    )
    (package_dir / "client.py").write_text("", encoding="utf-8")

    try:
        _load_patch().patch(package_dir)
    except RuntimeError as exc:
        assert str(exc) == "expected sandbox-k8s 0.1.13 validators in both spec classes"
    else:
        raise AssertionError("partial sandbox-k8s source shape was unexpectedly patched")


def test_root_patch_sends_keepalive_ping_during_long_exec(tmp_path: Path) -> None:
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    _write_upstream_package(package_dir)
    _load_patch().patch(package_dir)

    tree = ast.parse((package_dir / "client.py").read_text(encoding="utf-8"))
    client = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    drain = next(
        node for node in client.body if isinstance(node, ast.FunctionDef) and node.name == "_drain_exec_stream"
    )
    isolated = ast.ClassDef(
        name="K8sSandboxClient",
        bases=[],
        keywords=[],
        body=[drain],
        decorator_list=[],
        type_params=[],
    )
    ast.fix_missing_locations(isolated)

    class Clock:
        values = iter((0.0, 0.0, 31.0))

        @classmethod
        def monotonic(cls) -> float:
            return next(cls.values)

    namespace: dict[str, Any] = {"Any": Any, "time": Clock}
    module = ast.Module(body=[isolated], type_ignores=[])
    exec(compile(module, "<patched-client>", "exec"), namespace)

    class Socket:
        def __init__(self) -> None:
            self.pings: list[str] = []

        def ping(self, payload: str) -> None:
            self.pings.append(payload)

    class Response:
        def __init__(self) -> None:
            self.sock = Socket()
            self.updates = 0

        def is_open(self) -> bool:
            return self.updates == 0

        def update(self, *, timeout: float) -> None:
            assert timeout == 1.0
            self.updates += 1

    response = Response()
    namespace["K8sSandboxClient"]._drain_exec_stream(response, timeout=60)

    assert response.sock.pings == ["scaled-evals-keepalive"]


def test_root_patch_accepts_formatting_only_validator_difference(tmp_path: Path) -> None:
    package_dir = tmp_path / "sandbox_k8s"
    package_dir.mkdir()
    wrapped_validator = UPSTREAM_VALIDATOR.replace(
        '            raise ValueError("run_as_user must not be 0 (root). Sandboxes must not '
        'run as root; use a non-zero UID (e.g. 1000).")\n',
        """            raise ValueError(
                "run_as_user must not be 0 (root). Sandboxes must not run as root; "
                "use a non-zero UID (e.g. 1000)."
            )
""",
    )
    _write_upstream_package(package_dir, validator=wrapped_validator)

    _load_patch().patch(package_dir)

    assert (package_dir / "types.py").read_text().count("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED") == 2


def test_root_patch_breaks_uv_style_hardlinks_between_runners(tmp_path: Path) -> None:
    first = tmp_path / "first" / "sandbox_k8s"
    second = tmp_path / "second" / "sandbox_k8s"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _write_upstream_package(first)
    os.link(first / "types.py", second / "types.py")
    os.link(first / "client.py", second / "client.py")
    os.link(first / "sandbox.py", second / "sandbox.py")

    module = _load_patch()
    module.patch(first)

    assert os.stat(first / "types.py").st_ino != os.stat(second / "types.py").st_ino
    assert "SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED" not in (second / "types.py").read_text()

    module.patch(second)

    assert (first / "types.py").read_text().count("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED") == 2
    assert (second / "types.py").read_text().count("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED") == 2
