# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the narrow scaled-evals root-execution compatibility patch.

This is intentionally pinned to sandbox-k8s 0.1.13 behavior. Source structure
is checked before replacement so formatting-only wheel differences are safe,
while a future SDK behavior change fails the image build for review.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _write_text_breaking_links(path: Path, text: str) -> None:
    """Atomically replace a uv-hardlinked package file before modifying it."""
    temporary = path.with_name(f".{path.name}.scaled-evals.tmp")
    if temporary.exists():
        raise RuntimeError(f"temporary patch path already exists: {temporary}")
    try:
        temporary.write_text(text)
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one sandbox-k8s 0.1.13 patch target in {path}: {old[:80]!r}")
    _write_text_breaking_links(path, text.replace(old, new, 1))


def _insert_after_once(path: Path, needle: str, addition: str) -> None:
    text = path.read_text()
    if text.count(needle) != 1:
        raise RuntimeError(f"expected exactly one sandbox-k8s 0.1.13 insertion target in {path}: {needle[:80]!r}")
    _write_text_breaking_links(path, text.replace(needle, f"{needle}{addition}", 1))


def _expected_validator_dump() -> str:
    source = (
        """class Expected:
    @field_validator("run_as_user")
    @classmethod
    def _reject_root_user(cls, v: int | None) -> int | None:
        if v is not None and v == 0:
            raise ValueError("run_as_user must not be 0 (root). Sandboxes must not """
        'run as root; use a non-zero UID (e.g. 1000).")\n'
        """        return v
"""
    )
    tree = ast.parse(source)
    function = tree.body[0].body[0]
    return ast.dump(function, include_attributes=False)


def _patch_root_validators(path: Path, replacement: str) -> None:
    text = path.read_text()
    tree = ast.parse(text)
    expected = _expected_validator_dump()
    targets: list[ast.FunctionDef] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in {
            "SandboxSpec",
            "SandboxTemplateSpec",
        }:
            continue
        matches = [
            child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "_reject_root_user"
        ]
        if len(matches) != 1 or ast.dump(matches[0], include_attributes=False) != expected:
            raise RuntimeError(f"unexpected sandbox-k8s 0.1.13 run_as_user validator in {node.name}")
        targets.append(matches[0])
    if len(targets) != 2:
        raise RuntimeError("expected sandbox-k8s 0.1.13 validators in both spec classes")

    lines = text.splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    for target in sorted(targets, key=lambda item: item.lineno, reverse=True):
        start = min(decorator.lineno for decorator in target.decorator_list) - 1
        assert target.end_lineno is not None
        lines[start : target.end_lineno] = replacement_lines
    _write_text_breaking_links(path, "".join(lines))


_SYSCTLS_LOW_PORTS_ONLY = """        if os.environ.get(
            "SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS", ""
        ).lower() == "true":
            ctx["sysctls"] = [
                {
                    "name": "net.ipv4.ip_unprivileged_port_start",
                    "value": "0",
                }
            ]
"""

_SYSCTLS_LO_ONLY = """        sysctls = []
        if os.environ.get(
            "SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS", ""
        ).lower() == "true":
            sysctls.append(
                {
                    "name": "net.ipv4.ip_unprivileged_port_start",
                    "value": "0",
                }
            )
        if os.environ.get(
            "SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK", ""
        ).lower() == "true":
            sysctls.append(
                {
                    "name": "net.ipv6.conf.lo.disable_ipv6",
                    "value": "0",
                }
            )
        if sysctls:
            ctx["sysctls"] = sysctls
"""

_SYSCTLS_GRANT = """        sysctls = []
        if os.environ.get(
            "SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS", ""
        ).lower() == "true":
            sysctls.append(
                {
                    "name": "net.ipv4.ip_unprivileged_port_start",
                    "value": "0",
                }
            )
        if os.environ.get(
            "SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK", ""
        ).lower() == "true":
            sysctls.extend(
                [
                    {"name": "net.ipv6.conf.all.disable_ipv6", "value": "0"},
                    {"name": "net.ipv6.conf.default.disable_ipv6", "value": "0"},
                    {"name": "net.ipv6.conf.lo.disable_ipv6", "value": "0"},
                ]
            )
        if sysctls:
            ctx["sysctls"] = sysctls
"""


def patch_capabilities(package_dir: Path) -> None:
    """Upgrade an inherited root-enabled runner with the narrow capability grant."""
    client_path = package_dir / "client.py"
    old = '"capabilities": {"drop": ["ALL"]},\n'
    # Keyed on KILL so a runner carrying an older grant fails loudly here rather
    # than inheriting a list that is missing a capability.
    marker = '                                "KILL",\n'
    # A base image built before KILL joined the grant already carries the rest of
    # the list, so splice KILL into it instead of rewriting the whole manifest.
    preceding = '                                "FOWNER",\n'
    text = client_path.read_text()
    old_count = text.count(old)
    marker_count = text.count(marker)
    if old_count == 0 and marker_count == 1:
        return
    if old_count == 0 and marker_count == 0 and text.count(preceding) == 1:
        _write_text_breaking_links(client_path, text.replace(preceding, preceding + marker, 1))
        return
    if old_count != 1 or marker_count != 0:
        raise RuntimeError("unexpected sandbox-k8s capability manifest while upgrading inherited runner")
    new = """"capabilities": {
                    "drop": ["ALL"],
                    **(
                        {
                            "add": [
                                "CHOWN",
                                "DAC_OVERRIDE",
                                "FOWNER",
                                "KILL",
                                "SETGID",
                                "SETUID",
                            ]
                        }
                        if (
                            container.name == "sandbox"
                            and not container.read_only_root_filesystem
                            and os.environ.get(
                                "SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED", ""
                            ).lower()
                            == "true"
                        )
                        else {}
                    ),
                },
"""
    _write_text_breaking_links(client_path, text.replace(old, new, 1))


def patch_sysctls(package_dir: Path) -> None:
    """Upgrade an inherited runner to the deterministic IPv6 loopback sysctls.

    ``net.ipv6.conf.all.disable_ipv6`` is a broadcast write: the kernel stamps it
    onto ``default`` and onto every interface that already exists, ``lo``
    included. GKE injects ``all=1`` into every pod sandbox on an IPv4-only
    cluster, and runc applies the sysctl map with ``for k, v := range``, whose
    order Go randomises per execution. Requesting ``lo=0`` alone therefore
    restores ``::1`` only sometimes. Overriding ``all`` and ``default`` to 0
    leaves no key in the map that writes a 1 to ``lo``, which removes the race
    by construction rather than by luck.

    Two inherited shapes are handled: a base image predating the IPv6 gate
    entirely, and one carrying an earlier ``lo``-only grant. Missing the second
    would silently leave a runner emitting the racy single sysctl.
    """
    client_path = package_dir / "client.py"
    text = client_path.read_text()
    if "net.ipv6.conf.all.disable_ipv6" in text:
        return
    for old in (_SYSCTLS_LOW_PORTS_ONLY, _SYSCTLS_LO_ONLY):
        if text.count(old) == 1:
            _write_text_breaking_links(client_path, text.replace(old, _SYSCTLS_GRANT, 1))
            return
    raise RuntimeError("unexpected inherited sandbox-k8s sysctl manifest")


def patch_sidecar_capabilities(package_dir: Path) -> None:
    """Let a writable-root sidecar carry the same narrow grant the task container has.

    ``patch_capabilities`` keys the grant on the container being named
    ``sandbox``, which withholds it from every sidecar. That breaks the ordinary
    database images these multi-service tasks depend on: ``postgres`` and
    ``mysql`` start as root and hand off to their service account with ``gosu``
    or ``su-exec``, which needs ``SETUID``/``SETGID``. With ``drop: ["ALL"]`` the
    entrypoint cannot drop privilege and the container dies before it is ready.

    The name was never the security boundary -- the other two conditions are, and
    they still hold: the container must be explicitly configured with a writable
    root, and the deployment must be operator-authorized. This also matches
    upstream's own intent, since ``build_sidecar_pod_spec`` gives sidecars the
    main container's security context.
    """
    client_path = package_dir / "client.py"
    old = """                            container.name == "sandbox"
                            and not container.read_only_root_filesystem
"""
    new = """                            not container.read_only_root_filesystem
"""
    text = client_path.read_text()
    # A runner that already carries the widened grant is left alone, so this
    # stays safe to re-run against an inherited image.
    if old not in text:
        return
    if text.count(old) != 1:
        raise RuntimeError("unexpected sandbox-k8s capability gate while widening to sidecars")
    _write_text_breaking_links(client_path, text.replace(old, new, 1))


def patch_sidecar_host_aliases(package_dir: Path) -> None:
    """Resolve each sidecar's own name to loopback inside the pod.

    A multi-service task addresses its services by the names its compose file
    gives them -- ``http://legacy-app:8000``, ``postgres``, ``customer`` -- and
    every container in a pod shares one network namespace, so loopback is where
    those names have to point. ``sandbox_k8s`` already does this in
    ``build_sidecar_pod_spec``, the Compose-shaped helper, but nothing calls it:
    the ``sidecars`` list this adapter uses goes through the high-level pod
    builder, which emits no ``hostAliases`` at all. Without them a service that
    waits on a sibling by name never resolves it and simply never becomes ready.
    """
    client_path = package_dir / "client.py"
    anchor = """            "securityContext": self._build_security_context(spec),
        }
"""
    addition = """        if spec.sidecars:
            pod_spec["hostAliases"] = [
                {
                    "ip": "127.0.0.1",
                    "hostnames": [sidecar.name for sidecar in spec.sidecars],
                }
            ]
"""
    text = client_path.read_text()
    # A base image that already carries the grant is left alone, so this stays
    # safe to re-run against an inherited runner.
    if "hostAliases" in text:
        return
    if text.count(anchor) != 1:
        raise RuntimeError("unexpected sandbox-k8s pod builder while adding sidecar host aliases")
    _write_text_breaking_links(client_path, text.replace(anchor, anchor + addition, 1))


def patch(package_dir: Path) -> None:
    types_path = package_dir / "types.py"
    client_path = package_dir / "client.py"
    sandbox_path = package_dir / "sandbox.py"

    _replace_once(
        types_path,
        "from enum import Enum\n",
        "from enum import Enum\nimport os\n",
    )
    new_validator = """    @field_validator("run_as_user")
    @classmethod
    def _reject_root_user(cls, v: int | None) -> int | None:
        if v is not None and v == 0 and os.environ.get(
            "SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED", ""
        ).lower() != "true":
            raise ValueError(
                "run_as_user must not be 0 (root) without internal operator authorization"
            )
        return v
"""
    _patch_root_validators(types_path, new_validator)
    _insert_after_once(
        types_path,
        """    volume_mounts: list[VolumeMount] = Field(
        default_factory=list,
        description="Volume mounts for this container",
    )
""",
        """    read_only_root_filesystem: bool = Field(
        default=True,
        description="Whether the container image root filesystem is read-only.",
    )
""",
    )

    _replace_once(
        client_path,
        "import shlex\nimport time\n",
        "import shlex\nimport shutil\nimport subprocess\nimport time\n"
        "\nfrom websocket import WebSocketConnectionClosedException\n",
    )
    _replace_once(
        client_path,
        '"readOnlyRootFilesystem": True,\n',
        '"readOnlyRootFilesystem": container.read_only_root_filesystem,\n',
    )
    patch_capabilities(package_dir)
    patch_sidecar_capabilities(package_dir)
    patch_sidecar_host_aliases(package_dir)

    _replace_once(
        client_path,
        """        poll_sec = 1.0
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        timed_out = False

        while resp.is_open():
            if deadline is not None and time.monotonic() >= deadline:
""",
        """        poll_sec = 1.0
        keepalive_sec = 30.0
        next_keepalive = time.monotonic() + keepalive_sec
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        timed_out = False

        while resp.is_open():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
""",
    )
    _insert_after_once(
        sandbox_path,
        """        network_policy: NetworkPolicy | dict[str, Any] | None = None,
        image_pull_policy: ImagePullPolicy = "IfNotPresent",
""",
        """        read_only_root_filesystem: bool = True,
        _scaled_evals_writable_root_authorized: bool = False,
""",
    )
    _insert_after_once(
        sandbox_path,
        """        self._image_pull_policy = image_pull_policy
""",
        """        if not read_only_root_filesystem and (
            run_as_user != 0
            or _scaled_evals_writable_root_authorized is not True
            or os.environ.get("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED", "").lower()
            != "true"
        ):
            raise SandboxConfigError(
                "writable root requires operator-authorized run_as_user=0"
            )
        self._read_only_root_filesystem = read_only_root_filesystem
""",
    )
    _insert_after_once(
        sandbox_path,
        """        # Common settings
        image: str = DEFAULT_IMAGE,
        image_pull_policy: ImagePullPolicy = "IfNotPresent",
""",
        """        read_only_root_filesystem: bool = True,
        _scaled_evals_writable_root_authorized: bool = False,
""",
    )
    _insert_after_once(
        sandbox_path,
        """                image_pull_policy=image_pull_policy,
""",
        """                read_only_root_filesystem=read_only_root_filesystem,
                _scaled_evals_writable_root_authorized=(
                    _scaled_evals_writable_root_authorized
                ),
""",
    )
    _insert_after_once(
        sandbox_path,
        """                resources=self._resources,
""",
        """                read_only_root_filesystem=self._read_only_root_filesystem,
""",
    )
    _insert_after_once(
        client_path,
        """                timed_out = True
                break
""",
        """            if now >= next_keepalive:
                resp.sock.ping("scaled-evals-keepalive")
                next_keepalive = now + keepalive_sec
""",
    )

    _insert_after_once(
        client_path,
        """        self._api_client = self._load_config(kubeconfig_path, context, in_cluster, verify_ssl)
""",
        """        self._kubeconfig_path = kubeconfig_path
        self._context = context
""",
    )

    _insert_after_once(
        client_path,
        """        ctx: dict[str, Any] = {
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        }
""",
        _SYSCTLS_GRANT,
    )

    _insert_after_once(
        client_path,
        """        return stdout_data, stderr_data, timed_out

""",
        """    @staticmethod
    def _exec_in_pod_with_kubectl(
        pod_name: str,
        command: list[str],
        namespace: str = "default",
        container: str | None = None,
        timeout: float | None = None,
        stdin_data: bytes | None = None,
        kubeconfig_path: str | None = None,
        context: str | None = None,
    ) -> CommandResult | None:
        \"\"\"Fallback exec path for OpenShift clusters whose Python websocket exec breaks.

        Some clusters accept `kubectl exec` from the same in-cluster
        service-account kubeconfig while Python Kubernetes `stream()` can fail
        or hang behind the API-server exec backend. Keep this isolated to the
        explicit/native path so clusters where Python streaming works keep the
        lower-overhead path.
        \"\"\"
        kubectl = shutil.which("kubectl") or shutil.which("oc")
        if kubectl is None:
            return None

        argv = [kubectl]
        kubeconfig = (kubeconfig_path or os.environ.get("KUBECONFIG", "")).strip()
        if kubeconfig:
            argv.extend(["--kubeconfig", kubeconfig])
        exec_context = (context or os.environ.get("SANDBOX_CONTEXT", "")).strip()
        if exec_context and exec_context != "incluster":
            argv.extend(["--context", exec_context])
        argv.extend(["-n", namespace, "exec"])
        if stdin_data is not None:
            argv.append("-i")
        if container:
            argv.extend(["-c", container])
        argv.extend([pod_name, "--", *command])

        attempts = 10
        for attempt in range(1, attempts + 1):
            try:
                result = subprocess.run(
                    argv,
                    input=stdin_data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxTimeoutError(
                    f"Command in pod {namespace}/{pod_name} timed out after {timeout}s"
                ) from exc
            except OSError:
                logger.exception("kubectl exec fallback failed to start")
                return None

            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            backend_tls_error = (
                "error dialing backend" in stderr.lower()
                and "first record does not look like a tls handshake" in stderr.lower()
            )
            if result.returncode == 0 or not backend_tls_error or attempt == attempts:
                return CommandResult(
                    exit_code=result.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            logger.warning(
                "kubectl exec fallback hit a backend TLS handshake error for "
                "%s/%s; retrying attempt %d/%d",
                namespace,
                pod_name,
                attempt + 1,
                attempts,
            )
            time.sleep(min(2 ** (attempt - 1), 5))

        return None

    @staticmethod
    def _prefer_kubectl_exec(context: str | None = None) -> bool:
        value = os.environ.get("SANDBOX_K8S_PREFER_NATIVE_EXEC", "").strip().lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        # Opt-in auto-enable for clusters whose kubeconfig context name is known to
        # need the fallback. Configured rather than hardcoded: keying behavior off a
        # particular cluster's name is not something a general deployment can reason
        # about. Unset means the explicit flag above is the only way in.
        match = os.environ.get("SANDBOX_K8S_NATIVE_EXEC_CONTEXT_MATCH", "").strip().lower()
        if not match:
            return False
        selected_context = context or os.environ.get("SANDBOX_CONTEXT", "")
        return match in selected_context.lower()

""",
    )

    old_exec_error_block = (
        """            if e.status == 404:
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
"""
    )
    new_exec_error_block = """        except WebSocketConnectionClosedException as e:
            logger.warning(
                "Kubernetes exec stream disconnected for %s/%s: %s",
                namespace,
                pod_name,
                e,
            )
            raise SandboxExecutionError(
                f"Kubernetes exec stream disconnected for {namespace}/{pod_name}"
            ) from e
        except ApiException as e:
            if e.status == 404:
                raise SandboxNotFoundError(
                    f"Pod {namespace}/{pod_name} not found "
                    "(sandbox may have been deleted or timed out)",
                    **_api_exc_fields(e),
                ) from e
            fallback = self._exec_in_pod_with_kubectl(
                pod_name,
                exec_command,
                namespace,
                container,
                timeout,
                stdin_data,
                self._kubeconfig_path,
                self._context,
            )
            if fallback is not None:
                logger.warning(
                    "Python Kubernetes exec stream failed for %s/%s; used kubectl fallback",
                    namespace,
                    pod_name,
                )
                return fallback
            logger.error("Failed to exec command in %s/%s: %s", namespace, pod_name, e)
            raise SandboxExecutionError(
                f"Failed to exec command in {namespace}/{pod_name}: {e.reason}",
                **_api_exc_fields(e),
            ) from e
"""
    _replace_once(
        client_path,
        "        except ApiException as e:\n" + old_exec_error_block,
        new_exec_error_block,
    )

    _replace_once(
        client_path,
        """                    if event_type in ("ADDED", "MODIFIED"):
                        info = self._parse_sandbox_response(obj)
                        result = self._check_sandbox_status(info, namespace, name)
                        if result is not None:
                            w.stop()
                            return result

            except ApiException as e:
""",
        """                    if event_type in ("ADDED", "MODIFIED"):
                        info = self._parse_sandbox_response(obj)
                        result = self._check_sandbox_status(info, namespace, name)
                        if result is not None:
                            w.stop()
                            return result

                info = self.get_sandbox(name, namespace)
                result = self._check_sandbox_status(info, namespace, name)
                if result is not None:
                    w.stop()
                    return result

            except ApiException as e:
""",
    )

    _insert_after_once(
        client_path,
        """        exec_command = self._prepare_exec_command(command, workdir)
""",
        """        if self._prefer_kubectl_exec(self._context):
            native = self._exec_in_pod_with_kubectl(
                pod_name,
                exec_command,
                namespace,
                container,
                timeout,
                stdin_data,
                self._kubeconfig_path,
                self._context,
            )
            if native is not None:
                logger.info(
                    "Using native kubectl exec for %s/%s because SANDBOX_CONTEXT "
                    "requires it",
                    namespace,
                    pod_name,
                )
                return native

""",
    )

    _replace_once(
        client_path,
        """        ctx: dict[str, Any] = {
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        }
""",
        """        root_authorized = (
            spec.run_as_user == 0
            and os.environ.get("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED", "").lower()
            == "true"
        )
        ctx: dict[str, Any] = {
            "runAsNonRoot": not root_authorized,
            "seccompProfile": {"type": "RuntimeDefault"},
        }
""",
    )


def verify() -> None:
    """Prove default denial and explicitly authorized manifest rendering."""
    import inspect
    import os
    from types import SimpleNamespace

    from pydantic import ValidationError
    from sandbox_k8s.client import K8sSandboxClient
    from sandbox_k8s.sandbox import K8sSandbox
    from sandbox_k8s.types import ContainerSpec, SandboxSpec

    constructor_parameters = inspect.signature(K8sSandbox).parameters
    assert "read_only_root_filesystem" in constructor_parameters
    assert "_scaled_evals_writable_root_authorized" in constructor_parameters

    try:
        SandboxSpec(name="denied", namespace="default", run_as_user=0)
    except ValidationError:
        pass
    else:
        raise AssertionError("root unexpectedly allowed without authorization")

    os.environ["SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED"] = "true"
    try:
        spec = SandboxSpec(name="allowed", namespace="default", run_as_user=0)
        assert K8sSandboxClient._build_security_context(spec)["runAsNonRoot"] is False
        os.environ["SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS"] = "true"
        non_root_spec = SandboxSpec(name="rootless", namespace="default", run_as_user=1000)
        security_context = K8sSandboxClient._build_security_context(non_root_spec)
        assert security_context["runAsNonRoot"] is True
        assert security_context["sysctls"] == [
            {
                "name": "net.ipv4.ip_unprivileged_port_start",
                "value": "0",
            }
        ]
        os.environ["SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK"] = "true"
        security_context = K8sSandboxClient._build_security_context(non_root_spec)
        assert security_context["sysctls"] == [
            {"name": "net.ipv4.ip_unprivileged_port_start", "value": "0"},
            {"name": "net.ipv6.conf.all.disable_ipv6", "value": "0"},
            {"name": "net.ipv6.conf.default.disable_ipv6", "value": "0"},
            {"name": "net.ipv6.conf.lo.disable_ipv6", "value": "0"},
        ]
        os.environ.pop("SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS", None)
        security_context = K8sSandboxClient._build_security_context(non_root_spec)
        assert security_context["sysctls"] == [
            {"name": "net.ipv6.conf.all.disable_ipv6", "value": "0"},
            {"name": "net.ipv6.conf.default.disable_ipv6", "value": "0"},
            {"name": "net.ipv6.conf.lo.disable_ipv6", "value": "0"},
        ]
        os.environ["SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS"] = "true"
        assert K8sSandboxClient._prefer_kubectl_exec() is False
        # Unset match env var means a context name alone never enables the fallback.
        assert K8sSandboxClient._prefer_kubectl_exec("ns/api-openshift-example-invalid") is False
        os.environ["SANDBOX_K8S_NATIVE_EXEC_CONTEXT_MATCH"] = "openshift"
        try:
            assert K8sSandboxClient._prefer_kubectl_exec("ns/api-openshift-example-invalid") is True
            assert K8sSandboxClient._prefer_kubectl_exec("ns/api-other-example-invalid") is False
        finally:
            del os.environ["SANDBOX_K8S_NATIVE_EXEC_CONTEXT_MATCH"]
        container = SimpleNamespace(
            name="sandbox",
            image="example.invalid/task@sha256:" + "a" * 64,
            working_dir="/workspace",
            image_pull_policy="IfNotPresent",
            read_only_root_filesystem=False,
            command=[],
            args=[],
            env=[],
            resources=None,
            volume_mounts=[],
        )
        manifest = K8sSandboxClient._build_container_manifest(container)
        assert manifest["securityContext"]["capabilities"] == {
            "drop": ["ALL"],
            "add": ["CHOWN", "DAC_OVERRIDE", "FOWNER", "KILL", "SETGID", "SETUID"],
        }
        # A sidecar carries the same grant, because the database images these
        # multi-service tasks run start as root and hand off with gosu. Writable
        # root is what gates the grant, not the container's name.
        container.name = "postgres"
        assert K8sSandboxClient._build_container_manifest(container)["securityContext"]["capabilities"] == {
            "drop": ["ALL"],
            "add": ["CHOWN", "DAC_OVERRIDE", "FOWNER", "KILL", "SETGID", "SETUID"],
        }
        container.read_only_root_filesystem = True
        assert K8sSandboxClient._build_container_manifest(container)["securityContext"]["capabilities"] == {
            "drop": ["ALL"]
        }
        container.read_only_root_filesystem = False

        # A sidecar's own compose service name has to resolve, or a service that
        # waits on a sibling by name hangs until the readiness gate times out.
        aliased = SandboxSpec(
            name="aliased",
            namespace="default",
            run_as_user=0,
            sidecars=[ContainerSpec(name="legacy-app", image="example.invalid/app:v1")],
        )
        pod_spec = K8sSandboxClient._build_pod_spec_from_high_level(K8sSandboxClient, aliased)
        assert pod_spec["hostAliases"] == [{"ip": "127.0.0.1", "hostnames": ["legacy-app"]}]
        # A single-container sandbox must not grow the field at all.
        plain = SandboxSpec(name="plain", namespace="default", run_as_user=0)
        assert "hostAliases" not in K8sSandboxClient._build_pod_spec_from_high_level(K8sSandboxClient, plain)
    finally:
        os.environ.pop("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED", None)
        os.environ.pop("SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS", None)
        os.environ.pop("SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK", None)
        os.environ.pop("SANDBOX_CONTEXT", None)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        patch(Path(sys.argv[1]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--capabilities-only":
        # This mode upgrades a runner that inherited an already-patched
        # sandbox_k8s from the base image, so every step that a base image may
        # predate has to run here too -- each one no-ops when already present.
        patch_capabilities(Path(sys.argv[2]))
        patch_sidecar_capabilities(Path(sys.argv[2]))
        patch_sidecar_host_aliases(Path(sys.argv[2]))
        patch_sysctls(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: patch_sandbox_k8s_root.py [--capabilities-only] PACKAGE_DIR")
    verify()
