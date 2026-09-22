# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import asyncio
import io
import re
import runpy
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

try:
    from scaled_evals.dispatch import sandbox_k8s
    from scaled_evals.models.runtime import LaunchHandle
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

PLUGIN_ROOT = Path(__file__).parents[1]
HARBOR_PATCH = PLUGIN_ROOT / "harbor-patches/sandbox_k8s_harbor.py"
ROOT_PATCH = PLUGIN_ROOT / "harbor-patches/patch_sandbox_k8s_root.py"


def _harbor_environment_methods(names: set[str]) -> type:
    tree = ast.parse(HARBOR_PATCH.read_text())
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment"
    )
    unsupported = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_service_operations_unsupported"
    )
    methods = [
        node
        for node in environment.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    # `list` is invariant, so widen before handing the methods to ClassDef.
    body: list[ast.stmt] = [*methods]
    isolated = ast.ClassDef(
        name="Environment",
        bases=[],
        keywords=[],
        body=body,
        decorator_list=[],
        type_params=[],
    )
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "_ensure_harbor": lambda: None,
        "_shell_quote": lambda value: "'" + value.replace("'", "'\\''") + "'",
        "_to_exec_result": lambda result: result,
        "sanitize_k8s_name": lambda value: re.sub(
            r"-{2,}",
            "-",
            re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-"),
        ),
    }
    exec(
        compile(ast.Module(body=[unsupported, isolated], type_ignores=[]), str(HARBOR_PATCH), "exec"),
        namespace,
    )
    return namespace["Environment"]


def test_separate_verifier_and_artifact_paths_are_portable() -> None:
    source = HARBOR_PATCH.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_is_verifier_sandbox", "_effective_sandbox_image"}
    }
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": dict,
        "_task_config_docker_image": lambda config: config.get("docker_image"),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=list(functions.values()), type_ignores=[])),
            str(HARBOR_PATCH),
            "exec",
        ),
        namespace,
    )
    is_verifier = namespace["_is_verifier_sandbox"]
    effective_image = namespace["_effective_sandbox_image"]
    assert is_verifier("task__abc__verifier__trial")
    assert not is_verifier("task-environment")
    assert (
        effective_image(
            session_id="task__abc__verifier__trial",
            task_env_config={"docker_image": "registry.invalid/verifier:v1"},
            profile_image="registry.invalid/task:v1",
        )
        == "registry.invalid/verifier:v1"
    )

    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"_HARBOR_PATH_ALLOW_PATTERNS", "_HARBOR_PATH_DENY_REPLACEMENTS"}
    }
    allow = assignments["_HARBOR_PATH_ALLOW_PATTERNS"]
    assert all(
        any(re.search(pattern, path) for pattern in allow)
        for path in ("/app/result.json", "/root/work/output", "/audit", "/shared")
    )
    assert assignments["_HARBOR_PATH_DENY_REPLACEMENTS"][r"^/root"] == (
        r"^/root/?$",
        r"^/root/\.",
    )
    assert 'getattr(tarfile, "data_filter", None)' in source


def test_multi_service_exec_and_artifact_transfer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class ServiceOperationsUnsupportedError(RuntimeError):
        pass

    base: Any = ModuleType("harbor.environments.base")
    base.ServiceOperationsUnsupportedError = ServiceOperationsUnsupportedError
    monkeypatch.setitem(sys.modules, "harbor", ModuleType("harbor"))
    monkeypatch.setitem(sys.modules, "harbor.environments", ModuleType("harbor.environments"))
    monkeypatch.setitem(sys.modules, "harbor.environments.base", base)

    class Sandbox:
        commands: list[tuple[list[str], dict[str, Any]]] = []
        downloads: list[tuple[str, dict[str, Any]]] = []

        async def run_command(self, command: list[str], **kwargs: Any) -> Any:
            self.commands.append((command, kwargs))
            return SimpleNamespace(stdout="ok", stderr="", return_code=0)

        async def download_file(self, source: str, **kwargs: Any) -> bytes:
            self.downloads.append((source, kwargs))
            return b"artifact"

    environment = _harbor_environment_methods(
        {
            "is_main_service",
            "_sidecar_container",
            "service_exec",
            "service_download_file",
            "service_download_dir",
            "service_download_dir_with_exclusions",
        }
    )()
    environment._sandbox = Sandbox()
    environment._sidecar_containers = {"postgres-db"}
    environment._command_timeout = 30.0
    environment.logger = SimpleNamespace(warning=lambda *_args: None)
    environment.is_main_service = lambda service: service is None or service == "main"
    environment.exec = lambda *_args, **_kwargs: None
    target = tmp_path / "database.dump"

    asyncio.run(
        environment.service_exec(
            "pg_dump",
            service="postgres_db",
            env={"PGPASSWORD": "secret"},
        )
    )
    command, kwargs = environment._sandbox.commands[0]
    assert command[:2] == ["sh", "-c"]
    assert "export PGPASSWORD='secret'" in command[2]
    assert kwargs == {"timeout": 30.0, "container": "postgres-db", "workdir": "/"}

    asyncio.run(environment.service_download_file("/tmp/db", target, service="postgres_db"))
    assert target.read_bytes() == b"artifact"
    assert environment._sandbox.downloads == [("/tmp/db", {"container": "postgres-db"})]

    calls: list[dict[str, Any]] = []

    async def with_exclusions(**call: Any) -> None:
        calls.append(call)

    environment.service_download_dir_with_exclusions = with_exclusions
    asyncio.run(
        environment.service_download_dir(
            source_dir="/shared",
            target_dir=tmp_path / "shared",
            service="postgres_db",
        )
    )
    assert calls[0]["exclude"] == []
    with pytest.raises(ServiceOperationsUnsupportedError):
        asyncio.run(environment.service_exec("true", service="missing"))
    environment._withheld_sidecar_containers = {"postgres-db"}
    with pytest.raises(ServiceOperationsUnsupportedError, match="withheld"):
        asyncio.run(environment.service_exec("true", service="postgres_db"))
    del base.ServiceOperationsUnsupportedError
    with pytest.raises(RuntimeError, match="withheld"):
        asyncio.run(environment.service_exec("true", service="postgres_db"))


def test_compose_restore_and_dispatch_extraction_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "pack"
    task_tree = pack_root / "tasks/task"
    task_tree.mkdir(parents=True)
    (pack_root / "docker-compose.yaml").write_text("services:\n  api:\n    image: api\n")
    sandbox_k8s._restore_compose_definition(pack_root, task_tree)
    assert (task_tree / "environment/docker-compose.yaml").is_file()

    archive = tmp_path / "pack.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("one", "two"):
            content = b"ab"
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))

    monkeypatch.setattr(sandbox_k8s.settings, "task_pack_max_members", 1)
    monkeypatch.setattr(sandbox_k8s.settings, "task_pack_max_extracted_size_bytes", 100)
    with pytest.raises(ValueError, match="member limit"):
        sandbox_k8s._extract_pack(archive, tmp_path / "members")

    monkeypatch.setattr(sandbox_k8s.settings, "task_pack_max_members", 10)
    monkeypatch.setattr(sandbox_k8s.settings, "task_pack_max_extracted_size_bytes", 3)
    with pytest.raises(ValueError, match="extracted-size limit"):
        sandbox_k8s._extract_pack(archive, tmp_path / "size")


def test_root_patch_constrains_sidecars_and_ipv6(tmp_path: Path) -> None:
    patch = runpy.run_path(str(ROOT_PATCH))
    client = tmp_path / "client.py"
    client.write_text(
        """                            container.name == "sandbox"
                            and not container.read_only_root_filesystem
"""
        + patch["_SYSCTLS_LOW_PORTS_ONLY"]
    )

    patch["patch_sidecar_capabilities"](tmp_path)
    patch["patch_sysctls"](tmp_path)

    source = client.read_text()
    assert 'container.name == "sandbox"' not in source
    assert "not container.read_only_root_filesystem" in source
    assert all(
        sysctl in source
        for sysctl in (
            "net.ipv6.conf.all.disable_ipv6",
            "net.ipv6.conf.default.disable_ipv6",
            "net.ipv6.conf.lo.disable_ipv6",
        )
    )


def test_deployment_owns_baseline_rbac_and_documents_scoped_overrides() -> None:
    documents = list(yaml.safe_load_all((PLUGIN_ROOT / "deploy/k8s/sandbox-rbac.yaml").read_text()))
    role = next(document for document in documents if document["kind"] == "Role")
    sandbox_rule = next(
        rule
        for rule in role["rules"]
        if rule.get("apiGroups") == ["agents.x-k8s.io"] and rule.get("resources") == ["sandboxes"]
    )
    assert sandbox_rule["verbs"] == ["create", "delete", "get", "list", "watch"]
    claim_rule = next(
        rule
        for rule in role["rules"]
        if rule.get("apiGroups") == ["extensions.agents.x-k8s.io"] and rule.get("resources") == ["sandboxclaims"]
    )
    assert claim_rule["verbs"] == ["create", "delete", "get", "list", "watch"]

    docs = (PLUGIN_ROOT / "deploy/k8s/README.md").read_text()
    assert "enable_ipv6_loopback: true" in docs
    assert "environment.kwargs.persistent_env.TMPDIR" in docs
    assert "task-specific `setup_command`" in docs
    assert "ico-path-patch" not in docs


def test_cleanup_kubeconfig_flag_falls_back_when_recorded_path_is_absent(tmp_path: Path) -> None:
    """A cleanup must not replay a kubeconfig path from another pod's filesystem.

    Under Platform Jobs the path is recorded inside the Job pod (HOME=/tmp) but
    the cleanup runs in the dispatch worker (HOME=/sa-kube). Passing the stale
    path made every kubectl call fail and wedged the evaluation in `running`.
    """
    present = tmp_path / "config"
    present.write_text("apiVersion: v1\n")
    assert sandbox_k8s._kubeconfig_flag(present) == ["--kubeconfig", str(present)]

    # The Job pod's path, as seen from the worker that drains the cleanup.
    assert sandbox_k8s._kubeconfig_flag("/tmp/.kube/config") == []
    assert sandbox_k8s._kubeconfig_flag(None) == []
    assert sandbox_k8s._kubeconfig_flag("") == []


def test_sandbox_kubectl_base_drops_stale_kubeconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selector-based teardown path uses the same fallback."""
    monkeypatch.setattr(sandbox_k8s.shutil, "which", lambda name: "/usr/bin/kubectl" if name == "kubectl" else None)
    handle = LaunchHandle(
        backend="sandbox_k8s",
        external_id="eval_1",
        raw={
            "cleanup": {
                "selector": "scaled-evals.nvidia.com/evaluation=eval_1",
                # The Job pod's path, as seen from the worker draining the cleanup.
                "kubeconfig_path": "/tmp/.kube/config",
                "namespace": "evals",
            }
        },
    )
    result = sandbox_k8s._sandbox_kubectl_base(handle)
    assert result is not None
    base, selector = result
    assert "--kubeconfig" not in base
    assert base == ["/usr/bin/kubectl", "-n", "evals"]
    assert selector == "scaled-evals.nvidia.com/evaluation=eval_1"
