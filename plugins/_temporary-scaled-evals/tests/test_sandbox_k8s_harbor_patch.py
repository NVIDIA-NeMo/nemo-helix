# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
import base64
import binascii
import copy
import io
import json
import math
import re
import secrets
import shlex
import subprocess
import tarfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

PATCH = Path(__file__).parents[1] / "harbor-patches/sandbox_k8s_harbor.py"


class _SandboxConfigError(ValueError):
    pass


class _Model:
    def __init__(self, **values: Any) -> None:
        for key, value in values.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls, value: dict[str, Any]) -> "_Model":
        return cls(**value)

    def model_copy(self, *, deep: bool = False) -> "_Model":
        return copy.deepcopy(self) if deep else copy.copy(self)


class _ResourceRequirements(_Model):
    def __init__(
        self,
        requests: dict[str, str] | None = None,
        limits: dict[str, str] | None = None,
    ) -> None:
        super().__init__(requests=requests, limits=limits)


class _Toleration(_Model):
    def __init__(
        self,
        key: str | None = None,
        operator: str = "Equal",
        effect: str | None = None,
        **values: Any,
    ) -> None:
        super().__init__(key=key, operator=operator, effect=effect, **values)


def _gpu_helpers() -> dict[str, Any]:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    helpers: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name
        in {
            "_normalize_gpu_count",
            "_resolve_command_timeout",
            "_gpu_resources",
            "_gpu_tolerations",
        }
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "math": math,
        "ResourceRequirements": _ResourceRequirements,
        "Toleration": _Toleration,
        "SandboxConfigError": _SandboxConfigError,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(PATCH), "exec"), namespace)
    return namespace


def _image_helpers() -> dict[str, Any]:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    helpers: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name
        in {
            "_effective_sandbox_image",
            "_is_verifier_sandbox",
            "_sandbox_sdk_working_dir",
            "_task_config_docker_image",
        }
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": dict,
        "SandboxConfigError": _SandboxConfigError,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(PATCH), "exec"), namespace)
    return namespace


def _annotation_helpers() -> dict[str, Any]:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_sandbox_annotations"
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "_AUTOSCALER_SAFE_TO_EVICT_ANNOTATION": ("cluster-autoscaler.kubernetes.io/safe-to-evict"),
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(PATCH), "exec"), namespace)
    return namespace


def test_harbor_patch_marks_sandbox_pods_unsafe_to_evict() -> None:
    annotations = _annotation_helpers()["_sandbox_annotations"](
        {
            "example.com/profile": "preserved",
            "cluster-autoscaler.kubernetes.io/safe-to-evict": "true",
        }
    )

    assert annotations == {
        "example.com/profile": "preserved",
        "cluster-autoscaler.kubernetes.io/safe-to-evict": "false",
    }


def _writable_root_validator() -> Any:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_writable_root_request"
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "SandboxConfigError": _SandboxConfigError,
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(PATCH), "exec"), namespace)
    return namespace["_validate_writable_root_request"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requested_uid": 1000}, "operator-authorized"),
        ({"root_authorized": False}, "operator-authorized"),
        ({"writable_root_authorized": False}, "operator-authorized"),
        ({"template_name": "warm"}, "direct CRD"),
        ({"pod_spec": {}}, "full pod_spec"),
    ],
)
def test_writable_root_request_fails_closed(overrides: dict[str, Any], message: str) -> None:
    values = {
        "read_only_root_filesystem": False,
        "requested_uid": 0,
        "root_authorized": True,
        "writable_root_authorized": True,
        "template_name": None,
        "pod_spec": None,
        **overrides,
    }
    with pytest.raises(_SandboxConfigError, match=message):
        _writable_root_validator()(**values)


def test_writable_root_request_allows_authorized_direct_root_only() -> None:
    validate = _writable_root_validator()
    validate(
        read_only_root_filesystem=False,
        requested_uid=0,
        root_authorized=True,
        writable_root_authorized=True,
        template_name=None,
        pod_spec=None,
    )
    validate(
        read_only_root_filesystem=True,
        requested_uid=1000,
        root_authorized=False,
        writable_root_authorized=False,
        template_name="warm",
        pod_spec={},
    )


def _environment_method(name: str, *, extra_namespace: dict[str, Any] | None = None) -> Any:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment"
    )
    method = next(
        node
        for node in environment.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    )
    isolated = ast.ClassDef(
        name="Environment",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
        type_params=[],
    )
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "Any": Any,
        "CommandResult": Any,
        "Path": Path,
        "io": io,
        "secrets": secrets,
        "tarfile": tarfile,
        "_shell_quote": shlex.quote,
    }
    if extra_namespace:
        namespace.update(extra_namespace)
    exec(compile(ast.Module(body=[isolated], type_ignores=[]), str(PATCH), "exec"), namespace)
    return getattr(namespace["Environment"], name)


def _live_log_environment() -> type:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment"
    )
    names = {
        "_poll_sandbox_logs_once",
        "_emit_sandbox_log_text",
        "_emit_sandbox_log_line",
    }
    methods: list[ast.stmt] = [
        node
        for node in environment.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    isolated = ast.ClassDef(
        name="Environment",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
        type_params=[],
    )
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "base64": base64,
        "binascii": binascii,
        "json": json,
        "_SANDBOX_LOG_POLL_SCRIPT": "poll-script",
        "_SANDBOX_LOG_MAX_BYTES_PER_POLL": 256 * 1024,
        "_SANDBOX_LOG_MAX_BYTES_PER_FILE": 64 * 1024,
        "_SANDBOX_LOG_POLL_TIMEOUT_SECONDS": 15.0,
        "_SANDBOX_LOG_MAX_LINE_CHARS": 16 * 1024,
    }
    exec(compile(ast.Module(body=[isolated], type_ignores=[]), str(PATCH), "exec"), namespace)
    return namespace["Environment"]


def test_harbor_patch_defers_cleanup_failure_to_evaluation_runtime() -> None:
    class Sandbox:
        name = "sandbox-test"

        async def stop(self) -> None:
            raise RuntimeError("webhook timeout")

    class Logger:
        def __init__(self) -> None:
            self.warnings: list[tuple[str, object]] = []

        def info(self, *_args: object) -> None:
            pass

        def warning(self, message: str, value: object) -> None:
            self.warnings.append((message, value))

    environment: Any = type("Environment", (), {})()
    environment._started = True
    environment._sandbox = Sandbox()
    environment.logger = Logger()

    async def collect() -> None:
        pass

    environment._collect_pod_artifacts = collect
    environment._collect_sidecar_logs = collect
    environment._stop_sandbox_log_stream = collect

    asyncio.run(_environment_method("stop")(environment))

    assert environment._started is False
    assert len(environment.logger.warnings) == 1
    message, error = environment.logger.warnings[0]
    assert message == "K8s sandbox cleanup deferred to evaluation runtime: %s"
    assert str(error) == "webhook timeout"


def test_harbor_patch_streams_incremental_in_pod_log_lines() -> None:
    def record(path: str, offset: int, text: str) -> str:
        data = text.encode()
        return json.dumps(
            {
                "path": path,
                "offset": offset,
                "next_offset": offset + len(data),
                "reset": False,
                "data": base64.b64encode(data).decode(),
            }
        )

    class Sandbox:
        def __init__(self) -> None:
            self.outputs = [
                record("agent/session.jsonl", 0, "first\npart"),
                record("agent/session.jsonl", 10, "ial\nsecond\n"),
            ]
            self.calls: list[tuple[list[str], float]] = []

        async def run_command(self, command: list[str], *, timeout: float) -> Any:
            self.calls.append((command, timeout))
            return type(
                "Result",
                (),
                {"exit_code": 0, "stdout": self.outputs.pop(0), "stderr": ""},
            )()

    class Logger:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def info(self, message: str, *args: object) -> None:
            self.lines.append(message % args)

    environment = _live_log_environment()()
    environment._sandbox = Sandbox()
    environment._live_log_offsets = {}
    environment._live_log_buffers = {}
    environment.logger = Logger()

    asyncio.run(environment._poll_sandbox_logs_once())
    assert environment.logger.lines == ["[sandbox agent/session.jsonl] first"]
    assert environment._live_log_buffers == {"agent/session.jsonl": "part"}

    asyncio.run(environment._poll_sandbox_logs_once())
    assert environment.logger.lines == [
        "[sandbox agent/session.jsonl] first",
        "[sandbox agent/session.jsonl] partial",
        "[sandbox agent/session.jsonl] second",
    ]
    assert environment._live_log_buffers == {}
    assert environment._live_log_offsets == {"agent/session.jsonl": 21}
    assert all(call[1] == 15.0 for call in environment._sandbox.calls)


def test_harbor_patch_starts_and_stops_live_sandbox_log_task() -> None:
    source = PATCH.read_text(encoding="utf-8")

    assert "asyncio.create_task(" in source
    assert "await self._stop_sandbox_log_stream()" in source
    assert 'name=f"sandbox-logs-{self._sandbox.name}"' in source
    assert "task.cancel()" in source
    assert "await self._poll_sandbox_logs_once()" in source


def test_harbor_patch_forwards_sidecars_to_k8s_sandbox() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_K8S_SANDBOX_PARAMS" for target in node.targets)
    ]

    assert len(assignments) == 1
    forwarded = ast.literal_eval(assignments[0].value)
    assert "sidecars" in forwarded


def test_harbor_patch_allows_profile_owned_tools_mount() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_HARBOR_PATH_ALLOW_PATTERNS" for target in node.targets)
    )

    patterns = ast.literal_eval(assignment.value)
    assert r"^/installed-tools(/|$)" in patterns
    assert r"^/git(/|$)" in patterns


def test_cross_service_evidence_directories_transfer() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_HARBOR_PATH_ALLOW_PATTERNS" for target in node.targets)
    )
    allow = ast.literal_eval(assignment.value)

    def allowed(path: str) -> bool:
        return any(re.search(pattern, path) for pattern in allow)

    # A multi-service task grades the record its service wrote to a shared
    # volume, and declares that file as a per-service artifact. The base list
    # has no entry for /audit at all, and allows /shared only with a trailing
    # slash -- so a task naming the directory itself could not download it.
    assert allowed("/audit/action_log.jsonl")
    assert allowed("/audit")
    assert allowed("/shared/verify_snapshot.json")
    assert allowed("/shared")

    # The widening is anchored, so a lookalike outside those roots stays out.
    assert not allowed("/auditor/secrets")
    assert not allowed("/sharedsecrets/key")


def test_task_owned_home_paths_transfer_but_agent_dotfiles_stay_denied() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    values = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"_HARBOR_PATH_ALLOW_PATTERNS", "_HARBOR_PATH_DENY_REPLACEMENTS"}
    }
    allow = values["_HARBOR_PATH_ALLOW_PATTERNS"]
    replacements = values["_HARBOR_PATH_DENY_REPLACEMENTS"]

    # The base list denies /root wholesale, which refuses a task's own working
    # tree; the replacement must keep the home itself and its dotfiles denied.
    deny: list[str] = []
    for pattern in ("^/etc", "^/proc", "^/root", "^/dev"):
        deny.extend(replacements.get(pattern, (pattern,)))

    def verdict(path: str) -> str:
        if any(re.search(pattern, path) for pattern in deny):
            return "denied"
        return "allowed" if any(re.search(pattern, path) for pattern in allow) else "not-allowed"

    assert verdict("/root/ico/ico_patched") == "allowed"
    assert verdict("/root/patch_ico.py") == "allowed"
    assert verdict("/root") == "denied"
    assert verdict("/root/.ssh/id_rsa") == "denied"
    assert verdict("/root/.config/provider/key") == "denied"
    assert verdict("/etc/passwd") == "denied"
    assert verdict("/proc/1/environ") == "denied"


def test_persistent_path_reaches_verifier_exec_unless_explicitly_overridden() -> None:
    merge_env = _environment_method("_merge_env")
    environment: Any = type("Environment", (), {})()
    environment._persistent_env = {"PATH": "/installed-tools/bin:/usr/bin", "HOME": "/app"}

    inherited = merge_env(environment, {"VERIFY": "1"})
    overridden = merge_env(environment, {"PATH": "/verifier/bin:/usr/bin"})

    assert inherited == {
        "PATH": "/installed-tools/bin:/usr/bin",
        "HOME": "/app",
        "VERIFY": "1",
    }
    assert overridden == {"PATH": "/verifier/bin:/usr/bin", "HOME": "/app"}


def test_harbor_patch_wires_task_gpu_mapping() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment"
    )
    constructor = next(
        node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    gpu_capability = next(
        node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "supports_gpus"
    )

    constructor_source = ast.unparse(constructor)
    assert "_gpu_resources" in constructor_source
    assert "_gpu_tolerations" in constructor_source
    assert any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is True
        for node in ast.walk(gpu_capability)
    )


def test_harbor_patch_prefers_task_local_image_for_verifier_only() -> None:
    effective_image = _image_helpers()["_effective_sandbox_image"]
    task_env = _Model(docker_image="registry.example/deepswe:environment")
    verifier_env = _Model(docker_image="registry.example/deepswe:verifier")

    assert (
        effective_image(
            session_id="task__abc",
            task_env_config=task_env,
            profile_image="registry.example/deepswe:signed-task",
        )
        == "registry.example/deepswe:signed-task"
    )
    assert (
        effective_image(
            session_id="task__abc__verifier__trial",
            task_env_config=verifier_env,
            profile_image="registry.example/deepswe:signed-task",
        )
        == "registry.example/deepswe:verifier"
    )
    assert (
        effective_image(
            session_id="task__abc",
            task_env_config=task_env,
            profile_image=None,
        )
        == "registry.example/deepswe:environment"
    )


def test_harbor_patch_uploads_public_overlay_before_setup() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment"
    )
    start = next(node for node in environment.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "start")
    calls = [
        ast.unparse(node.value.value.func)
        for node in start.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call)
    ]
    assert calls.index("self._upload_environment_public_files") < calls.index("self._run_setup_command")


def test_harbor_patch_upload_dir_extracts_with_system_tar(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "solution.txt").write_text("portable", encoding="utf-8")
    target = tmp_path / "target"

    class Sandbox:
        async def upload_file(self, content: bytes, path: str) -> None:
            Path(path).write_bytes(content)

        async def run_command(self, command: list[str]) -> Any:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            return _Model(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    environment: Any = type("Environment", (), {})()
    environment._sandbox = Sandbox()

    asyncio.run(_environment_method("upload_dir")(environment, source, str(target)))

    assert (target / "solution.txt").read_text(encoding="utf-8") == "portable"


def test_harbor_patch_retries_start_after_sandbox_creation_conflict() -> None:
    source = PATCH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment"
    )
    start = next(node for node in environment.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "start")
    calls = [
        ast.unparse(node.value.value.func)
        for node in ast.walk(start)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call)
    ]

    assert "_is_sandbox_creation_conflict" in source
    assert calls.count("self._sandbox.start") == 2
    assert "await _delete_stale_sandbox(self._sandbox)" in source
    assert "await self._sandbox.stop()" not in ast.unparse(start)
    assert "deleting stale resource before retry" in source


def test_harbor_patch_deletes_conflict_without_stopping_new_sandbox() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    helpers: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in {"_sanitize_k8s_name", "_delete_stale_sandbox"}
    ]
    namespace: dict[str, Any] = {"Any": Any, "re": re, "suppress": suppress}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(PATCH), "exec"), namespace)

    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    class Backend:
        _name = "hbr-task-real"
        _namespace = "evals"
        _client = _Model(delete_network_policy="delete-policy", delete_sandbox="delete-sandbox")

        async def _run_sync(self, function, *args, **kwargs):  # noqa: ANN001, ANN202
            calls.append((function, args, kwargs))

    asyncio.run(namespace["_delete_stale_sandbox"](_Model(_backend=Backend())))

    assert calls == [
        ("delete-policy", ("hbr-task-real-network-policy", "evals"), {}),
        (
            "delete-sandbox",
            ("hbr-task-real", "evals"),
            {"wait": True, "wait_timeout": 60.0},
        ),
    ]


def test_harbor_patch_activates_rootless_overlay_after_setup() -> None:
    source = PATCH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment"
    )
    start = next(node for node in environment.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "start")
    calls = [
        ast.unparse(node.value.value.func)
        for node in start.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call)
    ]
    assert calls.index("self._run_setup_command") < calls.index("self._activate_rootless_overlay")
    assert "rootless_overlay owns container_command" in source
    assert "rootless_overlay requires sandbox_k8s direct CRD mode" in source
    assert "rootless_overlay is incompatible with full pod_spec overrides" in source
    assert 'os.environ["SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS"] = "true"' in source
    assert 'os.environ["SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK"] = "true"' in source
    # The marker is process-global and the runner builds two environments in
    # separate-verifier mode, so it must be cleared before each construction.
    assert 'os.environ.pop("SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK", None)' in source
    assert "enable_ipv6_loopback must be a boolean" in source
    assert '"/installed-tools/bin/rootless-client"' in source
    assert "_add_rootless_crd_mounts(volumes, volume_mounts)" in source
    assert '("rootless-git", "/git")' in source


def test_gpu_resources_leave_cpu_task_unchanged() -> None:
    resources = _ResourceRequirements(requests={"cpu": "1"}, limits={"cpu": "2"})

    result = _gpu_helpers()["_gpu_resources"](resources, 0)

    assert result is resources
    assert "nvidia.com/gpu" not in result.requests


def test_gpu_count_defaults_omitted_or_null_to_cpu_only() -> None:
    normalize = _gpu_helpers()["_normalize_gpu_count"]

    assert normalize(None) == 0
    assert normalize(0) == 0


def test_command_timeout_defaults_to_sandbox_startup_timeout() -> None:
    resolve = _gpu_helpers()["_resolve_command_timeout"]

    assert resolve(None, 2100) == 2100.0
    assert resolve(4200, 2100) == 4200.0


@pytest.mark.parametrize("value", [True, "4200", 0, -1, -1.0, float("inf"), float("nan")])
def test_command_timeout_rejects_invalid_values(value: object) -> None:
    with pytest.raises(_SandboxConfigError, match="positive number"):
        _gpu_helpers()["_resolve_command_timeout"](value, 2100)


@pytest.mark.parametrize("value", [True, False, "1", 1.0, -1])
def test_gpu_count_rejects_invalid_values(value: object) -> None:
    with pytest.raises(_SandboxConfigError):
        _gpu_helpers()["_normalize_gpu_count"](value)


@pytest.mark.parametrize("gpu_count", [1, 2])
def test_gpu_resources_add_equal_request_and_limit(gpu_count: int) -> None:
    result = _gpu_helpers()["_gpu_resources"](
        {"requests": {"cpu": "1"}, "limits": {"memory": "4Gi"}},
        gpu_count,
    )

    expected = str(gpu_count)
    assert result.requests == {"cpu": "1", "nvidia.com/gpu": expected}
    assert result.limits == {"memory": "4Gi", "nvidia.com/gpu": expected}


def test_gpu_resources_reject_conflicting_override() -> None:
    with pytest.raises(_SandboxConfigError, match="task requests 1 GPU"):
        _gpu_helpers()["_gpu_resources"](
            {"requests": {"nvidia.com/gpu": "2"}},
            1,
        )


def test_gpu_resources_reject_negative_count() -> None:
    with pytest.raises(_SandboxConfigError, match="zero or a positive integer"):
        _gpu_helpers()["_gpu_resources"](None, -1)


def test_gpu_toleration_is_opt_in_and_deduplicated() -> None:
    helper = _gpu_helpers()["_gpu_tolerations"]
    cpu_tolerations: list[_Toleration] = []

    assert helper(cpu_tolerations, 0) is cpu_tolerations

    gpu_tolerations = helper([], 1)
    assert [(item.key, item.operator, item.effect) for item in gpu_tolerations] == [
        ("nvidia.com/gpu", "Exists", "NoSchedule")
    ]
    assert helper(gpu_tolerations, 1) == gpu_tolerations


def test_harbor_patch_coerces_profile_volume_mappings() -> None:
    source = PATCH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}

    assert "_coerce_volume" in functions
    assert "_coerce_volume_mount" in functions
    assert "_coerce_volume(value)" in source
    assert "_coerce_volume_mount(value)" in source
    assert 'config["empty_dir"] = EmptyDirVolume(**dict(empty_dir))' in source


def test_harbor_patch_has_configurable_sidecar_readiness_timeout() -> None:
    source = PATCH.read_text(encoding="utf-8")

    assert "sidecar_wait_timeout: int = 120" in source
    assert "sidecar_wait_timeout must be a positive integer" in source
    assert "deadline = time.monotonic() + self._sidecar_wait_timeout" in source
    assert "timeout=10" in source
    assert "await asyncio.sleep(2)" in source
    assert "one long Kubernetes exec" in source


def test_harbor_patch_separates_startup_and_default_exec_timeouts() -> None:
    source = PATCH.read_text(encoding="utf-8")

    assert "self._timeout = timeout" in source
    assert "self._command_timeout = _resolve_command_timeout(command_timeout, timeout)" in source
    assert '"timeout": timeout' in source
    assert "effective_timeout = float(timeout_sec) if timeout_sec else self._command_timeout" in source
    assert "effective_timeout = timeout or self._command_timeout" in source
    assert "timeout=effective_timeout" in source


def test_rootless_command_uses_independent_default_timeout() -> None:
    class Sandbox:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], dict[str, object]]] = []

        async def run_command(self, command: list[str], **kwargs: object) -> object:
            self.calls.append((command, kwargs))
            return object()

    sandbox = Sandbox()
    environment: Any = type("Environment", (), {})()
    environment._rootless_active = True
    environment._command_timeout = 4200.0
    environment._working_dir = "/app"
    environment._sandbox = sandbox

    asyncio.run(
        _environment_method("_run_sandbox_command")(
            environment,
            ["bash", "-c", "pytest /tests"],
        )
    )

    command, kwargs = sandbox.calls[0]
    assert command[0] == "/installed-tools/bin/rootless-client"
    timeout_index = command.index("--timeout")
    assert command[timeout_index + 1] == "4200.0"
    assert kwargs["timeout"] == 4220.0


def _init_body() -> list[ast.stmt]:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K8sSandboxEnvironment")
    return next(node.body for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")


def test_the_verifier_pod_gets_no_duplicate_copy_of_the_task_services() -> None:
    body = _init_body()

    declared_at = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_sidecar_containers" for target in statement.targets
        )
    )
    dropped_at, guard = next(
        (index, statement)
        for index, statement in enumerate(body)
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_is_verifier_sandbox"
            for node in ast.walk(statement.test)
        )
    )

    # Order is the subtle part. `_sidecar_containers` is what the docker_compose
    # capability is read from, so dropping the sidecars before it is built would
    # make the verifier environment deny compose support -- and Harbor rejects
    # any task that references services against an environment that denies it,
    # so every multi-service task would stop running at all.
    assert declared_at < dropped_at

    dropped = ast.dump(ast.Module(body=guard.body, type_ignores=[]))
    assert "'sidecars'" in dropped
    assert "pop" in dropped
    # The port wait and log collection have to go with them, or the verifier
    # blocks for the full sidecar timeout waiting on ports nothing will bind.
    for cleared in ("sidecar_wait_ports", "sidecar_log_containers"):
        assert cleared in dropped


def test_the_verifier_is_recognized_from_the_name_that_carries_the_phase() -> None:
    is_verifier = _image_helpers()["_is_verifier_sandbox"]

    # Real names observed from Harbor 0.13 and 0.20 runs. The phase appears in
    # the session id and nowhere else; `environment_name` is the task's
    # environment directory and reads identically for both phases, so feeding it
    # here classifies every verifier as an agent -- which is silent, because the
    # verifier then boots against a duplicate of the task's own services.
    assert is_verifier("task__FFybNYn__verifier__trial")
    assert is_verifier("task__SaCaatF__verifier__trial")
    assert not is_verifier("task__FFybNYn__env")
    assert not is_verifier("task__SaCaatF")
    assert not is_verifier("erp-procurement-planning")


def test_every_caller_asks_about_the_session_id() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_is_verifier_sandbox"
    ]
    assert calls
    for call in calls:
        (argument,) = call.args
        assert isinstance(argument, ast.Name), ast.dump(argument)
        assert argument.id == "session_id", argument.id
