# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import socket
import threading
import time
from http.server import HTTPServer, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import pytest
from sandboxed_gym.environment_package import WHEELS_V1_SUBDIR
from sandboxed_gym.runtime import gym_host_runtime as runtime


class _FakeRolloutHelper:
    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            return row, {"response": {"output": []}, "reward": 0.0}

        return [_one(row) for row in examples]


@pytest.fixture
def ready_server():
    runtime._READY = True
    runtime._RUN_HELPER = MagicMock()
    runtime._HEAD_SERVER_CONFIG = MagicMock()
    runtime._ROLLOUT_HELPER = _FakeRolloutHelper()
    runtime.Handler.max_request_bytes = 1024
    runtime.Handler.max_response_bytes = 4096
    runtime.Handler.heartbeat_interval_s = runtime._HEARTBEAT_INTERVAL_S
    runtime.Handler.rollout_deadline_s = runtime._DEFAULT_ROLLOUT_DEADLINE_S

    server = HTTPServer(("127.0.0.1", 0), runtime.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        runtime._READY = False
        runtime._HEAD_SERVER_CONFIG = None
        runtime._ROLLOUT_HELPER = None
        runtime.Handler.heartbeat_interval_s = runtime._HEARTBEAT_INTERVAL_S
        runtime.Handler.rollout_deadline_s = runtime._DEFAULT_ROLLOUT_DEADLINE_S


def test_health_not_ready():
    runtime._READY = False
    server = HTTPServer(("127.0.0.1", 0), runtime.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import urllib.error
        import urllib.request

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
        assert exc.value.code == 503
        body = json.loads(exc.value.read().decode())
        assert body["status"] == "starting"
    finally:
        server.shutdown()
        server.server_close()


def test_health_is_not_ready_until_the_rollout_helpers_are(ready_server):
    """/health must gate on what do_POST gates on.

    Reporting ready off ``_READY`` alone lets a host pass ``wait_ready`` and then refuse every
    rollout with 503 bootstrap_failed -- a state the caller has no way to observe until it has
    already committed a batch.
    """
    import urllib.error
    import urllib.request

    runtime._ROLLOUT_HELPER = None
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{ready_server}/health", timeout=5)
        assert excinfo.value.code == 503
        assert json.loads(excinfo.value.read())["status"] == "starting"
    finally:
        runtime._ROLLOUT_HELPER = _FakeRolloutHelper()


def test_health_ready(ready_server):
    import urllib.request

    with urllib.request.urlopen(f"{ready_server}/health", timeout=5) as resp:
        assert resp.status == 200
        assert json.loads(resp.read().decode()) == {"status": "ready"}


def test_rollouts_run_returns_results(ready_server):
    import urllib.request

    payload = json.dumps({"examples": [{"agent_ref": {"name": "a"}, "id": 1}]}).encode()
    req = urllib.request.Request(
        f"{ready_server}/rollouts/run",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    assert len(body["results"]) == 1
    assert body["results"][0]["reward"] == 0.0


def test_rollouts_run_rejects_oversize_request(ready_server):
    import urllib.error
    import urllib.request

    payload = b"x" * 2048
    req = urllib.request.Request(
        f"{ready_server}/rollouts/run",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 413
    err = json.loads(exc.value.read().decode())
    assert err["error"]["code"] == "payload_too_large"


def test_a_malformed_content_length_answers_instead_of_dropping_the_connection(ready_server):
    """`int()` on a junk header would raise out of `do_POST`, and the client would see a closed
    socket rather than the error envelope every other rejection path uses."""
    parsed = urlsplit(ready_server)
    with socket.create_connection((parsed.hostname, parsed.port), timeout=5) as sock:
        sock.sendall(
            b"POST /rollouts/run HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: not-a-number\r\n"
            b"\r\n"
        )
        raw = b""
        while b"}" not in raw:
            chunk = sock.recv(4096)
            if not chunk:
                break
            raw += chunk

    head, _, body = raw.partition(b"\r\n\r\n")
    assert b" 400 " in head.splitlines()[0]
    assert json.loads(body.decode())["error"]["code"] == "internal"


def test_run_rollouts_sync_collects():
    helper = _FakeRolloutHelper()
    results = runtime.run_rollouts_sync(
        [{"agent_ref": {"name": "x"}}],
        MagicMock(),
        helper,
    )
    assert len(results) == 1


def test_apply_uv_dirs_sets_config_keys_in_container(monkeypatch):
    """Gym reads the CONFIG keys, not the env vars - the env alone gets overwritten."""
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "/opt/uv_cache\n")

    config: dict = {}
    runtime._apply_uv_dirs(config)

    assert config[runtime.UV_CACHE_DIR_KEY] == "/opt/uv_cache"
    assert config[runtime.UV_VENV_DIR_KEY] == "/opt/gym_venvs"


def test_apply_uv_dirs_noop_outside_container(monkeypatch):
    monkeypatch.delenv("NRL_CONTAINER", raising=False)
    monkeypatch.delenv("NEMO_GYM_VENV_DIR", raising=False)

    config: dict = {}
    runtime._apply_uv_dirs(config)

    assert config == {}


def test_apply_uv_dirs_does_not_override_explicit_config(monkeypatch):
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "/opt/uv_cache\n")

    config = {runtime.UV_CACHE_DIR_KEY: "/custom/cache"}
    runtime._apply_uv_dirs(config)

    assert config[runtime.UV_CACHE_DIR_KEY] == "/custom/cache"
    assert config[runtime.UV_VENV_DIR_KEY] == "/opt/gym_venvs"


def test_uv_cache_dir_prefers_the_configured_env_var(monkeypatch):
    """`uv cache dir` exits non-zero when the CWD's pyproject pins a conflicting
    [tool.uv] required-version - true in the nemo-platform image - so an explicit
    UV_CACHE_DIR must win without shelling out at all."""
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("UV_CACHE_DIR", "/home/ubuntu/.cache/uv")

    def _never(*a, **k):
        raise AssertionError("uv should not be invoked when UV_CACHE_DIR is set")

    monkeypatch.setattr(runtime.subprocess, "check_output", _never)

    assert runtime._uv_cache_dir() == "/home/ubuntu/.cache/uv"


def test_uv_cache_dir_returns_none_when_uv_unavailable(monkeypatch):
    """No env var and no usable `uv`: let Gym pick its own rather than crash."""
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)

    def _boom(*a, **k):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(runtime.subprocess, "check_output", _boom)

    assert runtime._uv_cache_dir() is None


# --------------------------------------------------------------------------------------------
# Rollout attribution
#
# The question this answers: if an evaluation posts examples here instead of driving the `gym`
# CLI, can each result still be tied back to the task that produced it? Attribution by position
# is not enough -- a reordering misattributes every reward silently -- and attribution by
# `_ng_task_index` alone depends on Gym continuing to copy that field through.
# --------------------------------------------------------------------------------------------


class _IdentityStrippingHelper:
    """A helper whose results carry no index, i.e. Gym did not copy the caller's stamp through."""

    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            return row, {"response": {"output": []}, "reward": 0.5}

        return [_one(row) for row in examples]


class _IdentityPreservingHelper:
    """A helper whose results carry Gym's own indices, which must win over the row's."""

    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            return row, {"reward": 1.0, "_ng_task_index": 99, "_ng_rollout_index": 7}

        return [_one(row) for row in examples]


def test_results_are_attributable_when_gym_drops_the_caller_stamp():
    # Two examples with distinct indices: the failure this guards against is both results coming
    # back indistinguishable, which reads as success and scores the wrong task.
    examples = [
        {"agent_ref": {"name": "a"}, "_ng_task_index": 0},
        {"agent_ref": {"name": "b"}, "_ng_task_index": 1},
    ]

    results = runtime.run_rollouts_sync(examples, MagicMock(), _IdentityStrippingHelper())

    assert [result["_ng_task_index"] for result in results] == [0, 1]


def test_an_index_gym_supplies_is_not_overwritten_by_the_row():
    # Gym assigns `_ng_rollout_index` itself, per attempt. Ours is a fallback, never an override:
    # clobbering Gym's would collapse repeats of one task onto a single trial.
    examples = [{"agent_ref": {"name": "a"}, "_ng_task_index": 3, "_ng_rollout_index": 0}]

    results = runtime.run_rollouts_sync(examples, MagicMock(), _IdentityPreservingHelper())

    assert results[0]["_ng_task_index"] == 99
    assert results[0]["_ng_rollout_index"] == 7


def test_examples_without_identity_are_passed_through_unchanged():
    # Customizer posts examples with no `_ng_*` fields at all; nothing may be invented for them.
    results = runtime.run_rollouts_sync([{"agent_ref": {"name": "a"}}], MagicMock(), _IdentityStrippingHelper())

    assert results[0] == {"response": {"output": []}, "reward": 0.5}


def test_identity_survives_the_http_boundary(ready_server):
    """The path an eval would actually take: POST examples, read attributable results back."""
    import urllib.request

    payload = json.dumps(
        {
            "examples": [
                {"agent_ref": {"name": "a"}, "_ng_task_index": 4},
                {"agent_ref": {"name": "b"}, "_ng_task_index": 9},
            ]
        }
    ).encode()
    runtime.Handler.max_request_bytes = 8192
    req = urllib.request.Request(
        f"{ready_server}/rollouts/run",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())

    assert [result["_ng_task_index"] for result in body["results"]] == [4, 9]


@pytest.mark.parametrize("result", ["not-a-dict", None, 42])
def test_a_non_mapping_result_is_left_alone(result):
    # The helper's result shape is Gym's to define; this must not assume it is always a dict.
    assert runtime._with_row_identity(result, {"_ng_task_index": 1}) is result


def test_a_caller_example_id_is_carried_onto_the_result():
    """The case Gym's own indices cannot express: several examples under one task.

    Both examples below share ``_ng_task_index``, so a consumer joining on it sees a duplicate
    and cannot tell the two apart. ``SG_EXAMPLE_ID`` is what distinguishes them, and Gym never
    sets it, so a result carries it only because this host copied it across.
    """
    rows = [
        {"_ng_task_index": 4, runtime.SG_EXAMPLE_ID: 17},
        {"_ng_task_index": 4, runtime.SG_EXAMPLE_ID: 18},
    ]
    results = [runtime._with_row_identity({"_ng_task_index": 4, "reward": 1.0}, row) for row in rows]

    assert [result[runtime.SG_EXAMPLE_ID] for result in results] == [17, 18]
    # Gym's own identity is passed through untouched, not displaced by the caller's.
    assert [result["_ng_task_index"] for result in results] == [4, 4]


def test_a_caller_example_id_returned_by_gym_is_not_overwritten():
    """Same additive rule as the Gym indices: what came back wins over what went out."""
    result = runtime._with_row_identity({runtime.SG_EXAMPLE_ID: 99}, {runtime.SG_EXAMPLE_ID: 17})

    assert result[runtime.SG_EXAMPLE_ID] == 99


def test_an_example_without_a_caller_id_gains_no_key():
    """Existing callers stamp nothing, so their results must look exactly as they did before."""
    result = runtime._with_row_identity({"reward": 1.0}, {"_ng_task_index": 4})

    assert runtime.SG_EXAMPLE_ID not in result


# --------------------------------------------------------------------------------------------
# wheels-v1 dependency install
# --------------------------------------------------------------------------------------------


def _write_manifest(env_dir, **fields):
    import yaml

    fields.setdefault("config_paths", ["resources_servers/test/configs/test.yaml"])
    fields.setdefault("metadata", {"name": "test-environment"})
    for relative_path in fields["config_paths"]:
        config = env_dir / relative_path
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("test: {}\n", encoding="utf-8")
    manifest_path = env_dir / runtime.ENVIRONMENT_MANIFEST_FILENAME
    manifest_path.write_text(yaml.safe_dump(fields), encoding="utf-8")


def _load_composition_package(env_dir, *, agents=(), resources_servers=(), package_format="native-v1"):
    import yaml

    config_paths = [
        *(f"responses_api_agents/{name}/configs/{name}.yaml" for name in agents),
        *(f"resources_servers/{name}/configs/{name}.yaml" for name in resources_servers),
    ]
    _write_manifest(env_dir, format=package_format, config_paths=config_paths)
    for name, relative_path in zip((*agents, *resources_servers), config_paths, strict=True):
        component_type = (
            "responses_api_agents" if relative_path.startswith("responses_api_agents/") else "resources_servers"
        )
        (env_dir / relative_path).write_text(
            yaml.safe_dump({name: {component_type: {f"{name}_implementation": {"entrypoint": "app.py"}}}}),
            encoding="utf-8",
        )
    if package_format == "wheels-v1":
        wheels_dir = env_dir / WHEELS_V1_SUBDIR
        wheels_dir.mkdir(exist_ok=True)
        (wheels_dir / "example_dependency-1.0-py3-none-any.whl").write_bytes(b"")
    return runtime._load_runtime_environment_package(str(env_dir), required=True)


def _component_selection(*, agent_config):
    return {
        runtime.ENVIRONMENT_COMPONENT_SELECTION_CONFIG_KEY: {
            "agent_instance": "selected_agent",
            "agent_config": agent_config,
            "resources_server_instance": "selected_resources",
            "resources_server_config": "resources_servers/selected_resources/configs/selected_resources.yaml",
            "model_config": "responses_api_models/inference_provider/configs/inference_provider.yaml",
        }
    }


@pytest.mark.parametrize(
    ("agents", "resources_servers", "agent_config", "expected_fallbacks"),
    [
        (("selected_agent",), ("selected_resources",), None, []),
        (
            ("selected_agent",),
            ("other_resources",),
            None,
            ["resources_servers/selected_resources/configs/selected_resources.yaml"],
        ),
        (
            ("other_agent",),
            ("selected_resources",),
            "responses_api_agents/selected_agent/configs/selected_agent.yaml",
            ["responses_api_agents/selected_agent/configs/selected_agent.yaml"],
        ),
        (
            ("other_agent",),
            ("other_resources",),
            "responses_api_agents/selected_agent/configs/selected_agent.yaml",
            [
                "responses_api_agents/selected_agent/configs/selected_agent.yaml",
                "resources_servers/selected_resources/configs/selected_resources.yaml",
            ],
        ),
    ],
    ids=["all-custom", "custom-agent", "custom-resources", "all-built-in"],
)
def test_environment_config_composes_custom_components_with_builtin_fallbacks(
    tmp_path,
    agents,
    resources_servers,
    agent_config,
    expected_fallbacks,
):
    package = _load_composition_package(tmp_path, agents=agents, resources_servers=resources_servers)
    global_config = _component_selection(agent_config=agent_config)
    original_config = dict(global_config)

    configured = runtime._compose_gym_config_with_environment_package(global_config, package)

    package_paths = [str(path) for path in package.config_paths]
    model_config = "responses_api_models/inference_provider/configs/inference_provider.yaml"
    expected = [
        *package_paths,
        *(path for path in expected_fallbacks if path.startswith("responses_api_agents/")),
        model_config,
        *(path for path in expected_fallbacks if path.startswith("resources_servers/")),
    ]
    assert configured["config_paths"] == expected
    assert runtime.ENVIRONMENT_COMPONENT_SELECTION_CONFIG_KEY not in configured
    assert configured is not global_config
    assert global_config == original_config


def test_environment_config_requires_agent_fallback_when_selected_agent_is_absent(tmp_path):
    package = _load_composition_package(tmp_path, agents=("other_agent",), resources_servers=("selected_resources",))
    global_config = _component_selection(agent_config=None)
    original_config = dict(global_config)

    with pytest.raises(RuntimeError, match="no built-in agent_config fallback"):
        runtime._compose_gym_config_with_environment_package(global_config, package)
    assert global_config == original_config


def test_no_manifest_is_a_no_op(tmp_path, monkeypatch):
    # Image-provided environments have no manifest and must retain the existing startup behavior.
    assert runtime._load_runtime_environment_package(str(tmp_path), required=False) is None


def test_compose_without_package_returns_a_copy():
    global_config = {"config_paths": ["image-owned.yaml"]}

    configured = runtime._compose_gym_config_with_environment_package(global_config, None)

    assert configured == global_config
    assert configured is not global_config


def test_no_environment_path_is_a_no_op(monkeypatch):
    # Curated Gym configurations can run without mounting a custom environment.
    assert runtime._load_runtime_environment_package("", required=False) is None


def test_explicit_environment_requires_a_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="missing required manifest"):
        runtime._load_runtime_environment_package(str(tmp_path), required=True)


def test_native_v1_is_loaded_without_installing_wheels(tmp_path, monkeypatch):
    _write_manifest(tmp_path, format="native-v1")

    package = runtime._load_runtime_environment_package(str(tmp_path), required=True)

    assert package is not None
    assert package.manifest.format == "native-v1"

    def fail_install(*args, **kwargs):
        pytest.fail("native-v1 must not install wheels")

    monkeypatch.setattr(runtime.subprocess, "run", fail_install)
    runtime._install_wheels_v1_dependencies(package, str(tmp_path / "work"))


def test_wheels_v1_with_no_wheels_directory_raises(tmp_path):
    # Declaring wheels-v1 without its required wheelhouse indicates an incomplete bundle.
    _write_manifest(tmp_path, format="wheels-v1")

    with pytest.raises(RuntimeError, match="invalid Gym environment package"):
        runtime._load_runtime_environment_package(str(tmp_path), required=True)


def test_wheels_v1_with_an_empty_wheels_directory_raises(tmp_path):
    # An empty wheelhouse is invalid for the same reason as a missing one.
    _write_manifest(tmp_path, format="wheels-v1")
    (tmp_path / WHEELS_V1_SUBDIR).mkdir()

    with pytest.raises(RuntimeError, match="non-empty wheels/ directory"):
        runtime._load_runtime_environment_package(str(tmp_path), required=True)


def test_malformed_manifest_fails_before_gym_startup(tmp_path):
    (tmp_path / runtime.ENVIRONMENT_MANIFEST_FILENAME).write_text("format: [native-v1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="nemo-environment.yaml is not valid YAML"):
        runtime._load_runtime_environment_package(str(tmp_path), required=True)


def test_wheels_v1_installs_every_wheel_with_no_index_access(tmp_path, monkeypatch, isolated_gym_host_process_state):
    # Model the fixed bundle layout: manifest at the root and wheels in the sibling `wheels/`.
    _write_manifest(tmp_path, format="wheels-v1")
    wheels_dir = tmp_path / WHEELS_V1_SUBDIR
    work_dir = tmp_path / "work"
    wheels_dir.mkdir()
    (wheels_dir / "a_dep-1.0-py3-none-any.whl").write_bytes(b"")
    (wheels_dir / "b_dep-2.0-py3-none-any.whl").write_bytes(b"")

    # Capture the uv command without trying to install these intentionally empty wheel fixtures.
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setenv("PYTHONPATH", "/image/packages")

    package = runtime._load_runtime_environment_package(str(tmp_path), required=True)
    runtime._install_wheels_v1_dependencies(package, str(work_dir))

    assert len(calls) == 1
    (args,), kwargs = calls[0]
    install_dir = work_dir / runtime.WHEELS_V1_INSTALL_SUBDIR
    assert args[:10] == [
        "uv",
        "pip",
        "install",
        "--target",
        str(install_dir),
        "--no-index",
        "--only-binary",
        ":all:",
        "--find-links",
        str(wheels_dir),
    ]
    # Every .whl is installed explicitly.
    assert args[10:] == [
        str(wheels_dir / "a_dep-1.0-py3-none-any.whl"),
        str(wheels_dir / "b_dep-2.0-py3-none-any.whl"),
    ]
    assert kwargs == {"check": True}
    assert install_dir.is_dir()
    assert runtime.os.environ[runtime.UV_FIND_LINKS_ENV_KEY] == str(wheels_dir)
    # Child processes and the already-running host both prefer staged packages over image packages.
    assert runtime.os.environ["PYTHONPATH"] == f"{install_dir}{runtime.os.pathsep}/image/packages"
    assert runtime.sys.path[0] == str(install_dir)
    # Index fallback is left in place unless the caller declares the wheelhouse self-sufficient.
    assert runtime.UV_OFFLINE_ENV_KEY not in runtime.os.environ


def _stage_wheelhouse(tmp_path, monkeypatch):
    """A wheels-v1 package staged for install, with uv stubbed out."""
    _write_manifest(tmp_path, format="wheels-v1")
    wheels_dir = tmp_path / WHEELS_V1_SUBDIR
    wheels_dir.mkdir()
    (wheels_dir / "a_dep-1.0-py3-none-any.whl").write_bytes(b"")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: None)
    return runtime._load_runtime_environment_package(str(tmp_path), required=True)


def test_an_offline_environment_takes_uv_off_the_index(tmp_path, monkeypatch, isolated_gym_host_process_state):
    """A wheelhouse the caller calls complete must resolve without an index.

    `--find-links` alone is not enough: a configured but unreachable index makes uv fail to
    resolve the `uv venv --seed` packages for Gym's per-component venvs rather than fall back
    to the staged wheels.
    """
    package = _stage_wheelhouse(tmp_path, monkeypatch)
    monkeypatch.setenv(runtime.ENVIRONMENT_OFFLINE_ENV_KEY, "true")

    runtime._install_wheels_v1_dependencies(package, str(tmp_path / "work"))

    assert runtime.os.environ[runtime.UV_OFFLINE_ENV_KEY] == "1"


def test_an_operators_own_uv_offline_setting_is_not_overwritten(tmp_path, monkeypatch, isolated_gym_host_process_state):
    """Whoever set it in the image knows something this flag does not."""
    package = _stage_wheelhouse(tmp_path, monkeypatch)
    monkeypatch.setenv(runtime.ENVIRONMENT_OFFLINE_ENV_KEY, "true")
    monkeypatch.setenv(runtime.UV_OFFLINE_ENV_KEY, "0")

    runtime._install_wheels_v1_dependencies(package, str(tmp_path / "work"))

    assert runtime.os.environ[runtime.UV_OFFLINE_ENV_KEY] == "0"


def test_bootstrap_composes_a_wheels_package_like_native_v1(tmp_path, monkeypatch):
    # wheels-v1 and native-v1 are the same environment; only dependency install differs.
    package = _load_composition_package(
        tmp_path,
        agents=("selected_agent",),
        resources_servers=("selected_resources",),
        package_format="wheels-v1",
    )
    captured: dict[str, object] = {}

    class FakeRunHelper:
        def start(self, config):
            captured["config"] = config

    class FakeBaseServerConfig:
        def __init__(self, *, host, port):
            self.host = host
            self.port = port

    fake_modules = {
        "nemo_gym": ModuleType("nemo_gym"),
        "nemo_gym.cli": ModuleType("nemo_gym.cli"),
        "nemo_gym.cli.env": ModuleType("nemo_gym.cli.env"),
        "nemo_gym.global_config": ModuleType("nemo_gym.global_config"),
        "nemo_gym.server_utils": ModuleType("nemo_gym.server_utils"),
        "omegaconf": ModuleType("omegaconf"),
    }
    setattr(fake_modules["nemo_gym.cli.env"], "RunHelper", FakeRunHelper)
    setattr(fake_modules["nemo_gym.global_config"], "GlobalConfigDictParserConfig", lambda **kwargs: kwargs)
    setattr(fake_modules["nemo_gym.server_utils"], "BaseServerConfig", FakeBaseServerConfig)
    setattr(fake_modules["omegaconf"], "DictConfig", lambda value: value)
    for name, module in fake_modules.items():
        monkeypatch.setitem(runtime.sys.modules, name, module)

    caller_agent_config = "responses_api_agents/simple_agent/configs/simple_agent.yaml"
    model_config = "responses_api_models/inference_provider/configs/inference_provider.yaml"
    monkeypatch.setattr(
        runtime,
        "_load_global_config_dict",
        lambda: {
            "config_paths": [caller_agent_config],
            **_component_selection(agent_config=caller_agent_config),
        },
    )
    monkeypatch.setattr(runtime, "_apply_uv_dirs", lambda config: None)
    monkeypatch.setattr(runtime, "_allocate_head_server_port", lambda config: 5000)
    monkeypatch.setattr(runtime, "_create_rollout_helper", lambda: "rollout-helper")
    monkeypatch.setattr(runtime, "_install_wheels_v1_dependencies", lambda loaded_package, work_path: None)
    monkeypatch.setenv("NMP_ENVIRONMENT_PATH", str(tmp_path))
    monkeypatch.setenv(runtime.ENVIRONMENT_PACKAGE_REQUIRED_ENV_KEY, "true")
    monkeypatch.setenv("NMP_WORK_PATH", str(tmp_path / "work"))
    monkeypatch.setenv(runtime.NEMO_GYM_EXTRA_ROOTS_ENV_KEY, "/operator/root")

    runtime.bootstrap_gym_host()

    config = cast(dict[str, Any], captured["config"])
    composed = cast(dict[str, Any], config["initial_global_config_dict"])
    assert composed["config_paths"] == [
        *[str(path) for path in package.config_paths],
        model_config,
    ]
    assert caller_agent_config not in composed["config_paths"]
    assert runtime.ENVIRONMENT_COMPONENT_SELECTION_CONFIG_KEY not in composed
    assert runtime.os.environ[runtime.NEMO_GYM_EXTRA_ROOTS_ENV_KEY] == f"{tmp_path.resolve()}:/operator/root"


def test_bootstrap_installs_wheels_before_starting_gym(monkeypatch):
    # Record bootstrap milestones so the assertion verifies ordering rather than only invocation.
    events = []

    class FakeRunHelper:
        def start(self, config):
            events.append("gym-started")

    class FakeBaseServerConfig:
        def __init__(self, *, host, port):
            self.host = host
            self.port = port

    # NeMo-Gym is supplied by the runtime image rather than this test environment, so provide only
    # the small API surface bootstrap imports.
    fake_modules = {
        "nemo_gym": ModuleType("nemo_gym"),
        "nemo_gym.cli": ModuleType("nemo_gym.cli"),
        "nemo_gym.cli.env": ModuleType("nemo_gym.cli.env"),
        "nemo_gym.global_config": ModuleType("nemo_gym.global_config"),
        "nemo_gym.server_utils": ModuleType("nemo_gym.server_utils"),
        "omegaconf": ModuleType("omegaconf"),
    }
    setattr(fake_modules["nemo_gym.cli.env"], "RunHelper", FakeRunHelper)
    setattr(fake_modules["nemo_gym.global_config"], "GlobalConfigDictParserConfig", lambda **kwargs: kwargs)
    setattr(fake_modules["nemo_gym.server_utils"], "BaseServerConfig", FakeBaseServerConfig)
    setattr(fake_modules["omegaconf"], "DictConfig", lambda value: value)
    for name, module in fake_modules.items():
        monkeypatch.setitem(runtime.sys.modules, name, module)

    monkeypatch.setattr(runtime, "_load_global_config_dict", lambda: {})
    monkeypatch.setattr(runtime, "_apply_uv_dirs", lambda config: None)
    monkeypatch.setattr(runtime, "_allocate_head_server_port", lambda config: 5000)
    monkeypatch.setattr(runtime, "_create_rollout_helper", lambda: "rollout-helper")
    package = MagicMock(root="/job/environment")
    monkeypatch.setattr(
        runtime,
        "_load_runtime_environment_package",
        lambda environment_path, required: events.append("environment-validated") or package,
    )
    monkeypatch.setattr(
        runtime,
        "_compose_gym_config_with_environment_package",
        lambda config, loaded_package: (
            (
                events.append("environment-composed"),
                config,
            )[1]
            if loaded_package is package
            else config
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_prepend_environment_search_root",
        lambda environment_root: events.append("environment-root-prepended"),
    )
    monkeypatch.setattr(
        runtime,
        "_install_wheels_v1_dependencies",
        lambda loaded_package, work_path: events.append("wheels-installed") if loaded_package is package else None,
    )
    monkeypatch.setenv("NMP_ENVIRONMENT_PATH", "/job/environment")
    monkeypatch.setenv(runtime.ENVIRONMENT_PACKAGE_REQUIRED_ENV_KEY, "true")
    monkeypatch.setenv("NMP_WORK_PATH", "/job/work")

    _, head_server_config, rollout_helper = runtime.bootstrap_gym_host()

    # Dependencies must be ready before RunHelper starts any Gym servers.
    assert events == [
        "environment-validated",
        "environment-composed",
        "environment-root-prepended",
        "wheels-installed",
        "gym-started",
    ]
    assert head_server_config.host == "127.0.0.1"
    assert head_server_config.port == 5000
    assert rollout_helper == "rollout-helper"


# --------------------------------------------------------------------------------------------
# Long-running rollouts
#
# The sandbox proxy caps how long it waits for a response's first byte, so a batch that takes
# longer than the cap fails in transit no matter how healthy the host is. These cover the host
# answering early and heartbeating, and the deadline that replaces the timeout no hop is left to
# apply. Docker has no proxy in the path, which is why none of this was caught by running it.
# --------------------------------------------------------------------------------------------


class _SlowRolloutHelper:
    """Blocks each rollout until released, so a batch's duration is controlled by the test.

    ``cancelled`` records that the collector was actually cancelled, which is the only way to tell
    an abandoned batch from one that merely finished while nobody was reading.
    """

    def __init__(self, release: threading.Event, *, fail: bool = False) -> None:
        self._release = release
        self._fail = fail
        self.cancelled = threading.Event()

    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            try:
                await asyncio.get_running_loop().run_in_executor(None, self._release.wait, 30)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            if self._fail:
                raise RuntimeError("environment exploded")
            return row, {"response": {"output": []}, "reward": 1.0}

        return [_one(row) for row in examples]


@pytest.fixture
def slow_server():
    """A threaded host whose rollouts block until the test releases them."""
    release = threading.Event()
    original = (
        runtime.Handler.heartbeat_interval_s,
        runtime.Handler.rollout_deadline_s,
        runtime.Handler.max_request_bytes,
        runtime.Handler.max_response_bytes,
    )
    started: list[ThreadingHTTPServer] = []

    def _start(*, fail: bool = False, heartbeat_s: float = 0.05, deadline_s: float = 30.0):
        runtime._READY = True
        runtime._RUN_HELPER = MagicMock()
        runtime._HEAD_SERVER_CONFIG = MagicMock()
        runtime._ROLLOUT_HELPER = _SlowRolloutHelper(release, fail=fail)
        runtime.Handler.max_request_bytes = 4096
        runtime.Handler.max_response_bytes = 4096
        runtime.Handler.heartbeat_interval_s = heartbeat_s
        runtime.Handler.rollout_deadline_s = deadline_s

        server = ThreadingHTTPServer(("127.0.0.1", 0), runtime.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", release

    try:
        yield _start
    finally:
        release.set()
        for server in started:
            server.shutdown()
            server.server_close()
        runtime._READY = False
        runtime._HEAD_SERVER_CONFIG = None
        runtime._ROLLOUT_HELPER = None
        (
            runtime.Handler.heartbeat_interval_s,
            runtime.Handler.rollout_deadline_s,
            runtime.Handler.max_request_bytes,
            runtime.Handler.max_response_bytes,
        ) = original


def _post_rollouts(base_url: str, examples: list[dict], timeout: float = 30.0):
    import urllib.request

    payload = json.dumps({"examples": examples}).encode()
    request = urllib.request.Request(
        f"{base_url}/rollouts/run", data=payload, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=timeout)


def test_rollouts_run_answers_before_the_batch_finishes(slow_server):
    """The status line must leave immediately, and the connection stay visibly alive after it.

    A proxy that has seen no byte for its cap gives up on the request, so a batch slower than the
    cap fails in transit however healthy the host is. Heartbeat whitespace is a valid JSON prefix,
    so it holds the connection open without the reader having to know about it.
    """
    base_url, release = slow_server(heartbeat_s=0.05)

    response = _post_rollouts(base_url, [{"id": 1}])
    # Headers are readable while the rollout is still blocked: proof the host answered first.
    assert response.status == 200
    assert response.headers.get("Content-Length") is None

    time.sleep(0.3)
    release.set()
    raw = response.read().decode()

    assert raw.startswith(" "), "no heartbeat preceded the payload"
    assert json.loads(raw)["results"][0]["reward"] == 1.0


def test_rollouts_run_gives_up_at_the_host_deadline(slow_server):
    """Nothing else can time a wedged batch out, so the host has to.

    Every other hop has been taught to wait, so without this a stuck rollout heartbeats until the
    sandbox's ttl_s and the caller learns nothing about why.
    """
    base_url, _ = slow_server(heartbeat_s=0.05, deadline_s=0.2)

    body = json.loads(_post_rollouts(base_url, [{"id": 1}]).read().decode())

    assert body["error"]["code"] == "deadline_exceeded"
    assert "0.2s" in body["error"]["message"]


def test_rollouts_run_reports_a_failed_batch_in_the_body(slow_server):
    """After the status line is committed, a failure can only be reported in the body.

    It carries a traceback because the host's stdout never reaches the job: without one the caller
    sees the exception's ``str`` and has no idea which of Gym's layers raised it.
    """
    base_url, release = slow_server(fail=True)
    release.set()

    body = json.loads(_post_rollouts(base_url, [{"id": 1}]).read().decode())

    assert body["error"]["code"] == "internal"
    assert "environment exploded" in body["error"]["message"]
    assert "Traceback" in body["error"]["message"]


def test_health_stays_answerable_during_a_rollout(slow_server):
    """A single-threaded server would queue /health behind the batch and read as unhealthy."""
    import urllib.request

    base_url, release = slow_server()
    rollout = threading.Thread(target=lambda: _post_rollouts(base_url, [{"id": 1}]).read())
    rollout.start()
    try:
        time.sleep(0.2)
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as health:
            assert json.loads(health.read().decode()) == {"status": "ready"}
    finally:
        release.set()
        rollout.join(timeout=10)


def test_a_disconnected_caller_cancels_the_batch(slow_server):
    """Nothing will read the results, so the host should stop paying for them.

    Without this the batch runs to completion against a socket that is gone, and with chunking
    several abandoned batches can pile up on the shared loop at once.
    """
    import urllib.request

    base_url, release = slow_server(heartbeat_s=0.05)
    helper = runtime._ROLLOUT_HELPER
    payload = json.dumps({"examples": [{"id": 1}]}).encode()
    request = urllib.request.Request(
        f"{base_url}/rollouts/run", data=payload, headers={"Content-Type": "application/json"}
    )
    response = urllib.request.urlopen(request, timeout=10)

    # Drop the connection while the host is still heartbeating into it.
    response.close()

    assert helper.cancelled.wait(10), "the abandoned batch kept running after the caller left"

    release.set()
    # And the server stays healthy for the next caller rather than wedging on the broken one.
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as health:
        assert json.loads(health.read().decode()) == {"status": "ready"}


def test_rollouts_reuse_one_live_event_loop():
    """Gym's shared HTTP client binds to the loop that created it.

    A loop per request would leave the client pointing at a closed one, so the second batch fails
    inside Gym rather than here.
    """
    first = runtime._ensure_event_loop()
    second = runtime._ensure_event_loop()

    assert first is second
    assert first.is_running()


def test_concurrent_rollout_batches_interleave_on_that_loop():
    """Chunked callers post several batches at once; they must not serialize or deadlock."""
    helper = _FakeRolloutHelper()
    config = MagicMock()

    futures = [runtime.submit_rollouts([{"id": index}], config, helper) for index in range(4)]
    results = [future.result(timeout=10) for future in futures]

    assert [len(batch) for batch in results] == [1, 1, 1, 1]


def _capture_line(**overrides):
    call = {"model_call_id": "c0", "status_code": 200, "started_at": 1.5, "completed_at": 1.75}
    return json.dumps({**call, **overrides})


def test_capture_is_enabled_so_a_rollout_records_per_call_timing(tmp_path):
    # Set, not defaulted: a caller's global config that left observability off would produce traces
    # with no per-call timing, which reads the same as a model that was never called.
    config = {"observability_enabled": False}

    runtime._apply_model_call_capture(config, str(tmp_path))

    assert config["observability_enabled"] is True
    assert config["model_call_capture_dir"] == str(tmp_path / runtime.MODEL_CALL_CAPTURE_SUBDIR)
    assert (tmp_path / runtime.MODEL_CALL_CAPTURE_SUBDIR).is_dir()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"_ng_task_index": 0, "_ng_rollout_index": 1}, "0-1.capture.jsonl"),
        ({"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_attempt_index": 0}, "0-1.capture.jsonl"),
        ({"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_attempt_index": 2}, "0-1-a2.capture.jsonl"),
        ({"_ng_task_index": 0}, None),
        ({"_ng_task_index": True, "_ng_rollout_index": 1}, None),
    ],
)
def test_capture_filename_follows_gyms_rollout_id(result, expected):
    assert runtime._capture_filename(result) == expected


def test_a_capture_is_read_back_onto_its_rollout(tmp_path):
    # Trailing blank line included: Gym appends per call, so a partially-flushed file has one.
    (tmp_path / "0-1.capture.jsonl").write_text(f"{_capture_line()}\n\n", encoding="utf-8")

    calls, spent = runtime._read_capture(str(tmp_path), {"_ng_task_index": 0, "_ng_rollout_index": 1}, budget=10_000)

    assert [call["model_call_id"] for call in calls] == ["c0"]
    assert spent > 0


def test_a_capture_over_budget_is_dropped_whole_rather_than_truncated(tmp_path):
    # Half a capture would project into a trace that looks complete while under-reporting the calls
    # the agent actually made.
    (tmp_path / "0-1.capture.jsonl").write_text(f"{_capture_line()}\n{_capture_line(model_call_id='c1')}\n")

    calls, spent = runtime._read_capture(str(tmp_path), {"_ng_task_index": 0, "_ng_rollout_index": 1}, budget=10)

    assert calls == []
    assert spent == 0


@pytest.mark.parametrize("body", ["", "{not json}\n", "[1, 2]\n"])
def test_an_unreadable_capture_costs_the_timing_not_the_rollout(tmp_path, body):
    (tmp_path / "0-1.capture.jsonl").write_text(body, encoding="utf-8")

    assert runtime._read_capture(str(tmp_path), {"_ng_task_index": 0, "_ng_rollout_index": 1}, budget=10_000) == ([], 0)


def test_a_missing_capture_is_not_an_error(tmp_path):
    assert runtime._read_capture(str(tmp_path), {"_ng_task_index": 9, "_ng_rollout_index": 9}, budget=10_000) == ([], 0)


def test_an_unwritable_work_path_disables_capture_rather_than_failing_the_host(tmp_path):
    # Whether the work path is writable varies by sandbox provider. A host that refuses to start is
    # a far worse outcome than one whose traces lack per-call timing.
    unwritable = tmp_path / "file-not-a-dir"
    unwritable.write_text("", encoding="utf-8")
    config = {}

    runtime._apply_model_call_capture(config, str(unwritable))

    assert config == {}


class _IndexedRolloutHelper:
    """Yields a result per example, carrying the rollout identity a capture is keyed by."""

    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            return row, {"response": {"output": []}, "reward": 0.0, **row}

        return [_one(row) for row in examples]


def _rollout_examples(count):
    return [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(count)]


async def _collect(tmp_path, examples, budget):
    return await runtime._collect_rollout_results(examples, MagicMock(), _IndexedRolloutHelper(), str(tmp_path), budget)


def test_each_rollouts_capture_is_attached_to_its_own_result(tmp_path):
    # The whole point of the hop: without this the record crosses the wire bare and the trace it
    # projects into has no per-call timing.
    for index in range(2):
        (tmp_path / f"{index}-0.capture.jsonl").write_text(f"{_capture_line(model_call_id=f'c{index}')}\n")

    results = asyncio.run(_collect(tmp_path, _rollout_examples(2), 10_000))

    attached = [[call["model_call_id"] for call in result[runtime.MODEL_CALLS_RESULT_KEY]] for result in results]
    assert attached == [["c0"], ["c1"]]


def test_the_budget_is_spent_across_rollouts_not_per_rollout(tmp_path):
    # Captures share one response, so the budget has to deplete. Applied per rollout instead, a
    # large run would blow the response cap and the host would refuse every result.
    for index in range(3):
        (tmp_path / f"{index}-0.capture.jsonl").write_text(f"{_capture_line(model_call_id=f'c{index}')}\n")
    one_capture = (tmp_path / "0-0.capture.jsonl").stat().st_size

    results = asyncio.run(_collect(tmp_path, _rollout_examples(3), one_capture * 2))

    # The first two fit; the third finds nothing left and returns without its capture.
    assert [runtime.MODEL_CALLS_RESULT_KEY in result for result in results] == [True, True, False]


def test_rollouts_still_return_when_no_capture_was_written(tmp_path):
    # The reward and output stand on their own; only the per-call timing is missing.
    results = asyncio.run(_collect(tmp_path, _rollout_examples(2), 10_000))

    assert len(results) == 2
    assert all(runtime.MODEL_CALLS_RESULT_KEY not in result for result in results)


def test_capture_spend_is_measured_in_bytes_not_characters(tmp_path):
    # The budget is spent against a byte cap. `len` on decoded text counts code points, so a CJK
    # capture would report about a third of what it costs on the wire -- the aggregate then
    # overshoots `max_response_bytes`, and an oversized response is refused whole.
    # `ensure_ascii=False` on purpose: escaped to \uXXXX the file would be pure ASCII, byte count
    # would equal character count, and the test would pass against the bug it is written for.
    payload = json.dumps(
        {"model_call_id": "c0", "note": "四十二といえば生命、宇宙、そして万物についての究極の疑問の答え"},
        ensure_ascii=False,
    )
    path = tmp_path / "0-1.capture.jsonl"
    path.write_text(f"{payload}\n", encoding="utf-8")
    assert path.stat().st_size > len(payload) + 1, "test data must be genuinely multibyte"

    _, spent = runtime._read_capture(str(tmp_path), {"_ng_task_index": 0, "_ng_rollout_index": 1}, budget=10_000)

    assert spent == path.stat().st_size


def test_a_capture_that_is_not_valid_utf8_costs_the_timing_not_the_rollout(tmp_path):
    (tmp_path / "0-1.capture.jsonl").write_bytes(b'{"model_call_id": "\xff\xfe"}\n')

    assert runtime._read_capture(str(tmp_path), {"_ng_task_index": 0, "_ng_rollout_index": 1}, budget=10_000) == ([], 0)


# ------------------------------------------------------------------------------------------
# Response framing
# ------------------------------------------------------------------------------------------


class _DelayedRolloutHelper:
    """A helper whose rollouts take long enough for the heartbeat to fire.

    Distinct from `_SlowRolloutHelper` above, which blocks until a test releases it: these
    tests read the wire to the end of the body, so the rollout has to finish on its own.
    """

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def run_examples(self, examples, head_server_config=None):
        async def _one(row):
            await asyncio.sleep(self.delay_s)
            return row, {"response": {"output": []}, "reward": 0.0}

        return [_one(row) for row in examples]


def _body_is_complete(received: bytes) -> bool:
    """Whether the framing in ``received`` marks the body as finished."""
    head, sep, body = received.partition(b"\r\n\r\n")
    if not sep:
        return False
    if b"Transfer-Encoding: chunked" in head:
        return body.endswith(b"0\r\n\r\n")
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            return len(body) >= int(line.split(b":")[1])
    return False


def _raw_rollout_exchange(base_url: str, payload: str, version: str) -> bytes:
    """POST /rollouts/run over a bare socket and return the response bytes as sent.

    urllib de-chunks transparently, which is the framing under test, so these assertions
    have to read the wire.
    """
    parsed = urlsplit(base_url)
    body = payload.encode()
    request = (
        f"POST /rollouts/run {version}\r\n"
        f"Host: {parsed.hostname}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    ).encode() + body

    with socket.create_connection((parsed.hostname, parsed.port), timeout=10) as sock:
        sock.sendall(request)
        sock.settimeout(10)
        received = b""
        while not _body_is_complete(received):
            part = sock.recv(4096)
            if not part:
                break
            received += part
    return received


def _one_example() -> str:
    return json.dumps({"examples": [{"agent_ref": {"name": "a"}, "_rowidx": 0}]})


def test_rollouts_run_frames_the_body_so_a_heartbeat_cannot_end_it(ready_server):
    """The heartbeat and the payload must be separately framed chunks.

    Regression guard for nvbug 6716627. The host used to flush the status line with neither
    a length nor chunking, leaving the body delimited only by connection close. The
    OpenSandbox proxy does not wait for that close: it returned HTTP 200 with the lone " "
    heartbeat -- or nothing at all -- as the finished response, and the caller failed with
    `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. Only the zero-length chunk
    may end this body.
    """
    runtime.Handler.heartbeat_interval_s = 0.02
    runtime._ROLLOUT_HELPER = _DelayedRolloutHelper(0.3)

    received = _raw_rollout_exchange(ready_server, _one_example(), "HTTP/1.1")

    head, _, body = received.partition(b"\r\n\r\n")
    assert b"Transfer-Encoding: chunked" in head
    assert b"Content-Length" not in head
    # The terminator, and nothing after it.
    assert body.endswith(b"0\r\n\r\n")
    # A heartbeat is its own chunk -- length-prefixed, so it cannot read as the end.
    assert body.startswith(b"1\r\n \r\n")

    decoded = ""
    rest = body
    while True:
        size_line, _, rest = rest.partition(b"\r\n")
        size = int(size_line, 16)
        if size == 0:
            break
        decoded += rest[:size].decode()
        rest = rest[size + 2 :]
    assert len(json.loads(decoded)["results"]) == 1


def test_rollouts_run_sends_a_length_to_an_http_10_caller(ready_server):
    """A 1.0 caller cannot parse chunks, so it gets the whole body with a length.

    Still explicitly framed -- that is the point. 1.0's own answer for a body of unknown
    length is the close-delimited one that lost the response, so the host buffers instead and
    pays for it in latency rather than in correctness.
    """
    runtime.Handler.heartbeat_interval_s = 0.02
    runtime._ROLLOUT_HELPER = _DelayedRolloutHelper(0.1)

    received = _raw_rollout_exchange(ready_server, _one_example(), "HTTP/1.0")

    head, _, body = received.partition(b"\r\n\r\n")
    assert b"Transfer-Encoding" not in head
    assert b"Content-Length: " in head
    # Buffered whole, so no heartbeat leaked into a body that is now length-delimited.
    assert not body.startswith(b" ")
    assert len(json.loads(body.decode())["results"]) == 1


def test_connections_are_never_reused(ready_server):
    """1.1 is here for chunked framing only; its connection reuse is declined.

    The sharp hazard: a request whose body the handler never reads (the 413 path declines on
    purpose) leaves those bytes in the socket, and the server parses them as the next request
    line. Pipelined here to prove it cannot happen -- before the connection was closed per
    request, this exact exchange answered `400 Bad request syntax` and swallowed the real
    /health.
    """
    parsed = urlsplit(ready_server)
    body = _one_example().encode()
    pipelined = (
        (
            "POST /rollouts/run HTTP/1.1\r\n"
            f"Host: {parsed.hostname}\r\n"
            "Content-Type: application/json\r\n"
            # Oversize by declaration: the 413 check reads Content-Length, not the body.
            f"Content-Length: {runtime.Handler.max_request_bytes + 1}\r\n"
            "\r\n"
        ).encode()
        + body
        + f"GET /health HTTP/1.1\r\nHost: {parsed.hostname}\r\n\r\n".encode()
    )

    with socket.create_connection((parsed.hostname, parsed.port), timeout=10) as sock:
        sock.settimeout(10)
        sock.sendall(pipelined)
        received = b""
        while True:
            part = sock.recv(4096)
            if not part:
                break
            received += part

    assert received.startswith(b"HTTP/1.1 413")
    # One response, and the leftover body was never parsed as a request of its own.
    assert received.count(b"HTTP/1.1") == 1
    assert b"Bad request" not in received
    # Advertised, so a pooling proxy drops the socket instead of reusing it.
    assert b"Connection: close" in received


def test_the_rollout_response_declines_reuse_too(ready_server):
    """The zero-length chunk says where the body ends; Connection: close says the socket is
    done. Neither substitutes for the other."""
    runtime.Handler.heartbeat_interval_s = 0.02
    runtime._ROLLOUT_HELPER = _DelayedRolloutHelper(0.1)

    received = _raw_rollout_exchange(ready_server, _one_example(), "HTTP/1.1")

    head, _, body = received.partition(b"\r\n\r\n")
    assert b"Transfer-Encoding: chunked" in head
    assert b"Connection: close" in head
    assert body.endswith(b"0\r\n\r\n")


def test_bodiless_responses_carry_a_length(ready_server):
    """HTTP/1.1 keep-alive reuses the socket, so even a 404 must say where it ends."""
    import urllib.error
    import urllib.request

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{ready_server}/nope", timeout=10)
    assert excinfo.value.code == 404
    assert excinfo.value.headers.get("Content-Length") == "0"


def test_an_empty_chunk_is_never_written():
    """A zero-length chunk is the terminator, so writing one for empty data ends the body early.

    Reached when a heartbeat or a result body is empty: the caller would see a well-formed but
    truncated response rather than an error.
    """
    written: list[bytes] = []
    handler = runtime.Handler.__new__(runtime.Handler)
    handler.wfile = cast(Any, SimpleNamespace(write=written.append, flush=lambda: None))

    handler._write_chunk(b"")
    assert written == []

    handler._write_chunk(b" ")
    assert written == [b"1\r\n \r\n"]
