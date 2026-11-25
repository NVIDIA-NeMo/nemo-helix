# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
from sandboxed_gym.host.entrypoint import (
    DEFAULT_GYM_ACTOR_VENV,
    DEFAULT_GYM_WRITABLE_SRC,
    DEFAULT_IMAGE_GIT_ROOT,
    default_gym_host_entrypoint,
    gym_host_script_path,
    packaged_gym_host_script,
)
from sandboxed_gym.host.models import GymHostSpec, GymHostVolumeMount
from sandboxed_gym.runtime import gym_host_runtime as runtime

#: These two tests instantiate the host provider, which drives the OpenSandbox SDK directly. The
#: SDK is an optional extra (`nemo-sandboxed-gym[opensandbox]`) because only a deployment that
#: provisions real sandboxes needs it, so it is absent from a plain workspace checkout. Everything
#: else in this package works without it -- the broker contract and the sandbox types are vendored
#: (see `sandboxed_gym.wire`).
requires_opensandbox = pytest.mark.skipif(
    importlib.util.find_spec("opensandbox") is None,
    reason="needs the OpenSandbox SDK; install the `opensandbox` extra to run",
)


def test_default_gym_host_entrypoint_uses_sandboxed_actor_venv():
    entrypoint = default_gym_host_entrypoint()
    assert entrypoint[0] == "/bin/sh"
    assert entrypoint[1] == gym_host_script_path()
    assert entrypoint[2] == DEFAULT_GYM_ACTOR_VENV
    assert entrypoint[3] == DEFAULT_IMAGE_GIT_ROOT
    assert entrypoint[4] == DEFAULT_GYM_WRITABLE_SRC
    assert entrypoint[5].endswith("gym_host_runtime.py")
    script = packaged_gym_host_script()
    text = script.read_text(encoding="utf-8")
    assert "cp -a" in text
    assert "gym_host_runtime.py" in text


def test_apply_uv_dirs_fills_defaults(tmp_path, monkeypatch):
    cache = tmp_path / "uv-cache"
    venvs = tmp_path / "gym-venvs"
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", str(venvs))
    cfg: dict = {}
    runtime._apply_uv_dirs(cfg)
    assert cfg["uv_cache_dir"] == str(cache)
    assert cfg["uv_venv_dir"] == str(venvs)


def test_apply_uv_dirs_preserves_config(tmp_path, monkeypatch):
    cache = tmp_path / "from-config-cache"
    venvs = tmp_path / "from-config-venvs"
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "ignored"))
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", str(tmp_path / "ignored-venvs"))
    cfg = {"uv_cache_dir": str(cache), "uv_venv_dir": str(venvs)}
    runtime._apply_uv_dirs(cfg)
    assert cfg["uv_cache_dir"] == str(cache)
    assert cfg["uv_venv_dir"] == str(venvs)


@requires_opensandbox
def test_opensandbox_host_provider_defaults_skip_health_check():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})
    assert provider._create_options["skip_health_check"] is True


@requires_opensandbox
def test_opensandbox_host_provider_honors_explicit_skip_health_check():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(
        connection={"domain": "x", "api_key": "k"},
        create={"skip_health_check": False},
    )
    assert provider._create_options["skip_health_check"] is False


@requires_opensandbox
def test_opensandbox_host_provider_uses_configured_protocol_for_bare_endpoints():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"protocol": "http"})

    assert provider._absolute_url("10.244.6.40:8080") == "http://10.244.6.40:8080"


def _gym_host_spec(*, entrypoint: tuple[str, ...] | None = None) -> GymHostSpec:
    return GymHostSpec(
        job_id="job-1",
        runtime_image="nhx-gym-host:dev",
        environment_mount=GymHostVolumeMount(
            pvc_claim="job-storage",
            sub_path="environment",
            mount_path="/job/environment",
            read_only=True,
        ),
        workspace_mount=GymHostVolumeMount(
            pvc_claim="job-storage",
            sub_path="work",
            mount_path="/job/work",
        ),
        entrypoint=entrypoint,
    )


@requires_opensandbox
def test_opensandbox_host_provider_supplies_runtime_entrypoint_when_omitted():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})
    sandbox_spec = provider._to_sandbox_spec(_gym_host_spec())

    assert sandbox_spec.entrypoint == ["python", "-I", "-u", "-m", "sandboxed_gym.runtime.gym_host_runtime"]


@requires_opensandbox
def test_opensandbox_host_provider_preserves_configured_entrypoint():
    from sandboxed_gym.host.opensandbox import OpenSandboxGymHostProvider

    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"})
    sandbox_spec = provider._to_sandbox_spec(_gym_host_spec(entrypoint=("custom-runtime", "--serve")))

    assert sandbox_spec.entrypoint == ["custom-runtime", "--serve"]


@requires_opensandbox
def test_opensandbox_host_provider_installs_agent_bundle():
    """The runtime image starts as usual; the runtime checks the agent once it has dropped privileges."""
    from sandboxed_gym.host.opensandbox import AGENT_BUNDLE_EXTENSION_KEY, OpenSandboxGymHostProvider
    from sandboxed_gym.runtime.agent_bundle import AGENT_EXECUTABLE_ENV_KEY

    bundle = {"image": "nvcr.io/x/codex:1", "installer": "/install", "entrypoint": "bin/codex"}
    provider = OpenSandboxGymHostProvider(connection={"domain": "x", "api_key": "k"}, agent_bundle=bundle)
    sandbox_spec = provider._to_sandbox_spec(_gym_host_spec(entrypoint=("custom-runtime", "--serve")))

    assert json.loads(sandbox_spec.provider_options["extensions"][AGENT_BUNDLE_EXTENSION_KEY]) == bundle
    assert sandbox_spec.entrypoint == ["custom-runtime", "--serve"]
    assert sandbox_spec.env[AGENT_EXECUTABLE_ENV_KEY] == "/installed-agent/bin/codex"


# --------------------------------------------------------------------------------------------
# gym_host.sh — the copy that stages Gym into a writable tree
#
# A caller can configure this explicit bootstrap for a compatible runtime image. The standard
# platform image instead starts the installed runtime module directly.
# --------------------------------------------------------------------------------------------


def _fake_host(tmp_path: Path) -> dict[str, Path]:
    """A tree the script can run against: a Gym source, a venv whose python exits 0, a runtime."""
    gym_tree = tmp_path / "Gym"
    (gym_tree / "nemo_gym").mkdir(parents=True)
    (gym_tree / "nemo_gym" / "__init__.py").write_text("", encoding="utf-8")

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)
    # ray start returns once the daemons are up; the stub records the argv the script passed.
    venv_ray = venv_bin / "ray"
    venv_ray.write_text(
        '#!/bin/sh\nif [ -n "${RAY_LOG:-}" ]; then printf \'%s\\n\' "$*" >> "$RAY_LOG"; fi\nexit 0\n',
        encoding="utf-8",
    )
    venv_ray.chmod(0o755)

    runtime = tmp_path / "runtime.py"
    runtime.write_text("", encoding="utf-8")
    (tmp_path / "root").mkdir()
    return {"tree": gym_tree, "venv": tmp_path / "venv", "runtime": runtime, "root": tmp_path / "root"}


def _run_gym_host(paths: dict[str, Path], gym_rw: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "/bin/sh",
            str(gym_host_script_path(git_root=str(paths["root"]))),
            str(paths["venv"]),
            str(paths["root"]),
            str(gym_rw),
            str(paths["runtime"]),
        ],
        env={**os.environ, "SANDBOXED_GYM_TREE": str(paths["tree"])},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_staged_path_is_published_for_search_root_discovery(tmp_path: Path) -> None:
    paths = _fake_host(tmp_path)
    recorded = tmp_path / "recorded-src"
    python = paths["venv"] / "bin" / "python"
    python.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$SANDBOXED_GYM_SRC_DIR" > "$RECORDED"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    gym_rw = tmp_path / "custom-gym" / "Gym"

    result = subprocess.run(
        [
            "/bin/sh",
            str(gym_host_script_path(git_root=str(paths["root"]))),
            str(paths["venv"]),
            str(paths["root"]),
            str(gym_rw),
            str(paths["runtime"]),
        ],
        env={**os.environ, "SANDBOXED_GYM_TREE": str(paths["tree"]), "RECORDED": str(recorded)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert recorded.read_text(encoding="utf-8") == f"{gym_rw}\n"


def test_the_jobs_variables_reach_ray_and_the_runtime_as_before_without_the_execd_token(tmp_path: Path) -> None:
    """This launcher has no privileged setup, so it unpacks the job's variables before starting anything.

    The script's own parameters and exports still win, a job's PYTHONPATH is still appended after the staged Gym tree, and
    names a shell can't export stay packed for the runtime. execd's token reaches nothing the launcher runs, a program on the job's PATH included.
    """
    from sandboxed_gym.runtime.gym_host_runtime import decode_job_env, encode_job_env

    paths = _fake_host(tmp_path)
    for tool, log in (("ray", "ray-env"), ("python", "python-env")):
        script = paths["venv"] / "bin" / tool
        script.write_text(f'#!/bin/sh\nenv > "{tmp_path / log}"\nexit 0\n', encoding="utf-8")
        script.chmod(0o755)
    (paths["venv"] / "bin" / "python").write_text(
        f'#!/bin/sh\nenv > "{tmp_path / "python-env"}"\nprintf \'%s\\n\' "$1" > "{tmp_path / "runtime-arg"}"\n',
        encoding="utf-8",
    )
    # A job's PATH can put its own programs in front of the ones the launcher stages the Gym tree with.
    job_bin = tmp_path / "job-bin"
    job_bin.mkdir()
    (job_bin / "cp").write_text(f'#!/bin/sh\nenv > "{tmp_path / "cp-env"}"\nexec /bin/cp "$@"\n', encoding="utf-8")
    (job_bin / "cp").chmod(0o755)
    job_env = {
        "PATH": f"{job_bin}:{os.environ['PATH']}",
        # Same names as the launcher's parameters: the job can't redirect what it starts.
        "venv": "/job/venv",
        "root": "/job/root",
        "gym_rw": "/job/gym",
        "runtime": "/job/missing.py",
        "PROMPT": "it's $(not) expanded",
        "PYTHONPATH": "/job/lib",
        "SANDBOXED_GYM_SRC_DIR": "/job/elsewhere",
        "odd.name": "kept packed",
    }
    gym_rw = tmp_path / "gym-src" / "Gym"

    result = subprocess.run(
        [
            "/bin/sh",
            str(gym_host_script_path(git_root=str(paths["root"]))),
            str(paths["venv"]),
            str(paths["root"]),
            str(gym_rw),
            str(paths["runtime"]),
        ],
        env={
            **os.environ,
            "SANDBOXED_GYM_TREE": str(paths["tree"]),
            "NHX_GYM_JOB_ENV": encode_job_env(job_env),
            "EXECD_ACCESS_TOKEN": "sandbox-token",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    ray_env = dict(line.split("=", 1) for line in (tmp_path / "ray-env").read_text().splitlines() if "=" in line)
    runtime_env = dict(line.split("=", 1) for line in (tmp_path / "python-env").read_text().splitlines() if "=" in line)
    cp_env = dict(line.split("=", 1) for line in (tmp_path / "cp-env").read_text().splitlines() if "=" in line)
    for env in (cp_env, ray_env, runtime_env):
        assert "EXECD_ACCESS_TOKEN" not in env
        assert env["PROMPT"] == "it's $(not) expanded"
    assert runtime_env["SANDBOXED_GYM_SRC_DIR"] == str(gym_rw)
    assert runtime_env["PYTHONPATH"] == f"{gym_rw}:{paths['root']}:/job/lib"
    assert decode_job_env(runtime_env["NHX_GYM_JOB_ENV"]) == {"odd.name": "kept packed"}
    assert (tmp_path / "runtime-arg").read_text() == f"{paths['runtime']}\n"


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs procfs")
def test_a_program_on_the_jobs_path_cannot_read_the_token_from_the_launchers_initial_environment(
    tmp_path: Path,
) -> None:
    """Unsetting a variable leaves it in /proc/<pid>/environ, so the launcher starts again without the token."""
    from sandboxed_gym.runtime.gym_host_runtime import encode_job_env

    paths = _fake_host(tmp_path)
    job_bin = tmp_path / "job-bin"
    job_bin.mkdir()
    (job_bin / "cp").write_text(
        f'#!/bin/sh\ncat /proc/$PPID/environ > "{tmp_path / "launcher-environ"}"\nexec /bin/cp "$@"\n',
        encoding="utf-8",
    )
    (job_bin / "cp").chmod(0o755)

    result = subprocess.run(
        [
            "/bin/sh",
            str(gym_host_script_path(git_root=str(paths["root"]))),
            str(paths["venv"]),
            str(paths["root"]),
            str(tmp_path / "gym-src" / "Gym"),
            str(paths["runtime"]),
        ],
        env={
            **os.environ,
            "SANDBOXED_GYM_TREE": str(paths["tree"]),
            "NHX_GYM_JOB_ENV": encode_job_env({"PATH": f"{job_bin}:{os.environ['PATH']}"}),
            "EXECD_ACCESS_TOKEN": "sandbox-token",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    launcher_environ = (tmp_path / "launcher-environ").read_bytes()
    assert launcher_environ
    assert b"sandbox-token" not in launcher_environ


def test_the_agent_bundle_is_checked_before_the_jobs_variables_arrive(tmp_path: Path) -> None:
    """This script has no privileged setup, so it checks the agent first, as the launcher that ran it did.

    The agent runs with its own directory first on PATH and none of the job's variables; the job's PATH then
    replaces PATH, and the runtime never sees the name, so it doesn't check again.
    """
    from sandboxed_gym.runtime.gym_host_runtime import encode_job_env

    paths = _fake_host(tmp_path)
    (paths["venv"] / "bin" / "python").write_text(f'#!/bin/sh\nenv > "{tmp_path / "python-env"}"\n', encoding="utf-8")
    agent_bin = tmp_path / "installed-agent" / "bin"
    agent_bin.mkdir(parents=True)
    agent = agent_bin / "codex"
    agent.write_text(f'#!/bin/sh\nenv > "{tmp_path / "agent-env"}"\necho codex 1.0\n', encoding="utf-8")
    agent.chmod(0o755)

    def run(agent_script: str) -> subprocess.CompletedProcess:
        agent.write_text(agent_script, encoding="utf-8")
        return subprocess.run(
            [
                "/bin/sh",
                str(gym_host_script_path(git_root=str(paths["root"]))),
                str(paths["venv"]),
                str(paths["root"]),
                str(tmp_path / "gym-src" / "Gym"),
                str(paths["runtime"]),
            ],
            env={
                **os.environ,
                "SANDBOXED_GYM_TREE": str(paths["tree"]),
                "NHX_GYM_AGENT_EXECUTABLE": str(agent),
                "NHX_GYM_JOB_ENV": encode_job_env({"PATH": "/job/bin:/usr/bin:/bin", "JOB_ONLY": "1"}),
            },
            capture_output=True,
            text=True,
            timeout=60,
        )

    result = run(f'#!/bin/sh\nenv > "{tmp_path / "agent-env"}"\necho codex 1.0\n')

    assert result.returncode == 0, result.stderr
    assert "codex 1.0" in result.stderr
    agent_env = dict(line.split("=", 1) for line in (tmp_path / "agent-env").read_text().splitlines() if "=" in line)
    runtime_env = dict(line.split("=", 1) for line in (tmp_path / "python-env").read_text().splitlines() if "=" in line)
    assert agent_env["PATH"].split(":")[0] == str(agent_bin)
    assert "JOB_ONLY" not in agent_env
    assert runtime_env["PATH"] == "/job/bin:/usr/bin:/bin"
    assert "NHX_GYM_AGENT_EXECUTABLE" not in runtime_env

    (tmp_path / "python-env").unlink()
    broken = run("#!/bin/sh\nexit 3\n")
    assert broken.returncode != 0
    assert not (tmp_path / "python-env").exists()


def test_ray_head_starts_in_the_background_and_is_published(tmp_path: Path) -> None:
    """The head must return so the host can start, and Gym must be told to join it.

    `--block` never returns, so the runtime would not start. The head is also the worker
    (`ray start --head` without `--num-cpus=0`). Gym reads `ray_head_node_address` from
    `NHX_GYM_GLOBAL_CONFIG`, which the orchestrator strips before the sandbox is created.
    """
    paths = _fake_host(tmp_path)
    ray_log = tmp_path / "ray-args"
    config_log = tmp_path / "config"
    python = paths["venv"] / "bin" / "python"
    python.write_text(
        '#!/bin/sh\nprintf \'%s\' "$NHX_GYM_GLOBAL_CONFIG" > "$CONFIG_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    python.chmod(0o755)

    result = subprocess.run(
        [
            "/bin/sh",
            str(gym_host_script_path(git_root=str(paths["root"]))),
            str(paths["venv"]),
            str(paths["root"]),
            str(tmp_path / "gym-src" / "Gym"),
            str(paths["runtime"]),
        ],
        env={
            **os.environ,
            "SANDBOXED_GYM_TREE": str(paths["tree"]),
            "RAY_LOG": str(ray_log),
            "CONFIG_LOG": str(config_log),
            "NHX_GYM_GLOBAL_CONFIG": '{"policy_model_name":"qwen"}',
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    args = ray_log.read_text(encoding="utf-8")
    assert args.startswith("start --head ")
    assert "--block" not in args
    assert "--num-cpus=0" not in args
    assert "--node-ip-address=127.0.0.1" in args
    assert "--port=6379" in args
    published = json.loads(config_log.read_text(encoding="utf-8"))
    assert published["policy_model_name"] == "qwen"
    assert published["ray_head_node_address"] == "127.0.0.1:6379"


def test_the_gym_tree_is_staged_where_pythonpath_expects_it(tmp_path: Path) -> None:
    paths = _fake_host(tmp_path)
    gym_rw = tmp_path / "gym-src" / "Gym"

    result = _run_gym_host(paths, gym_rw)

    assert result.returncode == 0, result.stderr
    assert (gym_rw / "nemo_gym").is_dir()


def test_a_restart_does_not_nest_the_tree(tmp_path: Path) -> None:
    """`cp -a src dst` copies *into* dst when dst exists.

    A leftover or half-written destination would become `$gym_rw/Gym/nemo_gym`, which leaves the
    guard still true — so each restart adds another layer and `PYTHONPATH` never resolves
    `nemo_gym`. Reproduced by seeding a destination that exists but is incomplete.
    """
    paths = _fake_host(tmp_path)
    gym_rw = tmp_path / "gym-src" / "Gym"
    gym_rw.mkdir(parents=True)
    (gym_rw / "leftover.txt").write_text("from an interrupted run", encoding="utf-8")

    for _ in range(2):
        assert _run_gym_host(paths, gym_rw).returncode == 0

    assert (gym_rw / "nemo_gym").is_dir()
    assert not (gym_rw / "Gym").exists(), "the tree was nested instead of replaced"
    assert not (gym_rw / "leftover.txt").exists(), "a half-written destination must be repaired"
