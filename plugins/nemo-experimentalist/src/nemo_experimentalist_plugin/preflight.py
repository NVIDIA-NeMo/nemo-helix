# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight checks: four natural-signature suites, two presentations (auto-run + doctor).

Read-only. Certain-failure conditions are ``severity="required"`` (the main
flow hard-errors on them); environment probes that can false-negative are
``severity="advisory"`` (warn and continue). Doctor prints everything.
"""

import importlib.util
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
from nemo_experimentalist_plugin.experimentalist.components.evaluator.dataset_layout import (
    TASK_CONFIG_FILENAME,
    find_task_dirs,
    is_task_dir,
)
from nemo_experimentalist_plugin.experimentalist.components.evaluator.entrypoint import (
    DEFAULT_AGENT_IMPORT_PATH,
    find_entrypoint_module,
    split_import_path,
)
from nemo_experimentalist_plugin.experimentalist.components.repository import (
    _redact_url,
    looks_like_git,
    pr_cli_for_repo_url,
    split_agent_source_uri,
    split_git_ref,
)
from nemo_experimentalist_plugin.profile import AgentProfile
from nemo_experimentalist_plugin.resolve import (
    ResolveError,
    classify_dataset_value,
    parse_local_insights,
    profile_storage_flags,
    select_local_insight,
)
from nemo_helix_plugin.nooa_model_client import configured_model_refs
from nemo_insights_plugin.contracts.checks import CheckResult, make_check_result
from nemo_insights_plugin.contracts.profile import resolve_profile_path
from pydantic import BaseModel

_SKELETON_HINT = (
    "Create optimizer.yaml next to your agent:\n"
    "  agent: <name>\n  task_template: ./path/to/task_template\n"
    "  datasets:\n    train: ./path\n    validation: ./path"
)


def _default_run_cmd(argv: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
    except FileNotFoundError as exc:
        return 127, str(exc)
    except subprocess.TimeoutExpired:
        return 124, ""
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _default_http_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=5, follow_redirects=True).status_code < 500
    except Exception:
        # Boolean reachability probe: any failure (including a malformed URL,
        # which raises httpx.InvalidURL, not HTTPError) means "not reachable".
        return False


class Probes(BaseModel):
    """Injectable system probes; defaults hit the real system."""

    model_config = {"arbitrary_types_allowed": True}

    run_cmd: Callable[[list[str]], tuple[int, str]] = _default_run_cmd
    http_ok: Callable[[str], bool] = _default_http_ok


def check_profile(profile: AgentProfile | None, profile_error: str | None) -> list[CheckResult]:
    """Profile presence/parse check. Doctor-only on purpose: the auto path
    announces a present profile via the loader, and a missing one is fine
    when flags fully specify the run."""
    if profile_error is not None:
        return [
            CheckResult(
                name="profile-parse",
                group="profile",
                status="fail",
                severity="required",
                message=profile_error,
                hint=_SKELETON_HINT,
            )
        ]
    if profile is None:
        return [
            CheckResult(
                name="profile-found",
                group="profile",
                status="fail",
                severity="required",
                message="no optimizer.yaml found (searched cwd and parents)",
                hint=_SKELETON_HINT,
            )
        ]
    return [
        CheckResult(
            name="profile-found",
            group="profile",
            status="pass",
            severity="required",
            message=f"profile for agent {profile.agent!r} at {profile.profile_dir}",
        )
    ]


def check_environment(
    *,
    profile: AgentProfile | None,
    insight: str | None,
    insight_id: str | None = None,
    base_url: str,
    enforce_insight_agent: bool = True,
    probes: Probes | None = None,
) -> list[CheckResult]:
    """Environment checks for Experiment and Doctor.

    Checks configured agent models, platform reachability, an optional insight
    file, Docker, and Harbor.

    ``enforce_insight_agent=False`` skips the local-insight agent match (an
    explicit ``--agent`` overrides the insight's agent).
    """
    p = probes or Probes()
    results = check_models()
    ok = p.http_ok(f"{base_url.rstrip('/')}/health/ready")
    display_base_url = _redact_url(base_url)
    results.append(
        make_check_result(
            "platform-reachable",
            "platform",
            ok,
            "advisory",
            f"{display_base_url} reachable",
            f"{display_base_url} unreachable",
            hint="is the platform running? check --base-url/NHX_BASE_URL",
        )
    )
    profile_agent = profile.agent if (profile is not None and enforce_insight_agent) else None
    results += _check_insight_file(insight, insight_id, profile_agent)
    code, _ = p.run_cmd(["docker", "info"])
    results.append(
        make_check_result(
            "docker",
            "runtime",
            code == 0,
            "required",
            "docker daemon running",
            "docker daemon not running",
            hint="start Docker; Harbor evaluation requires it",
        )
    )
    has_harbor = importlib.util.find_spec("harbor") is not None
    results.append(
        make_check_result(
            "harbor-import",
            "runtime",
            has_harbor,
            "required",
            "harbor importable",
            "harbor not importable",
            hint="uv sync",
        )
    )
    return results


def check_models() -> list[CheckResult]:
    """Check that the active Platform context has a complete model pair."""
    try:
        refs = configured_model_refs()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return [
            CheckResult(
                name="agent-models",
                group="models",
                status="fail",
                severity="required",
                message=str(exc),
                hint="run `nemo setup` and select default and fast agent models",
            )
        ]
    return [
        CheckResult(
            name="agent-models",
            group="models",
            status="pass",
            severity="required",
            message=f"default={refs.default}; fast={refs.fast}",
        )
    ]


def check_artifacts(
    profile: AgentProfile | None,
    *,
    task_template: str | None = None,
    agent_source: str | None = None,
    storage: dict | None = None,
    evaluator: dict | None = None,
    require_template: bool = True,
    probes: Probes | None = None,
) -> list[CheckResult]:
    """Task-template + agent-source (+ pr-cli-auth via *storage*) checks.

    The optional effective-value kwargs replace the profile-derived values when
    provided: the auto path passes its RESOLVED inputs so checks validate what
    the run will actually use (flags/--config may have overridden the profile),
    while doctor passes nothing because there the profile values are the
    effective ones. A profileless auto path supplies all effective values
    directly. ``require_template=False`` skips the task-template checks
    (insight-less runs never read the template). No dataset checks here: the
    experiment flow's resolved dataset URIs are proven-existing by construction,
    so dataset validation lives only in check_datasets (doctor).
    """
    p = probes or Probes()
    results: list[CheckResult] = []
    base_dir = profile.profile_dir if profile is not None else Path.cwd()
    effective_template = task_template or (profile.task_template if profile is not None else None)
    if require_template and effective_template is not None:
        results += _check_task_template(resolve_profile_path(effective_template, base_dir))
    results += _check_agent_source(profile, p, agent_source=agent_source, storage=storage, evaluator=evaluator)
    return results


def check_datasets(profile: AgentProfile, *, require_tasks: bool = True) -> list[CheckResult]:
    """Classify and validate the profile's train/validation dataset refs
    (path-exists / registry-ref pass). Doctor-only: the experiment flow
    resolves its datasets, which proves they exist.

    Existing is not enough to evaluate: Harbor reads the task directories a
    dataset holds, so a directory holding none fails the run. ``require_tasks``
    is off for an insight run, where Eval Author generates the tasks and both
    splits legitimately start empty.
    """
    results: list[CheckResult] = []
    for label, value in (("train", profile.datasets.train), ("validation", profile.datasets.validation)):
        try:
            kind = classify_dataset_value(value, profile.profile_dir)
        except ResolveError as exc:
            results.append(
                CheckResult(
                    name=f"dataset-{label}", group="artifacts", status="fail", severity="required", message=str(exc)
                )
            )
            continue
        if kind == "path":
            results.append(_check_dataset_path(label, resolve_profile_path(value, profile.profile_dir), require_tasks))
        else:
            results.append(
                CheckResult(
                    name=f"dataset-{label}",
                    group="artifacts",
                    status="pass",
                    severity="required",
                    message=f"{label}: registry ref {value!r}, resolved at run time",
                )
            )
    return results


def _check_dataset_path(label: str, path: Path, require_tasks: bool) -> CheckResult:
    """Check one resolved local dataset directory for the tasks a run evaluates."""
    if not path.is_dir():
        return CheckResult(
            name=f"dataset-{label}",
            group="artifacts",
            status="fail",
            severity="required",
            message=(
                f"{label} dataset path is not a directory: {path}"
                if path.exists()
                else f"{label} dataset path missing: {path}"
            ),
        )
    has_tasks = not require_tasks or bool(find_task_dirs(path))
    return make_check_result(
        f"dataset-{label}",
        "artifacts",
        has_tasks,
        "required",
        f"{label} dataset at {path}",
        f"{label} dataset holds no task directories: {path}",
        hint=(
            f"{path} is itself a task directory; point {label} at the directory holding it"
            if is_task_dir(path)
            else f"each task is a subdirectory with a {TASK_CONFIG_FILENAME}"
        ),
    )


def _check_task_template(tt: Path) -> list[CheckResult]:
    results = [
        make_check_result(
            "task-template-dir",
            "artifacts",
            tt.is_dir(),
            "required",
            f"task_template at {tt}",
            f"task_template dir missing: {tt}",
        )
    ]
    toml_path = tt / "task.toml"
    if toml_path.is_file():
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            task = data.get("task")
            name = task.get("name") if isinstance(task, Mapping) else None
            ok = isinstance(name, str) and "/" in name
            results.append(
                make_check_result(
                    "task-toml",
                    "artifacts",
                    ok,
                    "required",
                    "task.toml parses; name keeps org/name format",
                    f"task.toml [task].name {name!r} must keep org/name format",
                )
            )
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            results.append(
                CheckResult(
                    name="task-toml",
                    group="artifacts",
                    status="fail",
                    severity="required",
                    message=f"task.toml unreadable or does not parse: {exc}",
                    hint="save task.toml as readable UTF-8 TOML, then retry",
                )
            )
    elif tt.is_dir():
        results.append(
            CheckResult(
                name="task-toml", group="artifacts", status="fail", severity="required", message=f"missing {toml_path}"
            )
        )
    if tt.is_dir():
        results.append(
            make_check_result(
                "instruction-md",
                "artifacts",
                (tt / "instruction.md").is_file(),
                "advisory",
                "instruction.md present",
                "instruction.md missing (Eval Author fills it per trace)",
            )
        )
    return results


def _check_agent_source(
    profile: AgentProfile | None,
    p: Probes,
    *,
    agent_source: str | None = None,
    storage: dict | None = None,
    evaluator: dict | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    source = agent_source or (profile.agent_source if profile is not None else None)
    repo: str | None = None
    if source is not None and looks_like_git(source):
        try:
            core, _ = split_agent_source_uri(source)  # drop the #path fragment
        except ValueError as exc:
            return [
                CheckResult(
                    name="agent-source-path",
                    group="agent-source",
                    status="fail",
                    severity="required",
                    message=str(exc),
                    hint="use a normalized relative POSIX path without traversal, pathspecs, or .git components",
                )
            ]
        repo, _ = split_git_ref(core)  # drop the @ref pin (splits on ".git@", so ssh://git@host survives)
    elif source is not None:
        base_dir = profile.profile_dir if profile is not None else Path.cwd()
        path = resolve_profile_path(source, base_dir)
        results.append(
            make_check_result(
                "agent-source-dir",
                "agent-source",
                path.is_dir(),
                "required",
                f"agent source at {path}",
                f"agent source dir missing: {path}",
            )
        )
        if path.is_dir():
            results.append(_check_agent_entrypoint(path, evaluator))
    # Effective storage flags from the auto path (resolved config, which may come
    # from --config); doctor falls back to reading the profile's inline dict or
    # path-form experiment_config without importing the loop chain.
    if storage is not None:
        flags = storage
    elif profile is not None:
        flags = profile_storage_flags(profile)
    else:
        flags = {}
    persistence_requested = bool(flags.get("publish_winner") or flags.get("archive_candidates"))
    git_available = True
    if repo is not None or persistence_requested:
        git_code, _ = p.run_cmd(["git", "--version"])
        git_available = git_code == 0
        results.append(
            make_check_result(
                "git-installed",
                "agent-source",
                git_available,
                "required",
                "git is available",
                "'git' is not available but the effective agent source or storage behavior requires it",
                hint="install git, or disable candidate persistence and use a local agent source",
            )
        )
    if repo is not None and git_available:
        code, _ = p.run_cmd(["git", "ls-remote", repo])
        safe_repo = _redact_url(repo)
        failure_message = (
            f"git ls-remote timed out for {safe_repo}"
            if code == 124
            else f"git ls-remote failed for {safe_repo} (exit status {code})"
        )
        results.append(
            make_check_result(
                "agent-source-git",
                "agent-source",
                code == 0,
                "advisory",
                f"git source reachable: {safe_repo}",
                failure_message,
                hint="could not reach the git remote; the run will fail at clone time if this persists",
            )
        )
    if persistence_requested and repo is None:
        results.append(
            CheckResult(
                name="remote-persistence",
                group="agent-source",
                status="warn",
                severity="advisory",
                message="remote candidate persistence requires agent_source to be a git URL",
                hint="set agent_source to an HTTPS, SSH, or SCP-style git URL, or disable remote persistence",
            )
        )
    elif flags.get("publish_winner") and repo is not None:
        target = pr_cli_for_repo_url(repo)
        if target is None:
            results.append(
                CheckResult(
                    name="pr-cli-auth",
                    group="agent-source",
                    status="warn",
                    severity="advisory",
                    message="automatic PR/MR creation is unsupported for this repository host",
                    hint="use a GitHub or GitLab source, or disable publish_winner and use archive_candidates",
                )
            )
        else:
            cli, hostname = target
            authenticated = p.run_cmd([cli, "auth", "status", "--hostname", hostname])[0] == 0
            results.append(
                make_check_result(
                    "pr-cli-auth",
                    "agent-source",
                    authenticated,
                    "advisory",
                    f"{cli} authenticated for {hostname}",
                    f"{cli} is not authenticated for {hostname} PR/MR publishing",
                    hint=f"run `{cli} auth login --hostname {hostname}`, or publish_winner cannot open the winner PR/MR",
                )
            )
    return results


def _check_agent_entrypoint(agent_dir: Path, evaluator: dict | None) -> CheckResult:
    """Resolve the evaluator's entrypoint inside a local agent directory.

    The evaluator imports it from that directory alone, so an entrypoint it
    cannot find there fails every trial of every candidate.
    """
    configured = (evaluator or {}).get("import_path")
    import_path = DEFAULT_AGENT_IMPORT_PATH if configured is None else str(configured)
    try:
        module_name, attribute = split_import_path(import_path)
    except ValueError as exc:
        return CheckResult(
            name="agent-entrypoint",
            group="agent-source",
            status="fail",
            severity="required",
            message=f"evaluator import_path {import_path!r} is unusable: {exc}",
            hint="set evaluator.import_path in the experiment config to <module>:<attribute>",
        )
    found = find_entrypoint_module(agent_dir, module_name)
    expected = Path(*module_name.split("."))
    return make_check_result(
        "agent-entrypoint",
        "agent-source",
        found is not None,
        "required",
        f"evaluator entrypoint {import_path} at {found}",
        f"evaluator cannot import {module_name!r} from the agent source: {agent_dir}",
        hint=(
            f"add {expected}.py defining {attribute} to the agent directory, "
            "or point evaluator.import_path in the experiment config at the module you have"
        ),
    )


def _check_insight_file(
    insight: str | None,
    selector: str | None,
    profile_agent: str | None,
) -> list[CheckResult]:
    if insight is None:
        if selector is not None:
            return [
                CheckResult(
                    name="insight-file",
                    group="platform",
                    status="fail",
                    severity="required",
                    message="--insight-id requires a resolved local multi-insight file",
                )
            ]
        return []
    path = Path(insight)
    if not path.is_file():
        if selector is not None:
            return [
                CheckResult(
                    name="insight-file",
                    group="platform",
                    status="fail",
                    severity="required",
                    message=f"--insight-id requires a local multi-insight file; {insight!r} is a platform id or non-file ref",
                )
            ]
        return [
            CheckResult(
                name="insight-ref",
                group="platform",
                status="pass",
                severity="required",
                message=f"insight {insight!r}: platform id, verified at run time",
            )
        ]
    try:
        items = parse_local_insights(insight)
        if items is None:
            return []
        selected = select_local_insight(items, selector, insight)
    except ResolveError as exc:
        return [
            CheckResult(
                name="insight-file",
                group="platform",
                status="fail",
                severity="required",
                message=str(exc),
            )
        ]
    results = [
        CheckResult(
            name="insight-file",
            group="platform",
            status="pass",
            severity="required",
            message=f"insight file parses: {path} ({len(items)} insight{'s' if len(items) != 1 else ''})",
        )
    ]
    if profile_agent:
        selected_agent = selected.get("agent")
        if isinstance(selected_agent, str) and selected_agent:
            results.append(
                make_check_result(
                    "insight-agent",
                    "platform",
                    selected_agent == profile_agent,
                    "required",
                    f"insight agent matches profile: {profile_agent!r}",
                    f"insight is about agent {selected_agent!r} but the profile is for {profile_agent!r}",
                    hint="pass --agent to override, or use the matching profile (--profile)",
                )
            )
    return results
