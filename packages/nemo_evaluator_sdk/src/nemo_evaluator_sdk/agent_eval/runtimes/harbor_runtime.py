# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor-backed :class:`AgentTaskRunner` for the agent-eval pipeline.

Harbor already runs trials in (Docker) environments, retries them, and writes a
documented results tree: one ``<task>__<hash>/result.json`` per trial under the
job directory. This runtime adapts that tree into SDK :class:`AgentEvalTrial`
objects so an :class:`AgentEvaluator` can score and report Harbor runs through the
same seam as any other runtime.

Two ways to drive it:

* **Native** — pass a :class:`HarborRuntimeConfig` and a dataset directory; the
  runtime builds Harbor's ``JobConfig`` and runs it itself. The one-call
  :func:`run_harbor_eval` loads the tasks, runs, and scores, so caller code is a
  couple of lines. ``harbor`` is imported lazily inside ``run_tasks`` (it is an
  optional extra), so importing this module never requires Harbor. Custom
  ``import_path`` agents are supported too: set ``agent_import_path`` and, for a
  loose ``harbor_wrapper.py``, also ``agent_dir`` — the runtime then injects that
  directory into ``sys.modules`` for the duration of the run and tears it down
  after (see :func:`scoped_harbor_agent_import`). When ``agent_dir`` is omitted
  (the module is already importable) the path is handed to Harbor's importer
  unchanged, so nothing is imposed on how the agent is packaged.
* **Injected / offline** — pass a ``job_dir`` (and optionally a ``run_job``
  callback) to adapt an already-completed job dir or to run a caller-built job.

Trial adaptation validates Harbor's on-disk ``result.json`` files with Harbor's
own model. The import remains lazy, but invoking native execution or offline
adaptation requires the optional Harbor extra and Python >=3.12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric

import contextlib
import hashlib
import importlib.machinery
import json
import logging
import os
import re
import shutil
import sys
import threading
import tomllib
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.reward_keys import ParsedHarborRewards, validate_reward_key
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_trial_adapter import (
    _HARBOR_EXTRA_REQUIRED_MESSAGE,
    _iter_harbor_trial_results,
    _trial_from_harbor_result,
)
from nemo_evaluator_sdk.agent_eval.runtimes.provenance import redact_credentials
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask, AgentEvalTaskset
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, RunnerInfo
from nemo_evaluator_sdk.enums import MetricType
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

logger = logging.getLogger(__name__)

# Default reward key inside Harbor's ``verifier_result.rewards`` mapping.
DEFAULT_REWARD_KEY = "reward"
# Filename that marks a directory as a Harbor task, and the template dir to skip.
_TASK_CONFIG_FILENAME = "task.toml"
_TASK_TEMPLATE_DIRNAME = "task_template"
# Synthetic sys.modules root a custom ``import_path`` agent package is injected under.
_AGENT_IMPORT_ROOT = "_nemo_evaluator_harbor_agents"
# Guards the sys.modules mutation while injecting/removing scoped agent packages.
_IMPORT_LOCK = threading.Lock()
# Open scopes per content-addressed agent package. Identical agent contents share a
# package name, so teardown must wait for the last scope rather than the first.
_AGENT_PACKAGE_REFCOUNTS: dict[str, int] = {}
# Characters of the agent-content digest used to disambiguate the package name. Long
# enough that distinct agents don't collide; short enough to keep import paths (and
# Harbor's persisted JobConfig) readable.
_IMPORT_DIGEST_CHARS = 12
# Records which inputs produced a job dir, so a rerun can tell a reusable cache from
# a stale one. A file, not a directory: Harbor rmtree's stray directories in a job dir.
CACHE_STAMP_FILENAME = ".nemo-eval-harbor-cache.json"
# Public so downstreams can assert the SDK is new enough to own cache staleness.
CACHE_STAMP_VERSION = 2
# Excluded from the cache fingerprint — see :func:`_cache_stamp` for why each one.
_CACHE_IRRELEVANT_OPTIONS = frozenset(
    {"jobs_dir", "job_name", "force_rerun", "quiet", "n_concurrent_trials", "agent_dir", "reward_key"}
)
# Where Harbor persists the JobConfig it will compare a resume against.
_HARBOR_JOB_CONFIG_FILENAME = "config.json"
# Fields Harbor's own JobConfig equality ignores, so they can never be why it refused
# to resume. Pinned against Harbor upstream by the drift-guard test.
_HARBOR_EQ_IGNORED_FIELDS = frozenset({"job_name", "debug"})
# How Harbor says "this job dir cannot be resumed": one from the JobConfig comparison
# in `Job.create`, one from the lock.json check early in `Job.run`. Matching on the
# message is deliberate coupling, and it fails in the safe direction — an
# unrecognized FileExistsError propagates untouched rather than costing a job dir, so
# a Harbor reword degrades to a loud crash, never to a silent deletion.
_HARBOR_RESUME_REFUSALS = ("resumed with a different config", "does not match the resolved job lock")
# Cap on each value rendered into the "what differed" log line: enough for a scalar
# like `n_concurrent_trials`, bounded for a whole nested `agents` list.
_DRIFT_VALUE_CHARS = 80
# Markdown instruction files may carry repository license comments. Those are file metadata, not
# agent-facing task instructions.
_SPDX_HTML_COMMENT_RE = re.compile(r"<!--\s*SPDX-(?:FileCopyrightText|License-Identifier):[^>]*-->\s*")
# Derived/VCS noise skipped when digesting a directory. Deliberately NOT skipped:
# `node_modules` and other vendored dependency trees, which ship with the agent and
# change what it does. `.venv`/`.uv` stay skipped because they are environment, not
# deliverable — the Harbor wrapper does not upload them into the task container.
_DIGEST_SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", ".uv", ".mypy_cache", ".pytest_cache"})
_DIGEST_CHUNK_BYTES = 1 << 20
RunJob = Callable[[], Awaitable[None]]


class HarborRuntimeConfig(BaseModel):
    """Declarative config for running a Harbor job natively through the SDK.

    Holds only plain/pydantic fields so importing this module never needs Harbor;
    the fields are mapped onto Harbor's ``JobConfig`` lazily at run time.
    """

    model_config = ConfigDict(extra="forbid")

    jobs_dir: Path = Field(description="Parent directory Harbor writes the ``<job_name>/`` results tree into.")
    job_name: str | None = Field(default=None, description="Harbor job name; a timestamp is generated when omitted.")
    agent_name: str | None = Field(
        default="oracle",
        description="Built-in Harbor agent to run (e.g. 'oracle'). Ignored when ``agent_import_path`` is set.",
    )
    agent_import_path: str | None = Field(
        default=None,
        description="Custom Harbor agent import path (e.g. 'harbor_wrapper:WrappedAgent'); overrides ``agent_name``.",
    )
    agent_dir: Path | None = Field(
        default=None,
        description=(
            "Directory holding the module named by ``agent_import_path``. Set it for a loose "
            "wrapper file (the SDK makes it importable); leave it unset when the module is "
            "already importable (installed package), and Harbor imports it directly."
        ),
    )
    agent_model_name: str | None = Field(default=None, description="Optional model slug passed to the Harbor agent.")
    agent_kwargs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments forwarded to the Harbor agent's constructor, the equivalent of Harbor's "
            "``--ak key=value``. Not for secrets: Harbor persists them unredacted in the job dir's "
            "``config.json``; inject secrets through the environment instead."
        ),
    )
    n_attempts: int = Field(default=1, ge=1, description="Number of attempts Harbor runs per task.")
    n_concurrent_trials: int = Field(default=4, ge=1, description="Maximum concurrent Harbor trials.")
    quiet: bool = Field(default=True, description="Suppress Harbor's trial progress displays.")
    force_rerun: bool = Field(default=False, description="Delete an existing job dir before running.")
    artifacts: list[str] = Field(default_factory=list, description="Harbor artifact sources to collect per trial.")
    trace_dir: str | None = Field(
        default=None,
        description="Container path of agent traces to collect as the 'traces' artifact (e.g. '/app/traces').",
    )
    max_retries: int = Field(default=0, ge=0, description="Harbor per-trial retry attempts on transient failures.")
    timeout_multiplier: float | None = Field(default=None, description="Global Harbor timeout multiplier.")
    agent_timeout_multiplier: float | None = Field(default=None, description="Agent-phase timeout multiplier.")
    verifier_timeout_multiplier: float | None = Field(default=None, description="Verifier-phase timeout multiplier.")
    agent_setup_timeout_multiplier: float | None = Field(default=None, description="Agent-setup timeout multiplier.")
    environment_build_timeout_multiplier: float | None = Field(
        default=None, description="Environment-build timeout multiplier."
    )
    reward_key: str = Field(default=DEFAULT_REWARD_KEY, description="Key read from Harbor's rewards mapping.")

    @model_validator(mode="after")
    def _agent_dir_needs_import_path(self) -> HarborRuntimeConfig:
        if self.agent_dir is not None and self.agent_import_path is None:
            raise ValueError("agent_dir only applies to a custom agent_import_path")
        return self


def _effective_harbor_agent(config: HarborRuntimeConfig | None) -> str | None:
    """The agent a run will actually use, mirroring ``run_job``'s resolution order.

    ``agent_import_path`` wins when set; otherwise the built-in ``agent_name``, which itself falls back
    to Harbor's ``oracle`` default. Recording the resolved value keeps two runs with different custom
    agents distinguishable in provenance.
    """
    if config is None:
        return None
    return config.agent_import_path or config.agent_name or "oracle"


class HarborAgentTaskRunner:
    """An :class:`AgentTaskRunner` that runs a Harbor job, then adapts its results.

    Two construction modes:

    * **Native** — pass ``config`` (a :class:`HarborRuntimeConfig`); the runtime
      builds and runs Harbor's ``JobConfig`` itself (Harbor is imported lazily).
      The dataset directory is taken from the tasks handed to :meth:`run_tasks`
      (each carries ``metadata['harbor_dataset_path']`` from
      :func:`discover_harbor_tasks`), or from an explicit ``dataset_path``
      override, so it isn't repeated. ``task_names`` optionally restricts the run
      to a subset of tasks, and the ``config``'s ``job_dir`` doubles as a cache:
      an existing run whose Harbor-valid results cover every requested task (with
      ``n_attempts`` attempts each) is re-adapted instead of
      re-run (unless ``force_rerun`` is set). Caching only takes effect when a
      stable ``job_name`` is set on the config — the default timestamped
      ``job_name`` writes a fresh dir per run and never hits the cache.
    * **Injected / offline** — pass ``job_dir`` (and optionally a ``run_job``
      callback); ``run_job`` is awaited before the job dir is read, and
      ``run_job=None`` simply adapts an already-completed job dir.

    ``job_dir`` is the directory Harbor writes its per-trial
    ``<task>__<hash>/result.json`` files into.
    """

    def __init__(
        self,
        *,
        config: HarborRuntimeConfig | None = None,
        dataset_path: str | Path | None = None,
        task_names: Sequence[str] | None = None,
        job_dir: str | Path | None = None,
        run_job: RunJob | None = None,
        reward_key: str = DEFAULT_REWARD_KEY,
    ) -> None:
        if config is None and job_dir is None:
            raise ValueError("provide either a HarborRuntimeConfig or an explicit job_dir")
        self._config = config
        self._dataset_path = Path(dataset_path) if dataset_path is not None else None
        self._task_names = task_names
        self._job_dir = Path(job_dir) if job_dir is not None else None
        self._run_job = run_job
        self._reward_key = config.reward_key if config is not None else reward_key
        validate_reward_key(self._reward_key)

    def runner_info(self) -> RunnerInfo:
        """Identify this runner and the Harbor settings that shape its results.

        Records the *effective* agent, mirroring how ``run_job`` resolves it: ``agent_import_path``
        overrides ``agent_name`` (which itself defaults to ``oracle``). Reporting the configured
        ``agent_name`` alone would give two runs using different custom agents identical provenance.
        """
        config = self._config
        return RunnerInfo(
            name="harbor",
            kind="runner",
            config={
                "agent_name": config.agent_name if config is not None else None,
                "agent_import_path": config.agent_import_path if config is not None else None,
                "agent_model_name": config.agent_model_name if config is not None else None,
                "agent_kwargs": redact_credentials(config.agent_kwargs) if config is not None else None,
                "effective_agent": _effective_harbor_agent(config),
                "n_attempts": config.n_attempts if config is not None else None,
                # Native mode resolves the concrete job directory inside run_tasks (the name defaults
                # to a timestamp), so record the configured location rather than a not-yet-known path.
                "job_dir": str(self._job_dir) if self._job_dir is not None else None,
                "jobs_dir": str(config.jobs_dir) if config is not None else None,
                "job_name": config.job_name if config is not None else None,
                "reward_key": self._reward_key,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        """Run the Harbor job when needed, then return one trial per Harbor trial.

        In native mode the dataset directory is recovered from the tasks (each
        carries ``metadata['harbor_dataset_path']`` from
        :func:`discover_harbor_tasks`) unless a ``dataset_path`` override was given,
        so callers don't repeat it.

        ``job_dir`` doubles as a cache. Results are served straight off it only when
        **both** hold: every requested task already has ``n_attempts`` Harbor-valid
        results there, *and* the directory
        carries a cache stamp matching this run's inputs (agent contents, task
        contents, result-affecting options).

        Otherwise Harbor runs, and what happens to the directory depends on *which*
        check failed. A **stamp mismatch** discards it first: those results came from
        different inputs, so there is nothing safe to resume onto. A directory that
        merely lacks **coverage** — stamp matches, but not enough valid results —
        is handed to Harbor intact so its per-trial resume keeps the finished trials
        and runs only what is missing. Harbor may still refuse a directory on its own
        (stricter) terms; :func:`_build_native_job` then discards it and re-runs.

        The cache only engages when the config pins a stable ``job_name``; with the
        default timestamped name no fingerprint is computed at all.

        Assumes a **single writer per job directory**. Neither this runtime nor
        Harbor locks it, so two processes sharing a pinned ``job_name`` on a shared
        volume will race.
        """
        if self._config is not None:
            dataset_path = self._dataset_path or _dataset_path_from_tasks(tasks)
            job_name, job_dir = _resolve_job_dir(self._config)

            # Only fingerprint when the answer can depend on it: an unpinned job name
            # can never hit, and force_rerun/a missing dir already decided. This keeps
            # the digest I/O off every run of callers that don't pin a job name.
            stamp: dict[str, Any] | None = None
            if self._config.job_name is None or self._config.force_rerun or not job_dir.is_dir():
                stale = True
            else:
                stamp = _cache_stamp(self._config, dataset_path, tasks)
                stale = _cache_is_stale(job_dir, stamp)

            if stale or not _all_tasks_cached(job_dir, tasks, n_attempts=self._config.n_attempts):
                # Harbor's DatasetConfig matches local folder names, while SDK task ids
                # come from `[task] name`. Prefer folder names derived from the tasks
                # actually being scored so a filter like `harbor/hello-world` still
                # selects the `hello-world/` directory.
                harbor_task_names = _harbor_folder_names(tasks) or self._task_names
                job_dir, run_job = _build_native_job(
                    self._config,
                    dataset_path,
                    harbor_task_names,
                    job_name=job_name,
                    # Discard only when the inputs changed. Otherwise leave it off so
                    # Harbor resumes per trial and keeps completed work — including
                    # with `agent_dir` set, now that the scoped import path is
                    # content-addressed rather than a fresh uuid per run and Harbor's
                    # JobConfig comparison can therefore match (AALGO-430).
                    force_rerun=(self._config.force_rerun or stale),
                )
                # Fingerprint the inputs *before* running and confirm they are
                # unchanged afterwards. Stamping only the post-run state would label
                # results produced from the old sources with the new fingerprint, so
                # a later run would happily serve them. Covers what Harbor actually
                # ran: with no task_names filter that is the whole dataset, and
                # recording only the requested subset would make the next full-set
                # run look stale and re-run a complete job dir.
                coverage = _stamp_coverage(dataset_path, tasks, harbor_task_names)
                before = _cache_stamp(self._config, dataset_path, coverage)
                await run_job()
                if self._config.job_name is not None:
                    after = _cache_stamp(self._config, dataset_path, coverage)
                    if after == before:
                        _write_cache_stamp(job_dir, after)
                    else:
                        # Leaving it unstamped re-runs next time, which is the safe
                        # direction: we cannot say which inputs produced these results.
                        logger.warning(
                            "Agent or task contents changed while Harbor job %s was running; leaving it unstamped "
                            "so the next run re-executes rather than trusting these results.",
                            job_dir,
                        )
            return build_trials_from_job_dir(
                job_dir,
                tasks,
                reward_key=self._reward_key,
            )

        if self._job_dir is None:  # unreachable: __init__ guarantees config or job_dir
            raise ValueError("no job_dir configured")
        if self._run_job is not None:
            await self._run_job()
        return build_trials_from_job_dir(
            self._job_dir,
            tasks,
            reward_key=self._reward_key,
        )

    def scoring_metrics(
        self,
        task: AgentEvalTask,
        trials: Sequence[AgentEvalTrial],
    ) -> Sequence[Metric]:
        """Finalize this task's Harbor rewards through the trial-aware metrics hook.

        This method makes ``HarborAgentTaskRunner`` satisfy
        :class:`nemo_evaluator_sdk.agent_eval.trials.TrialAwareMetricsProvider`, defined in
        ``agent_eval/trials.py``. It adds secondary rewards discovered in this task's trials as
        optional outputs.
        """
        keys: set[str] = set()
        for metric in task.metrics:
            if metric_type_name(metric) == MetricType.HARBOR_REWARD:
                keys.update(output.name for output in metric.output_spec() if not output.required)
        for trial in trials:
            # A reward the verifier emitted but the adapter could not use still names an output:
            # the metric declares it and reports the rejection, rather than hiding the key.
            rewards = ParsedHarborRewards.from_metadata(trial.metadata)
            keys.update(rewards.values)
            keys.update(rewards.rejected_by_key)
        reward_keys = (self._reward_key, *sorted(keys - {self._reward_key}))
        # ``output_name`` is deliberately the runner's ``reward_key``, not the task metric's own:
        # ``discover_harbor_tasks`` builds ``HarborRewardMetric()`` with the default name, and a run
        # with ``reward_key="score"`` relies on this rename. It is why this hook lives on the runner
        # rather than on the metric.
        return [
            _harbor_reward_metric(output_name=self._reward_key, reward_keys=reward_keys)
            if metric_type_name(metric) == MetricType.HARBOR_REWARD
            else metric
            for metric in task.metrics
        ]


def _dataset_path_from_tasks(tasks: Sequence[AgentEvalTask]) -> Path:
    """Recover the Harbor dataset dir stamped on tasks by :func:`discover_harbor_tasks`."""
    for task in tasks:
        stamped = task.metadata.get("harbor_dataset_path")
        if isinstance(stamped, str) and stamped:
            return Path(stamped)
    raise ValueError(
        "native Harbor run needs a dataset path: pass dataset_path, or build tasks with "
        "discover_harbor_tasks/HarborTasksetLoader (which stamp metadata['harbor_dataset_path'])"
    )


def _harbor_folder_names(tasks: Sequence[AgentEvalTask]) -> list[str] | None:
    """Return Harbor local-dataset folder names for ``tasks``, or ``None`` if incomplete.

    Harbor's ``DatasetConfig.task_names`` matches directory names
    (``LocalTaskId.get_name()`` → ``path.name``), while SDK task ids come from
    ``[task] name``. When every task carries ``metadata['harbor_task_dir']``,
    derive the folder list so a filter like ``harbor/hello-world`` still selects
    the ``hello-world/`` directory. Return ``None`` when any task is missing that
    stamp so callers can fall back to an explicit filter.
    """
    names: list[str] = []
    for task in tasks:
        stamped = task.metadata.get("harbor_task_dir")
        if not isinstance(stamped, str) or not stamped:
            return None
        names.append(Path(stamped).name)
    return names or None


def _all_tasks_cached(job_dir: Path, tasks: Sequence[AgentEvalTask], *, n_attempts: int) -> bool:
    """Return True when each requested task has ``n_attempts`` Harbor-valid results.

    Every valid result counts, including one with ``exception_info``;
    Harbor itself treats such a result as an existing completed attempt. Missing,
    unreadable, and schema-invalid results do not count. Caching only takes effect
    when a stable ``job_name`` is set on the config; with the default timestamped
    name every run writes a fresh directory and never hits the cache.
    """
    if not job_dir.is_dir():
        return False
    requested_task_ids = {task.id for task in tasks}
    counts: dict[str, int] = {}
    for _trial_dir, data in _iter_harbor_trial_results(job_dir):
        name = data.get("task_name")
        if name in requested_task_ids:
            counts[name] = counts.get(name, 0) + 1
    return all(counts.get(task.id, 0) >= n_attempts for task in tasks)


def _feed(digest: "hashlib._Hash", label: bytes, payload: bytes) -> None:
    """Append a length-framed field to ``digest``.

    Framing matters: concatenating ``name \0 content \0`` is ambiguous because file
    *contents* may contain NUL, so two different trees can produce an identical byte
    stream. Prefixing every variable-length field with its length makes the encoding
    injective, which is what stops a collision from being read as "unchanged".
    """
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _safe_resolve(path: Path) -> Path:
    """``Path.resolve()`` that degrades instead of raising.

    The digest is a best-effort guard, not a reason to fail a run that would
    otherwise succeed, so fall back to the unresolved absolute path.

    ``RuntimeError`` is caught alongside ``OSError`` and is the case that actually
    fires: on CPython 3.12 — the floor this package targets — a **symlink loop**
    surfaces as ``RuntimeError("Symlink loop from ...")``, because ``resolve()``
    translates ``ELOOP`` before re-raising. It is not an ``OSError``, so catching
    only that would let a loop under a task directory kill the run. A loop raises
    deterministically, not as a race. ``OSError`` covers the narrower case of a
    symlink that disappears mid-walk, since ``resolve()`` calls ``os.readlink``.

    Both are 3.12/3.13 behaviours: 3.14 resolves a loop without raising at all.
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _is_executable(path: Path) -> bool:
    """Whether the owner-execute bit is set, following symlinks.

    Only the execute bit, mirroring what git tracks: read/write bits vary with umask
    and would evict the cache for nothing, but flipping +x on ``tests/test.sh`` or an
    agent entrypoint genuinely changes what Harbor does.
    """
    try:
        return bool(path.stat().st_mode & 0o100)
    except OSError:
        return False


def _digest_directory(root: Path, *, exclude: frozenset[Path] = frozenset()) -> str:
    """Content-hash a directory tree, skipping build/VCS noise and excluded roots.

    Contents rather than mtimes: callers routinely materialize the directory with
    ``copytree`` (the optimizer does, per candidate), which rewrites every mtime and
    would defeat the cache entirely.

    ``exclude`` takes *resolved* directories to skip wholesale. It exists because
    ``jobs_dir`` is caller-chosen and may sit **under** the dataset or agent
    directory; without excluding it the digest would hash the growing results tree
    it is meant to validate, and would never stabilize.

    Symlinks are **followed**, not skipped. Skipping them silently defeats the whole
    guard: a directory assembled out of links to shared sources would hash to the
    empty digest, so edits behind those links would never invalidate the cache. The
    link target is folded in alongside the contents, so re-pointing a link is a
    change even when both targets happen to hold identical bytes.

    Unreadable or vanished files are folded in as a marker rather than raised: a
    transient read failure must not kill a run that would otherwise have succeeded.
    """
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    # Keep only exclusions strictly *inside* this tree. An excluded root that is an
    # ancestor of (or equal to) `root` would otherwise match every entry and yield
    # an empty digest — silently disabling invalidation for the whole directory,
    # which is exactly the failure this function exists to prevent. jobs_dir being
    # a parent of the agent/task dir is a legitimate layout, not a reason to stop
    # hashing it.
    root_resolved = _safe_resolve(root)
    excluded = {
        resolved
        for resolved in (_safe_resolve(path) for path in exclude)
        if resolved != root_resolved and resolved.is_relative_to(root_resolved)
    }
    # Resolved dirs already walked, so a symlink cycle terminates instead of hanging.
    visited: set[Path] = set()

    def walk(directory: Path) -> None:
        resolved_dir = _safe_resolve(directory)
        if resolved_dir in visited:
            return
        visited.add(resolved_dir)
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            logger.warning("Could not list %s while fingerprinting %s: %s", directory, root, exc)
            _feed(digest, b"unlistable", b"")
            return
        for path in entries:
            if path.name in _DIGEST_SKIP_DIRS:
                continue
            resolved = _safe_resolve(path)
            if any(resolved == item or item in resolved.parents for item in excluded):
                continue

            _feed(digest, b"name", path.relative_to(root).as_posix().encode("utf-8"))
            _feed(digest, b"mode", b"x" if _is_executable(path) else b"-")
            if path.is_symlink():
                # Record where the link points, so retargeting counts as a change.
                try:
                    target = os.readlink(path).encode("utf-8")
                except OSError as exc:
                    # The link vanished mid-walk. Same contract as an unreadable
                    # file: degrade to a marker rather than kill the run.
                    logger.warning("Could not read link %s while fingerprinting %s: %s", path, root, exc)
                    target = b"<unreadable-link>"
                _feed(digest, b"symlink", target)
            if path.is_dir():
                _feed(digest, b"dir", b"")
                walk(path)
                continue
            if not path.is_file():
                # Broken link, socket, fifo: nothing to hash, but its presence counts.
                _feed(digest, b"not-a-file", b"")
                continue
            content = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    # Streamed: Harbor datasets may ship large seeds or build contexts.
                    for chunk in iter(lambda: handle.read(_DIGEST_CHUNK_BYTES), b""):
                        content.update(chunk)
            except OSError as exc:
                logger.warning("Could not read %s while fingerprinting %s: %s", path, root, exc)
                content.update(b"<unreadable>")
            # The sub-digest is fixed width, so file bytes can never be confused with
            # the framing around them.
            _feed(digest, b"file", content.digest())

    walk(root)
    return digest.hexdigest()


def _task_dirs_for(dataset_path: Path, tasks: Sequence[AgentEvalTask]) -> dict[str, Path | None]:
    """Resolve each task's on-disk directory, falling back to re-discovery.

    :func:`discover_harbor_tasks` stamps ``metadata['harbor_task_dir']``, but callers
    may build :class:`AgentEvalTask` objects by hand (the Evaluator plugin builds
    them from a job spec), so the metadata is not guaranteed.

    A task that cannot be resolved maps to ``None`` — the caller must treat that as
    un-cacheable rather than silently omitting it from the fingerprint, which would
    be a stale-cache hole.
    """
    # `_safe_resolve`, not bare `resolve()`: this walk is a best-effort cache guard, so
    # a symlink that vanishes mid-run must degrade to an unresolved absolute path
    # rather than raise out of a job that would otherwise succeed.
    dataset_root = _safe_resolve(dataset_path)
    resolved: dict[str, Path | None] = {}
    for task in tasks:
        stamped = task.metadata.get("harbor_task_dir")
        candidate = Path(stamped) if isinstance(stamped, str) and stamped else None
        # The stamp records where a task was *discovered*, which is not necessarily
        # where this run executes it: `dataset_path` can be overridden on the runner.
        # Trusting a stale or foreign path would fingerprint one dataset while Harbor
        # runs another, so anything missing or outside the active dataset is dropped
        # and re-discovered below.
        if candidate is not None:
            candidate_resolved = _safe_resolve(candidate)
            if not candidate.is_dir() or not candidate_resolved.is_relative_to(dataset_root):
                logger.debug(
                    "Ignoring stamped harbor_task_dir %s for task %r: not a directory under the active dataset %s",
                    candidate,
                    task.id,
                    dataset_root,
                )
                candidate = None
        resolved[task.id] = candidate
    if all(path is not None for path in resolved.values()):
        return resolved

    try:
        discovered = {
            task.id: Path(str(task.metadata["harbor_task_dir"])) for task in discover_harbor_tasks(dataset_path)
        }
    except (OSError, ValueError) as exc:
        # discover_harbor_tasks raises on ANY malformed task.toml in the dataset.
        # Refusing the cache is the safe reading; failing the run is not, since this
        # path previously never read those files.
        logger.warning(
            "Could not resolve Harbor task dirs under %s; treating the cache as stale: %s", dataset_path, exc
        )
        return dict.fromkeys(resolved, None)
    return {task_id: path or discovered.get(task_id) for task_id, path in resolved.items()}


def _stamp_coverage(
    dataset_path: Path,
    tasks: Sequence[AgentEvalTask],
    task_names: Sequence[str] | None,
) -> Sequence[AgentEvalTask]:
    """Tasks a written stamp must cover: everything Harbor was asked to run.

    ``task_names`` is the filter handed to Harbor's ``DatasetConfig``. When it is
    ``None`` Harbor runs every task in the dataset, which can be a superset of the
    tasks this call was asked to score — and a stamp that recorded only the smaller
    set would report the larger one as stale on the next run.
    """
    if task_names is not None:
        return tasks
    try:
        discovered = discover_harbor_tasks(dataset_path)
    except (OSError, ValueError):
        # Same reasoning as _task_dirs_for: a malformed sibling task must not fail a
        # run. Recording only the requested tasks just costs a re-run later.
        return tasks
    covered = {task.id: task for task in discovered}
    covered.update({task.id: task for task in tasks})
    return list(covered.values())


def _cache_stamp(
    config: HarborRuntimeConfig,
    dataset_path: Path,
    tasks: Sequence[AgentEvalTask],
) -> dict[str, Any]:
    """Fingerprint the inputs that decide whether a job dir can be reused.

    Covers the result-affecting options, the contents of ``agent_dir``, and the
    contents of every task directory. Two gaps are deliberate and worth knowing
    before trusting a hit: when ``agent_dir`` is ``None`` the agent is an already
    importable module, so only its *import path* is fingerprinted and edits to that
    installed package are invisible; and a task whose directory cannot be resolved
    is recorded as ``<unresolved>``, which always forces a re-run.

    Recorded per task rather than as one job-wide hash so that evaluating a
    **subset** of a previously-cached job still hits: staleness is decided only over
    the tasks actually requested.

    Excluded from the option hash: presentation and placement knobs (``quiet``,
    ``n_concurrent_trials``, ``jobs_dir``, ``job_name``, ``force_rerun``), which
    change nothing about the results; ``agent_dir``, an absolute path whose
    *content* is hashed separately, so a relocated but identical agent still hits;
    and ``reward_key``, which only selects which reward
    :func:`build_trials_from_job_dir` reads back and must not cost a Docker re-run.
    """
    options = config.model_dump(exclude=set(_CACHE_IRRELEVANT_OPTIONS), mode="json")
    # `_safe_resolve` throughout, matching `_task_dirs_for`: fingerprinting is
    # best-effort, so a symlink loop or a vanished link under any of these must
    # degrade to an unresolved path rather than raise out of `run_tasks` and fail a
    # run that would otherwise succeed.
    excluded_roots = frozenset({_safe_resolve(config.jobs_dir.expanduser())})

    agent_digest = "<none>"
    if config.agent_dir is not None:
        agent_digest = _digest_directory(_safe_resolve(config.agent_dir.expanduser()), exclude=excluded_roots)

    task_digests: dict[str, str] = {}
    for task_id, task_dir in sorted(_task_dirs_for(dataset_path, tasks).items()):
        task_digests[task_id] = (
            "<unresolved>" if task_dir is None else _digest_directory(_safe_resolve(task_dir), exclude=excluded_roots)
        )

    return {
        "version": CACHE_STAMP_VERSION,
        "options": hashlib.sha256(json.dumps(options, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "agent": agent_digest,
        "tasks": task_digests,
    }


def _cache_is_stale(job_dir: Path, stamp: Mapping[str, Any]) -> bool:
    """Return True when ``job_dir`` was not produced by the inputs in ``stamp``.

    A directory with no stamp is stale: it predates this check, or was written by
    plain Harbor, and re-running is the safe reading. An ``<unresolved>`` task
    digest is likewise always stale — we could not prove the inputs match. A
    directory that does not exist is stale too: there is nothing there to reuse, and
    answering "not stale" would be an invitation to serve zero trials.
    """
    if not job_dir.is_dir():
        return True
    try:
        stored = json.loads((job_dir / CACHE_STAMP_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stored = None

    reason: str | None = None
    if not isinstance(stored, Mapping):
        reason = "no usable cache stamp"
    elif stored.get("version") != stamp["version"]:
        reason = "cache stamp version changed"
    elif stored.get("options") != stamp["options"]:
        reason = "a result-affecting option changed"
    elif stored.get("agent") != stamp["agent"]:
        reason = "the agent directory changed"
    else:
        stored_tasks = stored.get("tasks")
        stored_tasks = stored_tasks if isinstance(stored_tasks, Mapping) else {}
        for task_id, digest in stamp["tasks"].items():
            if digest == "<unresolved>":
                reason = f"task {task_id!r} could not be resolved on disk"
                break
            if stored_tasks.get(task_id) != digest:
                reason = f"task {task_id!r} changed or was not part of the cached run"
                break

    if reason is None:
        return False
    logger.info("Re-running Harbor job %s instead of serving it from cache: %s.", job_dir, reason)
    return True


def _write_cache_stamp(job_dir: Path, stamp: Mapping[str, Any]) -> None:
    """Record the inputs a completed job dir was produced from.

    Best-effort: a job dir that could not be stamped simply re-runs next time, which
    is the safe direction. Written as a *file* deliberately — Harbor deletes any
    stray *directory* in a job dir that lacks ``result.json``.
    """
    try:
        (job_dir / CACHE_STAMP_FILENAME).write_text(
            json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Could not stamp Harbor job dir %s with its cache key: %s", job_dir, exc)


def _validated_job_dir(jobs_dir: Path, job_name: str) -> Path:
    """Resolve ``jobs_dir / job_name`` and require it stay under ``jobs_dir``.

    Harbor deletes this path on ``force_rerun`` (and again if it refuses to resume).
    ``job_name='..'``, ``'.'``, or an absolute path would otherwise ``rmtree``
    directories the caller did not name as the jobs root.
    """
    resolved_jobs_dir = jobs_dir.expanduser().resolve()
    candidate = (jobs_dir / job_name).expanduser().resolve()
    if candidate == resolved_jobs_dir or not candidate.is_relative_to(resolved_jobs_dir):
        raise ValueError(
            f"Resolved job directory {candidate} is not a strict descendant of "
            f"{resolved_jobs_dir} (job_name={job_name!r})"
        )
    return candidate


def _resolve_job_dir(config: HarborRuntimeConfig) -> tuple[str, Path]:
    """Resolve ``(job_name, job_dir)`` without importing Harbor.

    Split out from :func:`_build_native_job` because the caller must know the job
    directory *before* deciding whether to run: an unpinned ``job_name`` is a
    timestamp with microsecond precision, so resolving it twice would yield two
    different directories and the cache decision would be made about the wrong one.

    The resolved directory must be a strict descendant of ``jobs_dir``; see
    :func:`_validated_job_dir`.
    """
    job_name = config.job_name or datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S__%f")
    return job_name, _validated_job_dir(config.jobs_dir, job_name)


def _build_native_job(
    config: HarborRuntimeConfig,
    dataset_path: Path,
    task_names: Sequence[str] | None,
    *,
    job_name: str | None = None,
    force_rerun: bool | None = None,
) -> tuple[Path, RunJob]:
    """Build a Harbor ``JobConfig`` from ``config`` and return ``(job_dir, run_job)``.

    Harbor is imported inside ``run_job`` (not at module load) because it is an
    optional extra. The job name is resolved up front so ``job_dir`` is known
    without importing Harbor. When ``agent_import_path`` is set, ``run_job``
    scopes the user's agent package into ``sys.modules`` for the run and removes
    it afterwards (see :func:`scoped_harbor_agent_import`).

    Args:
        job_name: Pre-resolved job name from :func:`_resolve_job_dir`. Pass it when
            the caller already resolved the directory, so an unpinned name is not
            re-generated into a different timestamp.
        force_rerun: Overrides ``config.force_rerun`` for this build. Passed rather
            than applied via ``model_copy`` so the caller's config is never mutated
            and the job name stays fixed.
    """
    resolved_name = job_name if job_name is not None else _resolve_job_dir(config)[0]
    job_dir = _validated_job_dir(config.jobs_dir, resolved_name)
    effective_force_rerun = config.force_rerun if force_rerun is None else force_rerun

    async def run_job() -> None:
        try:
            from harbor.job import DatasetConfig, Job, JobConfig  # ty: ignore[unresolved-import,unused-ignore-comment]
            from harbor.models.job.config import RetryConfig  # ty: ignore[unresolved-import,unused-ignore-comment]
            from harbor.models.trial.config import (  # ty: ignore[unresolved-import,unused-ignore-comment]
                AgentConfig,
                ArtifactConfig,
                VerifierConfig,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_HARBOR_EXTRA_REQUIRED_MESSAGE) from exc

        if effective_force_rerun and job_dir.exists():
            shutil.rmtree(job_dir)

        artifacts: list[str | ArtifactConfig] = list(config.artifacts)
        verifier_kwargs: dict[str, Any] = {}
        if config.trace_dir is not None:
            artifacts = [ArtifactConfig(source=config.trace_dir, destination="traces"), *artifacts]
            # Nothing else tells a verifier where the traces are.
            verifier_kwargs["verifier"] = VerifierConfig(env={"TRACE_DIR": config.trace_dir})

        timeout_kwargs = {
            key: value
            for key, value in {
                "timeout_multiplier": config.timeout_multiplier,
                "agent_timeout_multiplier": config.agent_timeout_multiplier,
                "verifier_timeout_multiplier": config.verifier_timeout_multiplier,
                "agent_setup_timeout_multiplier": config.agent_setup_timeout_multiplier,
                "environment_build_timeout_multiplier": config.environment_build_timeout_multiplier,
            }.items()
            if value is not None
        }

        async def _create_and_run(agent: Any) -> None:
            job_config = JobConfig(
                job_name=resolved_name,
                jobs_dir=config.jobs_dir,
                n_attempts=config.n_attempts,
                n_concurrent_trials=config.n_concurrent_trials,
                quiet=config.quiet,
                retry=RetryConfig(max_retries=config.max_retries),
                artifacts=artifacts,
                agents=[agent],
                datasets=[DatasetConfig(path=dataset_path, task_names=list(task_names) if task_names else None)],
                **verifier_kwargs,
                **timeout_kwargs,
            )

            async def _attempt() -> None:
                job = await Job.create(job_config)
                await job.run()

            try:
                await _attempt()
            except FileExistsError as exc:
                # Harbor refuses to resume a job dir whose persisted `config.json` or
                # `lock.json` differs from this run's — and it refuses by raising, not
                # by re-running. Its comparison is deliberately stricter than the SDK
                # cache stamp: `quiet`, `n_concurrent_trials` and the `task_names`
                # filter all change the JobConfig without changing the results, so the
                # stamp excludes them (a full cache hit must not pay for a concurrency
                # tweak) while Harbor still rejects the directory. Honour the intent of
                # the rerun rather than propagating a crash.
                #
                # Identify the refusal positively before deleting anything. Both of
                # Harbor's refusals fire before any trial executes, so discarding costs
                # only completed work — but that reasoning holds *only* for those two.
                # An ordinary "file exists" raised from inside a trial, a hook, or an
                # environment build must not be mistaken for drift and answered by
                # destroying the directory.
                if not (job_dir.exists() and _is_harbor_resume_refusal(exc, job_dir)):
                    raise
                drift = _describe_job_config_drift(job_dir, job_config)
                logger.warning(
                    "Harbor refused to resume job dir %s, so it is being discarded and re-run from scratch: %s%s",
                    job_dir,
                    exc,
                    f" Differing config: {drift}." if drift else "",
                )
                shutil.rmtree(job_dir)
                await _attempt()

        agent_kwargs = dict(config.agent_kwargs)
        if config.agent_import_path is None:
            await _create_and_run(
                AgentConfig(name=config.agent_name or "oracle", model_name=config.agent_model_name, kwargs=agent_kwargs)
            )
        elif config.agent_dir is not None:
            # Loose wrapper file: make its directory importable for the run. The
            # jobs_dir exclusion must match _cache_stamp's, or a jobs_dir nested under
            # agent_dir would shift the package name as results accumulate.
            agent_dir = config.agent_dir.expanduser().resolve()
            excluded_roots = frozenset({config.jobs_dir.expanduser().resolve()})
            with scoped_harbor_agent_import(
                agent_dir, config.agent_import_path, exclude=excluded_roots
            ) as scoped_import:
                await _create_and_run(
                    AgentConfig(import_path=scoped_import, model_name=config.agent_model_name, kwargs=agent_kwargs)
                )
        else:
            # Already-importable module (installed package): let Harbor import it directly.
            await _create_and_run(
                AgentConfig(
                    import_path=config.agent_import_path, model_name=config.agent_model_name, kwargs=agent_kwargs
                )
            )

    return job_dir, run_job


def _is_harbor_resume_refusal(exc: FileExistsError, job_dir: Path) -> bool:
    """Return True when ``exc`` is Harbor declining to resume ``job_dir``.

    Separates Harbor's refusal — the one case where deleting the directory is the
    right answer — from an ordinary "file exists" surfacing from a trial, a hook or an
    environment build, where deleting it would destroy completed work to no purpose.

    Two signals, both required. Harbor constructs its refusals with a bare message, so
    ``errno`` is unset, while an OS-level ``EEXIST`` always carries one; and both
    refusals name the job directory and end in a known phrase.
    """
    if exc.errno is not None:
        return False
    message = str(exc)
    return str(job_dir) in message and any(phrase in message for phrase in _HARBOR_RESUME_REFUSALS)


def _describe_job_config_drift(job_dir: Path, job_config: Any) -> str:
    """Name the fields that differ between ``job_dir``'s persisted JobConfig and this run's.

    Harbor reports *that* an existing config differs, never *which* field, which
    leaves the resulting discard looking arbitrary. This reproduces enough of its
    comparison to say — turning "Harbor refused" into "n_concurrent_trials: 10 -> 4".

    Best-effort by construction. Returns ``""`` when the difference cannot be
    located: no ``config.json``, unparseable, or a refusal that came from
    ``lock.json`` instead, which has no JobConfig difference to report. Diagnostics
    must never mask the failure they explain, so every error here is swallowed.
    """
    try:
        stored_text = (job_dir / _HARBOR_JOB_CONFIG_FILENAME).read_text(encoding="utf-8")
        # Harbor persists with exclude_defaults=True, so the JSON omits every field
        # left at its default and comparing it raw would report phantom differences.
        # Round-tripping through the model refills them, which is what Harbor itself
        # compares after re-validating the stored config.
        stored = type(job_config).model_validate_json(stored_text).model_dump()
        current = job_config.model_dump()
        return ", ".join(
            f"{field}: {_truncated_repr(stored.get(field))} -> {_truncated_repr(value)}"
            for field, value in current.items()
            if field not in _HARBOR_EQ_IGNORED_FIELDS and stored.get(field) != value
        )
    except Exception:
        return ""


def _truncated_repr(value: Any) -> str:
    """Render ``value`` for a log line, bounded so a nested config can't flood it."""
    text = repr(value)
    return text if len(text) <= _DRIFT_VALUE_CHARS else f"{text[:_DRIFT_VALUE_CHARS]}..."


@contextlib.contextmanager
def scoped_harbor_agent_import(
    agent_dir: Path, import_path: str, *, exclude: frozenset[Path] = frozenset()
) -> Iterator[str]:
    """Make ``agent_dir`` importable under a content-addressed package for the block.

    Args:
        agent_dir: directory containing the module referenced by ``import_path``.
        import_path: Harbor agent path, ``"module"`` or ``"module:attribute"``.
        exclude: resolved directories to leave out of the content digest. Pass the
            same set :func:`_cache_stamp` uses — in practice ``jobs_dir``, which is
            caller-chosen and may sit *under* ``agent_dir``. Omitting it lets the
            growing results tree feed the package name, so the import path would
            change on every run and the resume this function exists to enable would
            never happen. See :func:`_digest_directory`.

    Yields:
        str: the rewritten import path Harbor should load (the module rooted under
        the injected synthetic package, preserving any ``:attribute`` suffix).

    Raises:
        ValueError: if ``import_path`` has no module component.

    **The package name is derived from the directory's contents, not a random
    UUID, and that is load-bearing.** This string becomes ``AgentConfig.import_path``
    and therefore part of Harbor's ``JobConfig``, which Harbor compares field-by-field
    when deciding whether an existing job directory may be resumed. A random suffix
    made that comparison fail on every rerun, so Harbor raised ``FileExistsError``
    instead of resuming and its per-trial resume was unreachable for any caller that
    sets ``agent_dir`` (AALGO-430). Content-addressing keeps distinct agents isolated
    while letting an unchanged agent resume — and makes an *edited* agent invalidate
    the job dir on Harbor's own terms.

    Identical contents therefore share a package name, so overlapping scopes are
    refcounted: the injected ``sys.modules`` entries are removed when the last
    scope exits, not the first (see :func:`_uninstall_agent_package`). The mutation
    is guarded by a process-wide lock. ``sys.modules`` is per-process, so concurrent
    *processes* were never at risk here.

    The name is ``<dirname>_<digest>``, so it tracks the directory's *location* as
    well as its contents — deliberately, because an opaque hash makes every traceback
    and import error unreadable. That is a narrow, knowing divergence from the cache
    stamp, which excludes ``agent_dir`` so a relocated but identical agent still hits
    (see :func:`_cache_stamp`). Relocating an agent while pinning the same
    ``job_name`` therefore leaves the stamp valid but changes this string, and Harbor
    declines to resume; :func:`_build_native_job` absorbs that into a clean re-run.
    The results stay correct — it costs one repeated job. Callers that rebuild agents
    under changing directory names (the Experimentalist does) are unaffected, because
    the agent name feeds their ``job_name`` too, so a rename lands in a different job
    dir with nothing to resume.

    **That last sentence only holds if the caller derives its job name from the
    *resolved* directory, as this function does.** Deriving it from the caller's
    spelling instead lets the two disagree: a symlink keeps its own name while
    resolving elsewhere, so flipping it at a fixed ``job_name`` would reuse one job
    dir for two different agents, caught only by Harbor's refusal rather than by
    design. The Experimentalist resolves first for exactly this reason
    (``resolve_harbor_run_inputs``).

    Only ``agent_dir`` (not ``sys.path``) is made importable, so a loose wrapper
    must be self-contained: a single module, or one that reaches siblings via
    relative imports (``from .helper import ...``). A wrapper that does an absolute
    ``import helper`` of a sibling file won't resolve — install it as a package and
    use the ``agent_dir``-less path instead.
    """
    module_name, sep, attribute = import_path.partition(":")
    module_name = module_name.strip().lstrip(".")
    if not module_name:
        raise ValueError("import_path must be 'module' or 'module:attribute'")
    # Hashed here rather than reused from the cache stamp: this must describe the tree
    # as it is about to be imported, and the extra walk is noise next to Docker.
    suffix = _digest_directory(agent_dir, exclude=exclude)[:_IMPORT_DIGEST_CHARS]
    package = f"{_AGENT_IMPORT_ROOT}.{_safe_identifier(agent_dir.name)}_{suffix}"
    with _IMPORT_LOCK:
        _install_agent_package(package, agent_dir)
    try:
        scoped = f"{package}.{module_name}"
        yield f"{scoped}:{attribute}" if sep else scoped
    finally:
        with _IMPORT_LOCK:
            _uninstall_agent_package(package)


def _safe_identifier(value: str) -> str:
    """Turn an arbitrary directory name into a valid Python identifier."""
    identifier = re.sub(r"\W+", "_", value).strip("_")
    if not identifier:
        return "agent"
    return identifier if identifier[0].isalpha() or identifier[0] == "_" else f"_{identifier}"


def _install_agent_package(package: str, agent_dir: Path) -> None:
    """Register ``package`` (and its parents) in ``sys.modules`` rooted at ``agent_dir``.

    Refcounted: package names are content-addressed, so two overlapping scopes on the
    same agent directory legitimately share one. Callers must hold ``_IMPORT_LOCK``.
    """
    parts = package.split(".")
    for idx in range(1, len(parts) + 1):
        name = ".".join(parts[:idx])
        if name not in sys.modules:
            module = ModuleType(name)
            module.__path__ = []  # namespace package; leaf __path__ is set below
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = module
            if idx > 1:
                setattr(sys.modules[".".join(parts[: idx - 1])], parts[idx - 1], module)
    installed = sys.modules[package]
    if not installed.__path__:
        installed.__path__ = [str(agent_dir)]
    elif installed.__path__ != [str(agent_dir)]:
        # Two directories sharing this package name share a content digest, so their
        # trees are byte-identical and the path already installed is exactly as
        # correct as this one — the excluded content (`.git`, `__pycache__`, the
        # env dirs, `jobs_dir`) is not importable. Repointing would swap the
        # directory out from under a scope that is still open, for no gain.
        logger.debug(
            "Agent package %s is already installed from %s; keeping it for the identical tree at %s",
            package,
            installed.__path__[0],
            agent_dir,
        )
    # Counted only once the injection it guards has succeeded. Incrementing first
    # would strand the count above zero if any step above raised — the scope never
    # opens, so nothing ever decrements it, and the package could never be torn down
    # again for the life of the process.
    _AGENT_PACKAGE_REFCOUNTS[package] = _AGENT_PACKAGE_REFCOUNTS.get(package, 0) + 1


def _uninstall_agent_package(package: str) -> None:
    """Remove ``package`` and its submodules from ``sys.modules`` on the last exit.

    Tearing down on the *first* exit would break a still-open scope sharing the same
    content-addressed name, so the removal waits for the refcount to reach zero.
    Callers must hold ``_IMPORT_LOCK``.
    """
    remaining = _AGENT_PACKAGE_REFCOUNTS.get(package, 0) - 1
    if remaining > 0:
        _AGENT_PACKAGE_REFCOUNTS[package] = remaining
        return
    _AGENT_PACKAGE_REFCOUNTS.pop(package, None)

    for name in [n for n in sys.modules if n == package or n.startswith(f"{package}.")]:
        sys.modules.pop(name, None)
    parent, _, child = package.rpartition(".")
    parent_module = sys.modules.get(parent)
    if parent_module is not None:
        with contextlib.suppress(AttributeError):
            delattr(parent_module, child)


def build_trials_from_job_dir(
    job_dir: str | Path,
    tasks: Sequence[AgentEvalTask],
    *,
    reward_key: str = DEFAULT_REWARD_KEY,
) -> list[AgentEvalTrial]:
    """Adapt Harbor's per-trial ``result.json`` files into :class:`AgentEvalTrial` objects.

    Reads ``<job_dir>/<task>__<hash>/result.json`` (the top-level aggregate
    ``<job_dir>/result.json`` is skipped because it is not nested). Each Harbor
    trial whose ``task_name`` matches a supplied task id becomes one trial, with
    the verifier reward in ``metadata``, typed error and measurement fields, and
    standard evidence descriptors pointing at the trial's on-disk artifacts.
    """
    validate_reward_key(reward_key)
    job_path = Path(job_dir)
    known_task_ids = {task.id for task in tasks}
    trials: list[AgentEvalTrial] = []
    for trial_dir, data in _iter_harbor_trial_results(job_path):
        task_id = data.get("task_name")
        if task_id not in known_task_ids:
            # Trial for a task we weren't asked to score (e.g. a wider dataset run).
            continue
        trials.append(
            _trial_from_harbor_result(
                trial_dir,
                data,
                reward_key=reward_key,
            )
        )

    # Surface tasks that produced no trial loudly: a mis-pointed job_dir or a
    # crashed run would otherwise silently score fewer tasks than requested.
    missing = known_task_ids - {trial.task_id for trial in trials}
    if missing:
        logger.warning("No Harbor trial result found for %d requested task(s): %s", len(missing), sorted(missing))
    if not trials:
        logger.warning(
            "No Harbor trial results under %s matched the requested tasks; nothing will be scored.", job_path
        )
    return trials


def _harbor_task_dirs(dataset_path: Path) -> list[Path]:
    """Return the Harbor task folders under ``dataset_path`` (or itself if it is one)."""
    if (dataset_path / _TASK_CONFIG_FILENAME).is_file():
        return [dataset_path]
    return sorted(
        path
        for path in dataset_path.iterdir()
        if path.is_dir() and path.name != _TASK_TEMPLATE_DIRNAME and (path / _TASK_CONFIG_FILENAME).is_file()
    )


def _strip_leading_spdx_html_comments(text: str) -> str:
    """Remove leading SPDX HTML comments from Markdown prompt content."""
    position = 0
    while match := _SPDX_HTML_COMMENT_RE.match(text, position):
        position = match.end()
    return text[position:]


def discover_harbor_tasks(dataset_path: str | Path) -> list[AgentEvalTask]:
    """Build one :class:`AgentEvalTask` per Harbor task folder in ``dataset_path``.

    Mirrors Harbor's own local-dataset discovery: every immediate subdirectory
    with a ``task.toml`` is a task. The task id is read from ``[task] name`` so it
    matches the ``task_name`` Harbor writes into each trial's ``result.json``, and
    each task is scored by a :class:`HarborRewardMetric`.

    Raises:
        ValueError: if a task's ``task.toml`` or ``instruction.md`` is malformed or
            unreadable — the offending path is named. A discovered task is never
            silently dropped, since that would quietly shrink eval coverage.
    """
    dataset_path = Path(dataset_path)
    tasks: list[AgentEvalTask] = []
    for task_dir in _harbor_task_dirs(dataset_path):
        config_path = task_dir / _TASK_CONFIG_FILENAME
        try:
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"malformed Harbor task config at {config_path}: {exc}") from exc
        task_name = config.get("task", {}).get("name", task_dir.name)
        instruction_path = task_dir / "instruction.md"
        try:
            instruction = (
                _strip_leading_spdx_html_comments(instruction_path.read_text(encoding="utf-8")).strip()
                if instruction_path.is_file()
                else task_name
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"unreadable Harbor instruction at {instruction_path}: {exc}") from exc
        tasks.append(
            AgentEvalTask(
                id=task_name,
                # `intent` is human-facing metadata, never shown to the agent; the task name is the
                # only human label Harbor's task.toml provides. The instruction the agent acts on
                # lives in `inputs["instruction"]`.
                intent=task_name,
                inputs={"instruction": instruction},
                metrics=[_harbor_reward_metric()],
                metadata={"harbor_dataset_path": str(dataset_path), "harbor_task_dir": str(task_dir)},
            )
        )
    return tasks


class HarborTasksetLoader:
    """Load a Harbor local-dataset directory as an :class:`AgentEvalTaskset`.

    Implements the :class:`AgentEvalTasksetLoader` protocol so "dataset dir in →
    tasks out" is a single call.
    """

    def __init__(self, dataset_path: str | Path, *, name: str = "harbor") -> None:
        self._dataset_path = Path(dataset_path)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def load(
        self,
        *,
        source: str | Path | None = None,
        limit: int | None = None,
        evidence_dir: Path | None = None,
    ) -> AgentEvalTaskset:
        """Discover Harbor tasks under ``source`` (or the configured path) into a taskset."""
        dataset_path = Path(source) if source is not None else self._dataset_path
        tasks = discover_harbor_tasks(dataset_path)
        if limit is not None:
            tasks = tasks[:limit]
        return AgentEvalTaskset(tasks=tasks, metadata={"harbor_dataset_path": str(dataset_path)})


async def run_harbor_eval(
    config: HarborRuntimeConfig,
    dataset_path: str | Path,
    *,
    task_names: Sequence[str] | None = None,
    metrics: Sequence[Metric] | None = None,
    run_config: AgentEvalRunConfig | None = None,
) -> AgentEvalResult:
    """Run a Harbor dataset natively and score it — the minimal-plumbing entry point.

    Loads the taskset from ``dataset_path``, runs Harbor via ``config``, and scores
    through :class:`AgentEvaluator`. Tasks are scored by :class:`HarborRewardMetric`
    unless ``metrics`` overrides them. Returns the scored :class:`AgentEvalResult`.
    """
    from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator

    dataset_path = Path(dataset_path)
    tasks = HarborTasksetLoader(dataset_path).load().tasks
    if task_names is not None:
        wanted = set(task_names)
        tasks = [task for task in tasks if task.id in wanted]
    if metrics is not None:
        tasks = [task.model_copy(update={"metrics": list(metrics)}) for task in tasks]

    runner = HarborAgentTaskRunner(config=config, task_names=task_names)
    return await AgentEvaluator().run(
        tasks=tasks,
        target=runner,
        config=run_config or AgentEvalRunConfig(),
    )


__all__ = [
    "CACHE_STAMP_FILENAME",
    "CACHE_STAMP_VERSION",
    "DEFAULT_REWARD_KEY",
    "HarborAgentTaskRunner",
    "HarborRewardMetric",
    "HarborRuntimeConfig",
    "HarborTasksetLoader",
    "build_trials_from_job_dir",
    "discover_harbor_tasks",
    "run_harbor_eval",
    "scoped_harbor_agent_import",
]


def _harbor_reward_metric(
    *,
    output_name: str = DEFAULT_REWARD_KEY,
    reward_keys: tuple[str, ...] = (),
) -> HarborRewardMetric:
    """Build the default reward metric, importing it lazily to keep this module light."""
    from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric

    return HarborRewardMetric(output_name=output_name, reward_keys=reward_keys)


def __getattr__(name: str) -> object:
    """Re-export ``HarborRewardMetric`` without importing the metric stack at module scope.

    Defining it here would pull ``MetricBase`` and the dataset-schema machinery (jinja2,
    jsonschema) onto the optimizer's light import path. See
    ``nemo_evaluator_sdk.metrics.runner_rewards``.
    """
    if name == "HarborRewardMetric":
        from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric

        return HarborRewardMetric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
