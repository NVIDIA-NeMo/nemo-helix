# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-access backend abstractions for the Experimentalist.

One :class:`ExperimentalistBackend` is built per run and shared through
:class:`ExperimentalistDeps`. The backend interface covers entity CRUD, dataset
and agent-code materialization, Intake reads, agent metadata reads, and terminal
result persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

from nemo_experimentalist_plugin.config import CandidateStorageConfig
from nemo_experimentalist_plugin.entities import (
    Candidate,
    EvaluationResult,
    ExperimentRun,
    ResourceRef,
    TrialResult,
    local_path_from_uri,
)
from nemo_experimentalist_plugin.experimentalist.atif import build_ingest_payload, read_session_id
from nemo_experimentalist_plugin.experimentalist.components.repository import (
    AgentSource,
    PRPublisher,
    _redact_url,
    clone_agent_repo,
    looks_like_git,
    split_agent_source_uri,
    split_git_ref,
)
from nemo_experimentalist_plugin.experimentalist.experiment_mirror import ExperimentMirror
from nemo_experimentalist_plugin.experimentalist.otlp import jsonl_to_protobuf, read_trace_id, spans_to_protobuf
from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult
from nemo_insights_plugin.entities import Insight
from nemo_insights_plugin.typed_client import AsyncInsightsClient
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NemoHTTPError, NotFoundError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import (
    AtifCreateRequest,
    EvaluatorResultCreateRequest,
    ListSpansQueryParams,
    Trace,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


async def _ingest_otlp(
    client: AsyncIntakeClient,
    payloads: Iterable[bytes],
    *,
    workspace: str,
    trial_id: str,
) -> None:
    """Send OTLP payloads to Intake, failing if it could not map every span.

    Intake answers 200 with a per-span error list, so a partial ingest is invisible
    unless the response body is read. Raising keeps the caller from repointing the
    trial at a trace that never fully landed; ``persist_evaluation`` catches this per
    trial, so the run continues.
    """
    for payload in payloads:
        response = (await client.create_otlp_traces(content=payload, workspace=workspace)).data()
        if response.errors:
            raise RuntimeError(
                f"Intake rejected {len(response.errors)} span(s) for trial {trial_id}: {response.errors}"
            )


async def _upload_trace_otlp(
    client: AsyncIntakeClient,
    workspace: str,
    ref: ResourceRef,
    *,
    evaluation_name: str,
    trial_id: str,
    task_id: str,
    extra_attrs: dict[str, str] | None = None,
) -> None:
    """Upload an experimentalist trace file to Intake as protobuf."""

    path = Path(urlparse(ref.uri).path)
    attrs: dict[str, str] = {
        "nemo.evaluation.name": evaluation_name,
        "nemo.test_case.name": task_id,
        "nemo.trial.id": trial_id,
        **(extra_attrs or {}),
    }
    await _ingest_otlp(
        client, jsonl_to_protobuf(path, extra_resource_attrs=attrs), workspace=workspace, trial_id=trial_id
    )


async def _upload_trace_atif(
    client: AsyncIntakeClient,
    workspace: str,
    ref: ResourceRef,
    *,
    evaluation_name: str,
    task_id: str,
    extra_attrs: dict[str, str] | None = None,
) -> None:
    """Upload an ATIF trajectory to Intake as JSON.

    Trial identity travels as the trajectory's own ``session_id``, which Intake
    adopts as the trace id — so unlike the OTLP path there is no separate
    ``trial_id`` to stamp. Intake performs the ATIF-to-spans mapping server-side.
    """

    payload = build_ingest_payload(
        ref,
        evaluation_name=evaluation_name,
        task_id=task_id,
        agent_attrs=extra_attrs or {},
    )
    payload.pop("workspace", None)
    await client.create_atif(workspace=workspace, body=AtifCreateRequest(root=payload))


_BASELINE_AGENT_LABEL = "agent-0"

_AGENT_COPY_EXCLUDE_NAMES = {
    "__pycache__",
    ".git",
    ".claude",
    ".uv",
    ".venv",
    "artifacts",
    "dataset",
    "eval-and-optimize",
    "scratch",
}


def _ignore_agent_copy(directory: str, contents: list[str]) -> set[str]:
    del directory
    return {name for name in contents if name in _AGENT_COPY_EXCLUDE_NAMES}


class ExperimentalistBackend(ABC):
    """Pluggable data-access interface for the Experimentalist."""

    def __init__(
        self,
        client: AsyncNemoClient,
        path: Path | None = None,
        storage: CandidateStorageConfig | None = None,
    ) -> None:
        self.client = client
        self.path = path
        # Candidate-archival / winner-PR settings (the config slice), so the archive/publish
        # verbs take only (workspace, candidate) and derive branch/title/body/options from here.
        self.storage = storage or CandidateStorageConfig()

    # -- Insight read --------------------------------------------------------

    @abstractmethod
    async def get_insight(self, *, workspace: str, insight_id: str) -> Insight:
        """Fetch or load the Insight identified by *insight_id*."""
        ...

    # -- ExperimentRun CRUD --------------------------------------------------

    @abstractmethod
    async def create_run(self, *, workspace: str, run: ExperimentRun) -> ExperimentRun:
        """Persist a new ExperimentRun and return it with its durable id."""
        ...

    @abstractmethod
    async def update_run(self, *, workspace: str, run: ExperimentRun) -> ExperimentRun:
        """Persist changes to an existing ExperimentRun."""
        ...

    # -- Candidate CRUD ------------------------------------------------------

    @abstractmethod
    async def create_candidate(self, *, workspace: str, candidate: Candidate) -> Candidate:
        """Persist a new Candidate and return it with its durable id."""
        ...

    @abstractmethod
    async def update_candidate(self, *, workspace: str, candidate: Candidate) -> Candidate:
        """Persist changes to an existing Candidate."""
        ...

    @abstractmethod
    async def get_candidate(self, *, workspace: str, candidate_id: str) -> Candidate:
        """Fetch a single Candidate by its durable id."""
        ...

    @abstractmethod
    async def list_candidates(self, *, workspace: str, run_id: str, include_discarded: bool = False) -> list[Candidate]:
        """Return the Candidates belonging to an ExperimentRun.

        Discarded candidates are excluded unless asked for: a rolled-back candidate is
        still on disk, and a consumer that got one back would evaluate, rank or publish
        work the run has already abandoned.
        """
        ...

    # -- Result persistence --------------------------------------------------

    @abstractmethod
    async def persist_result(self, *, workspace: str, result: ExperimentalistResult) -> None:
        """Persist the terminal result for an Experimentalist run.

        The result identifies the run and carries the terminal summary,
        completed round count, and optional winning Candidate.
        """
        ...

    @abstractmethod
    async def persist_evaluation(
        self,
        *,
        workspace: str,
        result: EvaluationResult,
        candidate: Candidate,
        split: str,
    ) -> None:
        """Persist traces and metrics for a completed evaluation.

        Derives the Evaluation name internally from candidate × split, ensuring
        the Evaluation exists before stamping traces.
        """
        ...

    @abstractmethod
    async def get_evaluation_name(self, *, workspace: str, candidate: Candidate, split: str) -> str:
        """Best-effort Evaluation name for *candidate* × *split* in *workspace*.

        Returns "" when projection fails — the name only tags Intake resource
        attributes, so a run must not break on it.
        """
        ...

    # -- Agent code access ---------------------------------------------------

    @abstractmethod
    async def get_agent_code(
        self, *, workspace: str, agent: str | Path, dest: Path, clone_depth: int | None = None
    ) -> AgentSource | None:
        """Materialize the agent's code files into *dest*.

        Returns the git :class:`AgentSource` provenance when *agent* is a git
        ``url@ref[#agent_path]`` (the clone retains ``.git`` so *dest* can serve as a PR/MR push
        target); returns ``None`` for a local path or non-git source. *clone_depth* optionally
        makes a shallow clone.
        """
        ...

    async def _clone_git_agent(self, agent: str, dest: Path, *, clone_depth: int | None = None) -> AgentSource:
        """Clone a git ``url@ref`` agent into *dest*, surfacing failures as ``ValueError``.

        Shared by every backend: git-sourced agents fetch identically regardless of
        backend. Safe :class:`AgentCloneError` details pass through; a legacy raw
        ``CalledProcessError`` is replaced so both reach the CLI's clean error path.
        """
        try:
            return await asyncio.to_thread(clone_agent_repo, agent, dest, clone_depth=clone_depth)
        except subprocess.CalledProcessError:
            remote, _ = split_git_ref(split_agent_source_uri(agent)[0])
            raise ValueError(f"failed to fetch --agent {_redact_url(remote)!r}") from None

    @abstractmethod
    async def archive_candidate(self, *, workspace: str, candidate: Candidate) -> str | None:
        """Persist a produced candidate's code to durable storage; return a locator for it, or None.

        Records the returned locator as the candidate's ``source_link``. Returns ``None`` when this
        backend cannot archive the candidate (e.g. no code source to store from) or there is nothing
        new to persist. How and where the code is stored is backend-specific. Best-effort: callers
        gate on :attr:`storage`.archive_candidates and never let a failure fail the run.
        """
        ...

    @abstractmethod
    async def publish_candidate(self, *, workspace: str, candidate: Candidate) -> str | None:
        """Surface a candidate (the winner) as a reviewable change; return its URL, or None.

        Records the returned URL as the candidate's ``source_link``. Returns ``None`` when this
        backend cannot publish (e.g. no code source) or there is nothing to publish. How the change
        is surfaced is backend-specific. Callers gate on :attr:`storage`.publish_winner.
        """
        ...

    # -- Agent reads ---------------------------------------------------------

    @abstractmethod
    async def get_ethos(self, *, workspace: str, ethos: str, dest: Path) -> Path:
        """Materialize the Ethos URI into *dest* and return *dest*."""
        ...


def load_candidate(path: Path) -> Candidate:
    """Deserialize one stored candidate record into a :class:`Candidate`.

    Free function rather than a backend method: read-only callers (the CodeEditBuilder's
    workspace tool) need the deserialization without constructing a backend,
    which would create directories and projection state as a side effect.
    """
    data = json.loads(path.read_text())
    candidate = Candidate.model_validate(data)
    # Restore private _id if present in the serialized form.
    entity_id = data.get("id", "")
    if entity_id:
        candidate._id = entity_id  # type: ignore[attr-defined]
    return candidate


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* so a reader never sees a partial file.

    ``os.replace`` is atomic within a filesystem, so a crash leaves either the previous
    contents or the new ones — never a truncated file.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_winner(run_dir: Path) -> Candidate:
    """The winning Candidate of a finished run, read from its stored record.

    Here so consumers do not rebuild the run layout themselves. ``run.json`` names the
    winner by **label** — 'agent-3', the form every artifact path in the run is built
    from — while the records under ``candidates/`` are filed by id, so this resolves one
    to the other. The winner's location then comes from its own artifact reference,
    which stays true even if candidates move off local disk.

    Args:
        run_dir: The run's directory — the one holding ``run.json`` and ``candidates/``.

    Raises:
        FileNotFoundError: if the run or the record it names is missing.
        ValueError: if the run completed without selecting a winner.
    """
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    winner_label = document.get("winner_agent")
    if not isinstance(winner_label, str) or not winner_label:
        raise ValueError(f"Run at {run_dir} completed without a selected winner")
    for record in sorted((run_dir / "candidates").glob("*.json")):
        candidate = load_candidate(record)
        if candidate.label == winner_label:
            return candidate
    raise FileNotFoundError(
        f"Run at {run_dir} names winner {winner_label!r}, but no candidate record carries that label"
    )


def _load_entity(cls: type[_ModelT], path: Path) -> _ModelT:
    """Deserialize *path* as JSON into *cls*, restoring the private ``_id`` field."""
    data = json.loads(path.read_text())
    entity_id = data.get("id", "")
    obj = cls.model_validate(data)
    if entity_id:
        cast(Any, obj)._id = entity_id  # PrivateAttr set the same way as the real entity client
    return obj


class LocalExperimentalistBackend(ExperimentalistBackend):
    """Persist entities under the optimizer's working directory.

    Entity CRUD and result persistence land in local files under the same
    ``eval-and-optimize/`` tree that AAD created::

        {path}/
          eval-and-optimize/
            insight.json          ← loaded via get_insight(insight_id=path)
            run.json              ← ExperimentRun state
            candidates/
              <candidate-id>.json ← Candidate metadata and measurements
              ...
            agents/
              agent-0/            ← the candidates' artifacts
              agent-1/
              ...
            analysis/
              round-0.md
              round-0-goal.json
              ...
            results/
            OPTIMIZATION.md       ← final report (same as AAD)

    Insight references may be local filesystem paths or Platform IDs. Dataset
    and agent-code references must be local filesystem paths.
    """

    def __init__(
        self,
        *,
        client: AsyncNemoClient,
        path: Path,
        storage: CandidateStorageConfig | None = None,
    ) -> None:
        super().__init__(client, path, storage)
        self.path = path
        self.intake_client = AsyncIntakeClient.from_client(client)
        self._eo = path / "eval-and-optimize"
        for subdir in ("agents", "analysis", "candidates", "results"):
            (self._eo / subdir).mkdir(parents=True, exist_ok=True)
        # Best-effort, one-way projection onto native platform Experiments. Mirrors are built
        # lazily by ``_project_best_effort`` and cached per workspace so reusing this backend
        # across workspaces never projects into the wrong one.
        self._mirrors: dict[str, ExperimentMirror] = {}
        # Run-level git provenance captured by get_agent_code: the fetched AgentSource (repo,
        # ref, sub-path) and the local .git clone that is the push target. archive_candidate /
        # publish_candidate derive everything from these, so their signatures stay (workspace, candidate).
        self._agent_source: AgentSource | None = None
        self._agent_checkout: Path | None = None
        self._pr_url: str | None = None
        # label -> archived branch source_link, so re-projection keeps the real git link
        # (set by archive_candidate) instead of falling back to the pseudo placeholder.
        self._candidate_source_links: dict[str, str] = {}

    async def _project_best_effort(self, workspace: str, call: Callable[[ExperimentMirror], Awaitable[None]]) -> None:
        """Run a mirror projection best-effort (spec §3 / F).

        Builds the workspace's mirror lazily (cached per workspace), runs *call* against it, and
        logs+swallows any failure so a projection problem can never fail the run or the
        local-file persistence. One-way: nothing read back into the loop.
        """
        mirror = self._mirrors.get(workspace)
        try:
            if mirror is None:
                mirror = self._mirrors[workspace] = ExperimentMirror(self.intake_client, workspace)
            await call(mirror)
        except Exception as exc:  # noqa: BLE001 - projection must never fail the run
            logger.warning("[MIRROR] projection failed (run continues): %s", exc)

    # -- Insight read --------------------------------------------------------

    async def get_insight(self, *, workspace: str, insight_id: str) -> Insight:
        # A local insight file preserves local-fixture behavior; otherwise treat
        # ``insight_id`` as a platform insight id and fetch it from the Insights
        # API. The dispatch is heuristic: an id that happens to match a path in
        # the cwd is read as a file, so callers wanting the platform must use an id
        # that isn't also a local path.
        p = Path(insight_id)
        if p.exists():
            return _load_entity(Insight, p)
        try:
            return await AsyncInsightsClient.from_client(self.client).get_insight(
                workspace=workspace, insight_id=insight_id
            )
        except NotFoundError as exc:
            raise ValueError(f"Insight not found on the platform: {insight_id!r}") from exc
        except NemoHTTPError as exc:
            raise ValueError(f"Failed to fetch insight {insight_id!r} from the platform: {exc}") from exc

    # -- Agent code access ---------------------------------------------------

    async def get_agent_code(
        self, *, workspace: str, agent: str | Path, dest: Path, clone_depth: int | None = None
    ) -> AgentSource | None:
        # Git source: clone (keeps .git for the PR push target) and return provenance.
        # Stash it so the mirror can use the real repo@ref as the baseline's source_link.
        if looks_like_git(str(agent)):
            self._agent_source = await self._clone_git_agent(str(agent), dest, clone_depth=clone_depth)
            self._agent_checkout = dest  # the .git clone is the archive/PR push target
            return self._agent_source
        # Local path: copy the agent into dest, replacing it (mirror clone_agent_repo)
        # so files removed from the local agent don't linger and get optimized/published
        # as stale code.
        src = Path(agent)
        if not src.exists():
            raise FileNotFoundError(f"Local agent path not found: {src}")
        if dest.resolve() != src.resolve():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest, ignore=_ignore_agent_copy)
        return None

    def _candidate_branch(self, candidate: Candidate) -> str:
        """Deterministic archive branch for a candidate: ``<prefix>/<run-id>/<label>``."""
        return f"{self.storage.candidate_branch_prefix}/{candidate.run_id}/{candidate.label}"

    def _candidate_code_dir(self, candidate: Candidate) -> Path:
        """Local directory holding *candidate*'s code, read off its artifact reference.

        Raises:
            ValueError: if the artifact does not exist. Falling back to the parent would
                resolve to the shared candidate root and push every candidate's code as
                this one's.
        """
        path = local_path_from_uri(candidate.artifact.uri, context="Candidate artifact")
        if not path.exists():
            raise ValueError(f"Candidate {candidate.label!r} ({candidate.id}) has no artifact at {path}")
        return path if path.is_dir() else path.parent

    async def archive_candidate(self, *, workspace: str, candidate: Candidate) -> str | None:
        """Git implementation: push ``<prefix>/<run-id>/<label>`` (subtree at the agent's
        ``agent_path``) cut from the source ref, and record ``{repo}@{branch}`` as the candidate's
        Experiment source_link. No-op (None) for a non-git source, or a skipped/empty-diff branch."""
        # Non-git source (local agent) or no checkout: nothing to push, so archival is a no-op.
        src = self._agent_source
        if src is None or self._agent_checkout is None:
            return None
        checkout, branch = self._agent_checkout, self._candidate_branch(candidate)
        code_dir = self._candidate_code_dir(candidate)
        message = (
            f"Experimentalist candidate {candidate.label} "
            f"(generation {candidate.generation}): {candidate.description}\n\n"
            f"Candidate-Id: {candidate.id}"
        )
        pushed = await asyncio.to_thread(
            lambda: PRPublisher(agent_dir=checkout).push_branch(
                src_dir=code_dir, branch=branch, base_ref=src.ref, agent_path=src.agent_path, message=message
            )
        )
        if not pushed:
            return None
        # Record the branch as the candidate's real source_link and re-project its Experiment
        # so the mirror surfaces the git branch instead of the pseudo placeholder.
        link = f"{src.repo_url}@{branch}"
        self._candidate_source_links[candidate.label] = link
        await self._project_best_effort(
            workspace, lambda m: m.project_candidate(candidate, agent_source=src, source_link=link)
        )
        return link

    async def publish_candidate(self, *, workspace: str, candidate: Candidate) -> str | None:
        """Git implementation: open a draft PR/MR from the candidate's ``<prefix>/<run-id>/<label>``
        branch (title/body derived from the candidate + run summary + archived-branch links, per the
        ``storage`` PR options) and stash the URL for the winner's source_link. No-op (None) for a
        non-git source or empty diff."""
        # Non-git source or no checkout: cannot open a PR, so this is a no-op.
        src = self._agent_source
        if src is None or self._agent_checkout is None:
            return None
        checkout, branch = self._agent_checkout, self._candidate_branch(candidate)
        # A PR title is one line, and `description` is the Proposer's graph-level prose —
        # often several sentences, sometimes with newlines — which forges truncate or
        # reject, and which `_compose_pr_body` already puts in the body. The optimization
        # type is the short handle, and lives in the payload the Proposer and Builder own.
        payload = candidate.generated_from.payload if candidate.generated_from is not None else {}
        kind = payload.get("optimization_type")
        title = self.storage.pr_title or (
            f"Experimentalist: candidate {candidate.label} ({kind if isinstance(kind, str) and kind else 'improvement'})"
        )
        body = self.storage.pr_body or await self._compose_pr_body(workspace=workspace, candidate=candidate)
        code_dir = self._candidate_code_dir(candidate)
        url = await asyncio.to_thread(
            lambda: PRPublisher(agent_dir=checkout).publish(
                winner_dir=code_dir,
                branch=branch,
                base_ref=src.ref,
                agent_path=src.agent_path,
                title=title,
                body=body,
                draft=self.storage.pr_draft,
                base_branch_override=self.storage.pr_base_branch,
                labels=self.storage.pr_labels,
            )
        )
        # Stash the PR URL so the mirror's finalize() records it on the winner's source_link.
        self._pr_url = url or None
        return url or None

    async def _compose_pr_body(self, *, workspace: str, candidate: Candidate) -> str:
        """Compose the winner PR body: run summary + links to every archived candidate branch."""
        run_path = self._eo / "run.json"
        run = _load_entity(ExperimentRun, run_path) if run_path.exists() else None
        summary = (run.summary if run is not None else None) or "Experimentalist run complete."
        if not self.storage.archive_candidates:
            return summary  # only the winner's branch exists; nothing else to link
        siblings = await self.list_candidates(workspace=workspace, run_id=candidate.run_id)
        lines = ["", "## Candidate branches", ""]
        for sib in siblings:
            if sib.label == _BASELINE_AGENT_LABEL:
                continue
            marker = " (winner)" if sib.label == candidate.label else ""
            metrics = sib.rewards["validation"].metrics or sib.rewards["train"].metrics or {}
            lines.append(f"- `{sib.label}`{marker}: `{self._candidate_branch(sib)}` — metrics={metrics}")
        return summary + "\n" + "\n".join(lines) + "\n"

    # -- Agent reads ---------------------------------------------------------

    async def get_ethos(self, *, workspace: str, ethos: str, dest: Path) -> Path:
        src = Path(ethos)
        if not src.is_file():
            raise FileNotFoundError(f"Ethos not found: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    # ------------------------------------------------------------------
    # ExperimentRun CRUD  (eval-and-optimize/run.json)
    # ------------------------------------------------------------------

    async def create_run(self, *, workspace: str, run: ExperimentRun) -> ExperimentRun:
        if not run.id:
            run._id = str(uuid.uuid4())  # type: ignore[attr-defined]
        _atomic_write(self._eo / "run.json", run.model_dump_json(indent=2))
        await self._project_best_effort(workspace, lambda m: m.ensure_group(run))
        return run

    async def update_run(self, *, workspace: str, run: ExperimentRun) -> ExperimentRun:
        # Written through a temp file and renamed, because this is rewritten on every
        # progress report and it is the only record of which run owns the candidates on
        # disk. A plain write_text truncates first, so killing the process mid-round leaves
        # a half-written file that resume cannot parse.
        _atomic_write(self._eo / "run.json", run.model_dump_json(indent=2))
        await self._project_best_effort(workspace, lambda m: m.update_group(run))
        return run

    # ------------------------------------------------------------------
    # Candidate CRUD  (eval-and-optimize/candidates/{id}.json)
    #
    # Metadata lives in a store the runner owns, keyed by the candidate's durable
    # id — never inside the artifact it describes. Listing therefore reads the
    # store rather than walking a directory layout, so a strategy that does not
    # produce one directory per candidate can still store, list and resume.
    # ------------------------------------------------------------------

    def _candidate_path(self, candidate_id: str) -> Path:
        """Where *candidate_id*'s record lives, refusing an id that is not one name.

        The id is interpolated into a path, so one containing a separator escapes the
        candidates directory: ``../run`` lands on ``eval-and-optimize/run.json`` and a
        candidate write would overwrite the run. Ids are minted as uuids here, but a
        Candidate can arrive from a component this repo did not write -- which is the
        point of the registry -- so the check belongs at the path, not at the mint.
        """
        if not candidate_id:
            raise ValueError("Candidate id is required")
        if candidate_id != Path(candidate_id).name or candidate_id in {".", ".."}:
            raise ValueError(f"Candidate id must be a single path component, got {candidate_id!r}")
        return self._eo / "candidates" / f"{candidate_id}.json"

    def _write_candidate(self, candidate: Candidate) -> None:
        path = self._candidate_path(candidate.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic for the same reason as run.json: this runs on every record_reward, and a
        # truncated candidates/<id>.json makes list_candidates raise, which leaves the whole
        # run unresumable rather than losing one record.
        _atomic_write(path, json.dumps({**candidate.model_dump(), "id": candidate.id}, indent=2))

    async def create_candidate(self, *, workspace: str, candidate: Candidate) -> Candidate:
        if not candidate.id:
            candidate._id = str(uuid.uuid4())  # type: ignore[attr-defined]
        self._write_candidate(candidate)
        await self._project_best_effort(
            workspace,
            lambda m: m.project_candidate(
                candidate,
                agent_source=self._agent_source,
                source_link=self._candidate_source_links.get(candidate.label),
            ),
        )
        return candidate

    async def update_candidate(self, *, workspace: str, candidate: Candidate) -> Candidate:
        self._write_candidate(candidate)
        await self._project_best_effort(
            workspace,
            lambda m: m.project_candidate(
                candidate,
                agent_source=self._agent_source,
                source_link=self._candidate_source_links.get(candidate.label),
            ),
        )
        return candidate

    async def get_candidate(self, *, workspace: str, candidate_id: str) -> Candidate:
        return load_candidate(self._candidate_path(candidate_id))

    async def list_candidates(self, *, workspace: str, run_id: str, include_discarded: bool = False) -> list[Candidate]:
        store = self._eo / "candidates"
        if not store.is_dir():
            return []
        candidates = [load_candidate(path) for path in sorted(store.glob("*.json"))]
        live = (c for c in candidates if c.run_id == run_id and (include_discarded or not c.discarded))
        return sorted(live, key=lambda c: (c.generation, c.label))

    # ------------------------------------------------------------------
    # Result persistence  (eval-and-optimize/OPTIMIZATION.md)
    # ------------------------------------------------------------------

    async def persist_evaluation(
        self, *, workspace: str, result: EvaluationResult, candidate: Candidate, split: str
    ) -> None:
        """Persist traces and metrics for a completed evaluation.

        Side effect: For trials with local file:// traces, uploads to Intake and rewrites
        trial.trace.uri to intake://traces/{id}. Original URI preserved in metadata.

        TODO(future): Uploading traces can be slow (large payloads, many trials). Consider
        moving this to a background task so the optimization loop can continue without
        blocking on Intake I/O. The loop would need to either: (a) await the background
        task before _reward_trajectories (which reads intake:// URIs), or (b) fall back
        to reading local file:// traces if upload is still in progress.
        """
        evaluation_name = await self.get_evaluation_name(workspace=workspace, candidate=candidate, split=split)
        if not evaluation_name:
            return  # projection failed (shouldn't happen, but defensive)

        agent_attrs = {
            k: str(result.metadata[k])
            for k in ("gen_ai.agent.name", "agent.version", "gen_ai.request.model")
            if k in result.metadata
        }
        for trial in result.trials:
            if trial.trace is None:
                logger.warning(
                    f"[INTAKE] Trial {trial.id} ({trial.task_id}) has no trace; skipping persistence. "
                    "The evaluator found no trace artifact in the configured format."
                )
                continue
            try:
                await self._persist_trial(
                    trial, workspace=workspace, evaluation_name=evaluation_name, agent_attrs=agent_attrs
                )
            except Exception as exc:  # noqa: BLE001 - Intake persistence is best-effort
                logger.warning(f"[INTAKE] persist_evaluation failed for trial {trial.id}: {exc}")

    async def _persist_trial(
        self, trial: TrialResult, *, workspace: str, evaluation_name: str, agent_attrs: dict[str, str]
    ) -> None:
        assert trial.trace is not None
        uri = trial.trace.uri
        if uri.startswith("intake://"):
            trace_id = uri.removeprefix("intake://traces/")
            trace = await self._retrieve_trace_with_retry(trace_id, workspace=workspace)
            ctx = trace.evaluation_context
            if ctx is None or ctx.get("evaluation_name") != evaluation_name:
                rows: list[dict[str, Any]] = []
                query_params: ListSpansQueryParams = {
                    "filter": {"trace_id": trace_id},
                    "mode": "detailed",
                    "page_size": 1000,
                }
                spans = await self.intake_client.list_spans(workspace=workspace, query_params=query_params)
                async for span in spans.items():
                    rows.append(span.model_dump(mode="json", exclude_none=True))
                attrs = {
                    "nemo.evaluation.name": evaluation_name,
                    "nemo.test_case.name": trial.task_id,
                    "nemo.trial.id": trial.id,
                    **agent_attrs,
                }
                await _ingest_otlp(
                    self.intake_client, spans_to_protobuf(rows, attrs), workspace=workspace, trial_id=trial.id
                )
                trace = await self._retrieve_trace_with_retry(trace_id, workspace=workspace)
        else:
            trace_format = str(trial.trace.metadata.get("trace_format", "otlp"))
            try:
                trace_id = read_session_id(trial.trace) if trace_format == "atif" else read_trace_id(trial.trace)
            except (ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
                # Empty, invalid, or missing trace file (e.g., cancelled trial) — skip upload
                logger.debug(f"[INTAKE] No valid trace found for trial {trial.id}: {exc}; skipping upload")
                return
            original_uri = uri
            if trace_format == "atif":
                await _upload_trace_atif(
                    self.intake_client,
                    workspace,
                    trial.trace,
                    evaluation_name=evaluation_name,
                    task_id=trial.task_id,
                    extra_attrs=agent_attrs,
                )
            else:
                await _upload_trace_otlp(
                    self.intake_client,
                    workspace,
                    trial.trace,
                    evaluation_name=evaluation_name,
                    trial_id=trial.id,
                    task_id=trial.task_id,
                    extra_attrs=agent_attrs,
                )
            trial.trace = ResourceRef(
                uri=f"intake://traces/{trace_id}",
                description=trial.trace.description,
                metadata={**trial.trace.metadata, "local_uri": original_uri},
            )
            trace = await self._retrieve_trace_with_retry(trace_id, workspace=workspace)
        for name, metric in trial.metrics.items():
            if trace.root_span_id is None:
                raise RuntimeError(f"Trace {trace.id!r} has no root span id; cannot attach evaluator result {name!r}")
            await self.intake_client.create_evaluator_result(
                workspace=workspace,
                body=EvaluatorResultCreateRequest(
                    span_id=trace.root_span_id,
                    session_id=trace.session_id,
                    name=name,
                    value=float(metric.value),
                    data_type="NUMERIC",
                ),
            )

    async def _retrieve_trace_with_retry(
        self, trace_id: str, *, workspace: str, retries: int = 5, initial_delay: float = 1.0
    ) -> Trace:
        """Retrieve trace with exponential backoff.

        Intake indexing after OTLP upload can take several seconds. Uses exponential
        backoff: 1s, 2s, 4s, 8s, 16s (up to 31s total) by default.
        """
        last_exc: Exception = RuntimeError(f"trace {trace_id!r} not found after {retries} retries")
        delay = initial_delay
        for attempt in range(retries):
            try:
                return (
                    await self.intake_client.get_trace(
                        id=trace_id, workspace=workspace, query_params={"mode": "detailed"}
                    )
                ).data()
            except NotFoundError as exc:
                last_exc = exc
                if attempt < retries - 1:  # Don't sleep after the last attempt
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
        raise last_exc

    async def get_evaluation_name(self, *, workspace: str, candidate: Candidate, split: str) -> str:
        mirror = self._mirrors.get(workspace)
        try:
            if mirror is None:
                mirror = self._mirrors[workspace] = ExperimentMirror(self.intake_client, workspace)
            return await mirror.ensure_experiment(candidate, split=split)
        except Exception as exc:  # noqa: BLE001 - projection is best-effort
            logger.warning("[MIRROR] ensure_experiment failed: %s", exc)
            return ""

    async def persist_result(self, *, workspace: str, result: ExperimentalistResult) -> None:
        report_path = self._eo / "OPTIMIZATION.md"
        # The optimizer's report writer creates the full document before this call.
        # Use the compact result summary only as a fallback when no report was produced.
        if not report_path.exists() or not report_path.read_text().strip():
            report_path.write_text(result.summary)
        run_path = self._eo / "run.json"
        if run_path.exists():
            run: ExperimentRun = _load_entity(ExperimentRun, run_path)
            run.status = "completed"
            run.progress_completed = result.progress_completed
            run.summary = result.summary  # so publish_candidate's _compose_pr_body reads the real summary
            if result.winner is not None:
                run.winner_agent = result.winner.label
            _atomic_write(run_path, run.model_dump_json(indent=2))
        await self._project_best_effort(
            workspace,
            lambda m: m.finalize(
                run_id=result.run_id, summary=result.summary, winner=result.winner, pr_url=self._pr_url
            ),
        )


def make_experimentalist_backend(
    *,
    client: AsyncNemoClient,
    experiments_output: str,
    storage: CandidateStorageConfig | None = None,
) -> ExperimentalistBackend:
    """Build the Experimentalist backend.

    Entity state is written under *experiments_output* using the ``eval-and-optimize/``
    tree layout. Evaluations and candidates are also projected to native Experiments on
    a best-effort basis.

    Args:
        client(AsyncNemoClient): Platform API client.
        experiments_output(str): Local output directory.
        storage(CandidateStorageConfig | None): Candidate-archival / winner-PR settings.
    Returns:
        ExperimentalistBackend: The experimentalist backend.
    """
    return LocalExperimentalistBackend(client=client, path=Path(experiments_output), storage=storage)
