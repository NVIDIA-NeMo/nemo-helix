# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-way, best-effort projection of the plugin's run/candidate entities onto the
platform's native ``ExperimentGroup``/``Experiment``.

This module is the *only* place that talks to ``client.experiments`` /
``client.evaluations``; the optimization loop never imports it. Mapping is
``ExperimentRun → ExperimentGroup`` (1:1) and ``Candidate → Experiment[]`` (one per
evaluated split). It mirrors **structure only** (identity/lineage/status/description);
eval results (reward/trials) are NOT copied into ``Experiment.metadata`` — those arrive
via the Intake rollup path (spec §4.3).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from nemo_experimentalist_plugin.entities import Candidate, ExperimentRun
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import (
    EvaluationCreateRequest,
    ExperimentCreateRequest,
    ExperimentUpdateRequest,
)

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"[^a-z0-9-]+")


def _slug(value: str) -> str:
    return _NAME_RE.sub("-", value.lower()).strip("-") or "run"


def group_name(run_id: str) -> str:
    """Deterministic, workspace-unique ExperimentGroup name from the run id."""
    return f"opt-{_slug(run_id)}"[:63].rstrip("-")


def experiment_name(gname: str, label: str, split: str) -> str:
    """Deterministic, workspace-unique Experiment name (one per candidate × split).

    ``label`` is sanitized with the same ``_slug`` as the group's run id. If the composed
    name would exceed the platform's 63-char limit, a short deterministic hash suffix
    preserves uniqueness — so distinct candidates/splits can never truncate onto the same
    name and silently overwrite each other via the create→conflict→update upsert path.
    """
    name = f"{gname}-{_slug(label)}-{split}"
    if len(name) <= 63:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]  # noqa: S324 - identity, not security
    return f"{name[:54].rstrip('-')}-{digest}"


def pseudo_source_link(gname: str, label: str) -> str:
    """Fallback HTTP URL, stable per candidate across its splits."""
    return f"https://nemo.local/optimizations/{gname}/candidate/{label}"


def experiment_status(candidate: Candidate) -> str:
    """Derive the producer status string from candidate lineage (spec §4.1)."""
    if candidate.discarded:
        return "discarded"
    if candidate.killed_generation is not None:
        return "killed"
    if candidate.is_baseline:
        return "baseline"
    return "survived"


def group_metadata(run: ExperimentRun) -> dict[str, str]:
    """The ExperimentRun fields with no first-class ExperimentGroup home (spec §4.1).

    Platform ``metadata`` is ``dict[str, str]``: every value must be a string. Non-string
    fields are serialized (``config_snapshot`` as JSON, the progress counter via ``str``);
    ``winner_candidate`` is omitted until a winner exists rather than sent as ``None``.
    """
    md = {
        "agent": run.agent,
        "config_snapshot": json.dumps(run.config_snapshot, sort_keys=True),
        "status": run.status,
        "progress_completed": str(run.progress_completed),
        "progress_unit": run.progress_unit,
    }
    if run.progress_total is not None:
        md["progress_total"] = str(run.progress_total)
    if run.winner_agent is not None:
        md["winner_candidate"] = run.winner_agent
    return md


def experiment_metadata(candidate: Candidate, split: str) -> dict[str, str]:
    """Identity/grouping metadata only. Eval results (reward/trials) are not copied here;
    scores and traces arrive via the Intake rollup path (spec §4.3).

    Platform ``metadata`` is ``dict[str, str]``, so ``generation`` is serialized via ``str``.
    Both the durable id and the display label are recorded: ``ancestor`` references are
    ids, while Experiment names are built from labels, so a consumer reconstructing
    lineage needs the pair."""
    return {
        "generation": str(candidate.generation),
        "candidate_id": candidate.id,
        "candidate_label": candidate.label,
        "split": split,
    }


class ExperimentMirror:
    """Best-effort, one-way projection to native Experiments. Callers wrap each method
    so failures don't propagate (spec F)."""

    def __init__(self, client: AsyncIntakeClient, workspace: str) -> None:
        self._client = client
        self._workspace = workspace
        self._group_ids: dict[str, str] = {}  # run_id -> ExperimentGroup id
        # Keyed by candidate *id*, because that is what ``Candidate.ancestor`` holds. The
        # Experiment *name* is built from the display label instead — it is meant to be
        # greppable — so the two are kept side by side rather than derived from each other.
        self._experiment_ids: dict[tuple[str, str], str] = {}  # (candidate id, split) -> Experiment id
        self._labels: dict[str, str] = {}  # candidate id -> display label

    # -- ExperimentGroup ----------------------------------------------------

    async def ensure_group(self, run: ExperimentRun) -> None:
        gname = group_name(run.id)
        body = self._group_create_request(run, gname)
        try:
            grp = (await self._client.create_experiment(workspace=self._workspace, body=body)).data()
        except ConflictError:
            grp = (await self._client.get_experiment(name=gname, workspace=self._workspace)).data()
        self._group_ids[run.id] = grp.id

    async def update_group(self, run: ExperimentRun) -> None:
        gname = group_name(run.id)
        await self._client.update_experiment(
            name=gname,
            workspace=self._workspace,
            body=self._group_update_request(run, gname),
        )

    def _group_create_request(self, run: ExperimentRun, gname: str) -> ExperimentCreateRequest:
        if run.insight:
            return ExperimentCreateRequest(
                name=gname,
                insight_id=run.insight,
                summary=run.summary or "",
                metadata=group_metadata(run),
            )
        return ExperimentCreateRequest(
            name=gname,
            summary=run.summary or "",
            metadata=group_metadata(run),
        )

    def _group_update_request(self, run: ExperimentRun, gname: str) -> ExperimentUpdateRequest:
        if run.insight:
            return ExperimentUpdateRequest(
                name=gname,
                insight_id=run.insight,
                summary=run.summary or "",
                metadata=group_metadata(run),
            )
        return ExperimentUpdateRequest(
            name=gname,
            summary=run.summary or "",
            metadata=group_metadata(run),
        )

    async def _group_id_for(self, run_id: str) -> str:
        gid = self._group_ids.get(run_id)
        if gid is None:  # resume path: group exists, look it up by deterministic name
            grp = (await self._client.get_experiment(name=group_name(run_id), workspace=self._workspace)).data()
            gid = self._group_ids[run_id] = grp.id
        return gid

    # -- Experiment ---------------------------------------------------------

    async def ensure_experiment(self, candidate: Candidate, split: str) -> str:
        """Ensure the Experiment exists for candidate × split; return its deterministic
        name (``opt-<run>-<label>-<split>``).

        The name — not the server-assigned id (``experiment-…``) — is what tags the trace's
        ``nemo.evaluation.name``: it is human-readable, greppable, and stable across resumes,
        and matches the Experiment's own ``name`` so the trace still joins back to it.
        """
        gname = group_name(candidate.run_id)
        gid = await self._group_id_for(candidate.run_id)
        await self._upsert_experiment(
            candidate=candidate,
            split=split,
            gname=gname,
            group_id=gid,
            agent_source=None,
        )
        return experiment_name(gname, candidate.label, split)

    async def project_candidate(
        self, candidate: Candidate, *, agent_source: Any = None, source_link: str | None = None
    ) -> None:
        gname = group_name(candidate.run_id)
        gid = await self._group_id_for(candidate.run_id)
        for split in candidate.rewards:
            # One Experiment per measured channel. Reward values are never copied here;
            # the channel's presence in the map is the signal that it was measured.
            await self._upsert_experiment(
                candidate=candidate,
                split=split,
                gname=gname,
                group_id=gid,
                agent_source=agent_source,
                source_link=source_link,
            )

    async def _upsert_experiment(
        self,
        *,
        candidate: Candidate,
        split: str,
        gname: str,
        group_id: str,
        agent_source: Any,
        source_link: str | None = None,
        status: str | None = None,
    ) -> None:
        name = experiment_name(gname, candidate.label, split)
        link = source_link or self._source_link(gname, candidate, agent_source)
        parent_id = await self._parent_experiment_id(candidate, gname)
        body = self._evaluation_request(
            name=name,
            split=split,
            group_id=group_id,
            link=link,
            candidate=candidate,
            parent_id=parent_id,
            status=status or experiment_status(candidate),
        )
        try:
            exp = (await self._client.create_evaluation(workspace=self._workspace, body=body)).data()
        except ConflictError:
            exp = (await self._client.update_evaluation(name=name, workspace=self._workspace, body=body)).data()
        if candidate.id:
            self._experiment_ids[(candidate.id, split)] = exp.id
            self._labels[candidate.id] = candidate.label

    def _evaluation_request(
        self,
        *,
        name: str,
        split: str,
        group_id: str,
        link: str,
        candidate: Candidate,
        parent_id: str | None,
        status: str,
    ) -> EvaluationCreateRequest:
        return EvaluationCreateRequest(
            name=name,
            dataset_name=self._dataset_name(split),
            dataset_version="v1",
            experiment_ids=[group_id],
            source_link=link,
            description=candidate.description,
            root_cause="",
            status=status,
            metadata=experiment_metadata(candidate, split),
            parent_evaluation_id=parent_id,
        )

    def _dataset_name(self, split: str) -> str:
        return split  # OQ-8: derive a real dataset name/version later

    def _source_link(self, gname: str, candidate: Candidate, agent_source: Any) -> str:
        if candidate.is_baseline and agent_source is not None:
            return f"{agent_source.repo_url}@{agent_source.ref}"
        return pseudo_source_link(gname, candidate.label)

    async def _existing_source_link(self, gname: str, label: str, split: str) -> str | None:
        """The source_link already stored for this candidate/split, or None.

        ``finalize`` re-upserts the winner via update without the run's
        ``agent_source``; reusing the link written during the run keeps a round-0 seed's real
        ``{repo}@{ref}`` from being clobbered with a pseudo link when no PR was opened.
        """
        try:
            exp = (
                await self._client.get_evaluation(name=experiment_name(gname, label, split), workspace=self._workspace)
            ).data()
        except NotFoundError:
            return None
        return exp.source_link

    async def _parent_experiment_id(self, candidate: Candidate, gname: str) -> str | None:
        """The ancestor's Experiment id, or None when this run cannot name it.

        ``ancestor`` is a candidate id and Experiment names are built from labels, so the
        ancestor's label has to be resolved before a name can be composed. Within a run
        the ancestor is always projected first; across a resume it may not be, and an
        omitted lineage link is the honest outcome rather than a guessed name.
        """
        if not candidate.ancestor:
            return None
        cached = self._experiment_ids.get((candidate.ancestor, "train"))
        if cached is not None:
            return cached
        ancestor_label = self._labels.get(candidate.ancestor)
        if ancestor_label is None:
            logger.debug(
                "Ancestor %r of candidate %r has not been projected; lineage link omitted",
                candidate.ancestor,
                candidate.label,
            )
            return None
        try:  # resume / ancestor created earlier this run
            exp = (
                await self._client.get_evaluation(
                    name=experiment_name(gname, ancestor_label, "train"), workspace=self._workspace
                )
            ).data()
        except NotFoundError:
            logger.debug(
                "Ancestor experiment %r not found for candidate %r; lineage link omitted",
                experiment_name(gname, ancestor_label, "train"),
                candidate.label,
            )
            return None
        self._experiment_ids[(candidate.ancestor, "train")] = exp.id
        return exp.id

    # -- Finalize -----------------------------------------------------------

    async def finalize(self, *, run_id: str, summary: str, winner: Candidate | None, pr_url: str | None = None) -> None:
        gname = group_name(run_id)
        gid = await self._group_id_for(run_id)
        grp = (await self._client.get_experiment(name=gname, workspace=self._workspace)).data()
        if grp.insight_id is not None:
            body = ExperimentUpdateRequest(
                name=gname,
                summary=summary,
                insight_id=grp.insight_id,
                metadata=grp.metadata,
            )
        else:
            body = ExperimentUpdateRequest(name=gname, summary=summary, metadata=grp.metadata)
        await self._client.update_experiment(name=gname, workspace=self._workspace, body=body)
        if winner is None:
            return
        # The winner's Experiment is always status="winner"; whether a PR was opened is
        # captured in source_link (=pr_url), so a separate "deployed" status would be
        # redundant (and "deployed" overstates a draft PR). With no PR, preserve the
        # source_link stored during the run so a round-0 winner's real {repo}@{ref} isn't
        # clobbered by the full-replace update (agent_source isn't threaded into finalize).
        for split in winner.rewards:
            link = pr_url or await self._existing_source_link(gname, winner.label, split)
            await self._upsert_experiment(
                candidate=winner,
                split=split,
                gname=gname,
                group_id=gid,
                agent_source=None,
                source_link=link,
                status="winner",
            )
