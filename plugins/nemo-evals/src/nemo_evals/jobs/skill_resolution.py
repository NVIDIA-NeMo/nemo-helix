# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze a Harbor target's stored skills into the canonical spec at submit.

Each :class:`~nemo_helix_plugin.agents.skills.SkillRef` is read through the Agents service and pinned to
the revision it named, so the spec holds a :class:`~nemo_helix_plugin.agents.skills.ResolvedSkill`: the
digest-pinned ref and the exact archive the worker fetches and verifies. A later publish or tag move
never changes what an accepted run injects.

Submission reads with the evaluator service's credentials, which can read more than the caller, so each
skill is first checked as the caller when an authorizer is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import quote

from nemo_evals.source_authorization import SourceReadAuthorizer
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.skills import PinnedSkillRef, ResolvedSkill, SkillRef
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.revision_refs import parse_subentity_ref


def skill_read_path(workspace: str, name: str) -> str:
    """The Agents route a caller must be allowed to ``GET`` to read a stored skill."""
    return f"/apis/agents/v2/workspaces/{quote(workspace, safe='')}/skills/{quote(name, safe='')}"


async def resolve_skills(
    refs: Sequence[SkillRef],
    *,
    workspace: str,
    agents: AsyncAgentsClient,
    authorize: SourceReadAuthorizer | None,
) -> list[ResolvedSkill]:
    """Pin each skill ref to its revision and archive, in the order given.

    Args:
        refs: The target's skill refs; an unqualified name resolves in ``workspace``, and an absent
            fragment means ``latest``.
        workspace: The run's workspace.
        agents: Agents client the service reads with.
        authorize: Caller-scoped read check, run for every skill before it is read.

    Raises:
        ValueError: Two refs name the same skill, a skill or revision does not exist, or two resolved
            revisions would be staged under the same skill name.
        PermissionError: The service may not read a skill.
        Exception: Whatever ``authorize`` raises for a read the caller is denied, typically a 403.
    """
    parsed = [parse_subentity_ref(ref.root, workspace) for ref in refs]
    # Harbor stages a skill under its name and keeps the last of two with the same one, so refuse
    # before reading anything. Names, not workspace/name: two workspaces' skills would still collide.
    seen: set[str] = set()
    for _, name, _ in parsed:
        if name in seen:
            raise ValueError(f"skills: {name!r} is named more than once; a run gives its agent each skill once")
        seen.add(name)

    resolved: list[ResolvedSkill] = []
    for skill_workspace, name, revision in parsed:
        if authorize is not None:
            await authorize("GET", skill_read_path(skill_workspace, name))
        try:
            skill = (await agents.get_skill_revision(workspace=skill_workspace, name=name, revision=revision)).data()
        except NotFoundError as exc:
            raise ValueError(f"skill {skill_workspace}/{name}#{revision} does not exist") from exc
        except PermissionDeniedError as exc:
            raise PermissionError(f"access denied to skill {skill_workspace}/{name}") from exc
        resolved.append(
            ResolvedSkill(
                ref=PinnedSkillRef(f"{skill_workspace}/{name}#{skill.content_hash}"),
                archive=skill.archive,
            )
        )

    staged = [skill.archive.skill_name for skill in resolved]
    if len(set(staged)) != len(staged):
        raise ValueError(f"skills: the named skills would be staged under duplicate names {staged}")
    return resolved
