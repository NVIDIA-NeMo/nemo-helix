# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire types for skill entities (Agents service) and the references other plugins hold to them.

A *skill* is an agent skill bundle — a directory with a ``SKILL.md`` plus scripts and assets —
stored as an immutable, content-addressed archive and versioned the way evals tasks are: a head
record with tag pointers, and ``rev.<n>`` revisions addressed by ``workspace/name#<tag-or-digest>``.

Shared here (not in the agents plugin) so other plugins can reference skills by :class:`SkillRef`
and call the Agents service through the typed client without importing the agents plugin.
"""

from __future__ import annotations

from datetime import datetime
from typing import NotRequired, TypedDict

from nemo_helix_plugin.agents.types import EntityMetadata
from nemo_helix_plugin.content_hash import DIGEST_PATTERN
from nemo_helix_plugin.refs import FILESET_REF_PATTERN
from nemo_helix_plugin.revision_refs import PINNED_SUBENTITY_REF_PATTERN, SUBENTITY_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel

#: A skill's own name, from its ``SKILL.md`` frontmatter (Agent Skills convention): lowercase
#: alphanumerics separated by single hyphens. It is also the skill entity's name and the directory
#: the skill is staged under, so it also satisfies the entity-store name rules (starts with a
#: letter, 2-63 characters; see ``entity_naming.NAME_PATTERN``) and is a safe single path segment.
SKILL_NAME_PATTERN = r"^[a-z](-?[a-z0-9])+$"
SKILL_NAME_MAX_LENGTH = 63

#: A digest-pinned reference that must name its workspace. Used where a ref is persisted inside
#: another entity, so a default workspace can never re-scope it.
QUALIFIED_PINNED_REF_PATTERN = r"^[\w\-.]+/[\w\-.]+#[0-9a-f]{64}$"


class SkillRef(RootModel[str]):
    """Reference to a stored skill: ``workspace/name``, ``name``, or either with ``#<tag-or-digest>``.

    An absent fragment means ``latest``. Anything persisted as part of a frozen spec must be pinned
    (see :class:`PinnedSkillRef`).
    """

    root: str = Field(
        pattern=SUBENTITY_REF_PATTERN,
        description="Reference to a stored skill (workspace/skill-name or skill-name), optionally pinned to a "
        "revision with '#<tag-or-digest>'.",
    )


class PinnedSkillRef(RootModel[str]):
    """A skill reference pinned to a revision's full content digest."""

    root: str = Field(
        pattern=PINNED_SUBENTITY_REF_PATTERN,
        description="Reference to a skill revision pinned by its 64-char content digest.",
    )


class SkillArchiveDescriptor(BaseModel):
    """Where a revision's archive lives and what its bytes must hash to.

    ``files_hash`` is the SHA-256 of the deterministic archive bytes (see
    :mod:`nemo_helix_plugin.skill_archive`). Every consumer re-hashes what it downloads and refuses a
    mismatch. ``archive_ref`` always points into the service-owned skills fileset, never at a
    location a caller chose.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_ref: str = Field(
        pattern=FILESET_REF_PATTERN,
        description="Fileset reference to the archive object (workspace/fileset#path).",
    )
    files_hash: str = Field(pattern=DIGEST_PATTERN, description="Lowercase hex SHA-256 of the archive bytes.")
    skill_name: str = Field(
        pattern=SKILL_NAME_PATTERN,
        max_length=SKILL_NAME_MAX_LENGTH,
        description="The skill's frontmatter name; the directory it is staged under.",
    )


class SkillFrontmatter(BaseModel):
    """The searchable subset of a ``SKILL.md`` frontmatter, parsed from the archived bytes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=SKILL_NAME_PATTERN, max_length=SKILL_NAME_MAX_LENGTH, description="Skill name.")
    description: str = Field(description="When the skill should be used; what the agent reads to decide.")
    tags: list[str] = Field(
        default_factory=list,
        description="Frontmatter `metadata.tags`, for search and filtering. Unrelated to revision tags.",
    )
    extra: dict[str, JsonValue] = Field(
        default_factory=dict, description="Every other frontmatter key, verbatim, for display."
    )


class SkillSource(BaseModel):
    """Where to publish a skill from: a directory in a fileset the caller can read.

    The directory must hold ``SKILL.md`` at its root. Publishing copies it into a service-owned
    archive, so later changes to the source never affect a published revision.
    """

    model_config = ConfigDict(extra="forbid")

    fileset: str = Field(
        pattern=FILESET_REF_PATTERN,
        description="Fileset reference to the skill directory (workspace/fileset#path/to/skill).",
    )


class SkillInput(BaseModel):
    """Create/replace body for a stored skill (the entity name comes from the path).

    Create publishes revision 1; replace publishes the next revision.
    """

    model_config = ConfigDict(extra="forbid")

    source: SkillSource = Field(description="The skill directory to publish.")
    taskset: str | None = Field(
        default=None,
        pattern=QUALIFIED_PINNED_REF_PATTERN,
        description="Optional linked taskset revision, fully qualified and digest-pinned "
        "(workspace/taskset#<64-hex digest>). Tags are not accepted.",
    )
    tags: list[str] = Field(
        default_factory=list, description="Extra revision tags to point at this revision; 'latest' is automatic."
    )


class Skill(EntityMetadata):
    """A skill as of one revision: identity from the head, content from the revision.

    Reading ``/skills/{name}`` returns the ``latest`` revision; ``/skills/{name}/revisions/{revision}``
    returns the one named. Unlike the taskset DTO, this carries ``content_hash`` so a caller can mint
    a pinned ref from a tag lookup.
    """

    revision: int = Field(ge=1, description="Ordinal of the revision this content belongs to.")
    content_hash: str = Field(pattern=DIGEST_PATTERN, description="Full content digest of this revision.")
    tags: dict[str, int] = Field(
        default_factory=dict,
        description="Tag → revision-ordinal pointers. For a specific revision, only the tags pointing at it.",
    )
    frontmatter: SkillFrontmatter = Field(description="Parsed SKILL.md frontmatter.")
    archive: SkillArchiveDescriptor = Field(description="The revision's immutable archive.")
    taskset: str | None = Field(default=None, description="Linked taskset revision, digest-pinned, if any.")
    published_at: datetime | None = Field(default=None, description="When this revision was published.")


class ListSkillsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListSkillRevisionsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]


class SkillTagQueryParams(TypedDict, total=False):
    revision: NotRequired[str]


class SkillProjectQueryParams(TypedDict, total=False):
    project: NotRequired[str]
