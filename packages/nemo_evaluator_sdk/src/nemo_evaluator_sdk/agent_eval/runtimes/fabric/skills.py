# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-skill injection for the Fabric agent-eval runtimes (PROTOTYPE).

An *agent skill* is a directory following the `agentskills.io <https://agentskills.io/specification>`_
spec: a folder named ``<name>/`` containing a required ``SKILL.md`` (YAML frontmatter with ``name`` +
``description``, then instructions) plus optional ``scripts/`` / ``references/`` / ``assets/``. We make
that bundle available to the harness before it runs a task so an A/B eval can score the same taskset
with and without the skill. The skill is a runtime-level knob: build one runtime with ``skill=None``
and one with ``skill=<AgentSkill>`` over the same tasks, then diff the scores.

An :class:`AgentSkill` points at a local skill directory; staging is an OS-level ``copytree`` (file
contents never pass through Python memory). The plugin resolves a platform fileset to a local
directory and constructs an ``AgentSkill`` from it — the SDK has no fileset concept of its own.

How the skill reaches the harness depends on the selected Fabric adapter, and which mode applies is
decided by *querying Fabric's own capability planner at runtime* (:func:`resolve_skill_mode` over a
``RunPlan.capability_plan``), not a hardcoded adapter list — so it tracks whatever the installed
adapters declare, including end-user adapters we don't ship:

* **Native** (:data:`SKILL_MODE_NATIVE`): the adapter advertises ``accepts: ["skills", ...]``, so
  Fabric's planner routes skills to ``harness_native``. We stage the bundle into an isolated
  ``<name>/`` dir and add it to the config's ``skills.paths``; the adapter loads it (Hermes → harness
  ``skills.external_dirs``). As of nemo-fabric 0.1.0rc3 the hermes, claude AND **codex** adapters all
  declare ``skills``, so this is the path every harness we ship currently takes.
* **Codex skills dir** (:data:`SKILL_MODE_CODEX_SKILLS_DIR`): a fallback for a codex-harness adapter
  that does *not* accept the native skills config. The Codex CLI itself discovers agentskills bundles
  from ``.agents/skills/`` in its working directory, so we place the bundle at
  ``<workspace>/.agents/skills/<name>/`` and let Codex find it — same discoverable-skill semantics as
  native (cross-harness A/B stays apples-to-apples), no Fabric adapter change needed.
  NOTE: the shipped codex adapter accepts ``skills`` today, so this branch is currently unreachable in
  production and is exercised only by the fake-backed tests. It is kept for adapters (ours or an
  end-user's) that route skills ``unsupported`` on a codex harness.

If an adapter neither routes skills natively nor is a Codex harness, :func:`resolve_skill_mode` returns
``None`` and the runtime fails fast rather than silently running a skill-free trial.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Required entry document of an agentskills bundle.
PRIMARY_SKILL_DOC = "SKILL.md"
#: Directory Codex scans (relative to its working dir) for agentskills bundles.
CODEX_SKILLS_DIR = ".agents/skills"

#: How an injected skill reaches the selected harness (resolved from Fabric's capability plan). The two
#: runtimes thread this value from :func:`resolve_skill_mode` down to :func:`install_skill` /
#: :func:`stage_skills_seed`, so a mistyped mode is a type error rather than a silent no-op.
SkillMode = Literal["native", "codex_skills_dir"]

#: Skill reaches the harness via the native Fabric ``skills`` config (adapter accepts it).
SKILL_MODE_NATIVE: SkillMode = "native"
#: Skill is placed under ``<workspace>/.agents/skills/<name>/`` for Codex to discover.
SKILL_MODE_CODEX_SKILLS_DIR: SkillMode = "codex_skills_dir"

# agentskills.io name rule: 1-64 chars, lowercase alphanumeric + single interior hyphens.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_LEN = 64

# Fabric capability-planner vocabulary (``RunPlan.capability_plan['routes']`` entries). A ``skills``
# route with target ``harness_native`` means the selected adapter declared native skills support; the
# runtime plans a probe skill path and reads these to decide the injection mode (see resolve_skill_mode).
_SKILLS_ROUTE_KIND = "skills"
_SKILLS_TARGET_NATIVE = "harness_native"
# Fabric adapter id of Codex, which self-discovers ``.agents/skills/`` rather than
# accepting the native ``skills`` config.
_CODEX_ADAPTER_ID = "nvidia.fabric.codex"


class SkillInjectionError(ValueError):
    """A skill could not be resolved, staged, or wired into the selected harness.

    Subclasses ``ValueError`` so the runtime's per-task error handling still catches it and fails
    only that task.
    """


class AgentSkill(BaseModel):
    """An agentskills.io bundle (a local directory) to make available to the agent before a task.

    ``name`` must satisfy the agentskills naming rule and is used as the staged bundle's directory name
    (spec: the name matches the directory name). ``directory`` is the local skill directory, which must
    contain a top-level ``SKILL.md``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="agentskills skill name; also the bundle directory name and provenance id.")
    directory: Path = Field(description="Local agentskills bundle directory (a SKILL.md at its root).")

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if len(value) > _MAX_NAME_LEN or not _SKILL_NAME_RE.match(value):
            raise ValueError(
                f"skill name {value!r} must be 1-{_MAX_NAME_LEN} chars, lowercase alphanumeric with "
                "single interior hyphens (agentskills.io naming rule)"
            )
        return value

    @classmethod
    def from_directory(cls, directory: str | Path, *, name: str | None = None) -> AgentSkill:
        """Build a skill from an on-disk agentskills bundle. ``name`` defaults to the directory basename."""
        root = Path(directory).expanduser().resolve()
        if not (root / PRIMARY_SKILL_DOC).is_file():
            raise SkillInjectionError(f"skill directory {str(directory)!r} has no {PRIMARY_SKILL_DOC}")
        return cls(name=name or root.name, directory=root)


class SkillProvenance(TypedDict):
    """Which skill was injected into a trial and how; stamped into trial metadata for the A/B diff.

    A plain (JSON-serializable) dict so it drops straight into trial metadata. ``None`` in that slot
    means the baseline (no skill).
    """

    name: str  #: The skill's agentskills name.
    hash: str  #: sha256 over the staged bundle — attributes a score delta to an exact skill version.
    mode: SkillMode  #: How it was injected (:data:`SKILL_MODE_NATIVE` / :data:`SKILL_MODE_CODEX_SKILLS_DIR`).
    adapter_id: str  #: The harness adapter the skill was wired into.
    location: str  #: Where the bundle was staged (absolute for native, workspace-relative for codex).


@dataclass
class SkillInstallation:
    """Result of installing a skill for one task.

    ``skill_paths`` are staged bundle roots the runtime hands to ``FabricConfig.add_skill_path`` (the
    native branch emits one; the Codex branch emits none because placement in the workspace is the
    delivery mechanism). ``provenance`` is stamped into trial metadata so the A/B comparison is
    auditable.
    """

    skill_paths: list[str]
    provenance: SkillProvenance


def native_skills_route(capability_plan: Mapping[str, object]) -> bool:
    """Whether Fabric's capability planner routed skills to the harness natively.

    ``capability_plan`` is the ``RunPlan.capability_plan`` mapping from ``Fabric.plan(...)`` planned with
    a skill path attached; its ``routes`` record each capability decision. A ``skills`` route with target
    ``harness_native`` means the selected adapter declares ``accepts: ["skills", ...]`` and Fabric hands
    the bundle to the harness itself. Any other outcome (``unsupported``, or no skills route) is False.
    """
    routes = capability_plan.get("routes")
    if not isinstance(routes, list):
        return False
    return any(
        isinstance(route, Mapping)
        and route.get("kind") == _SKILLS_ROUTE_KIND
        and route.get("target") == _SKILLS_TARGET_NATIVE
        for route in routes
    )


def resolve_skill_mode(*, capability_plan: Mapping[str, object], adapter_id: str) -> SkillMode | None:
    """Resolve how a skill would reach the selected harness, or ``None`` if it can't.

    Driven by Fabric's own capability routing (queried at runtime via ``Fabric.plan``) rather than a
    hardcoded adapter list, so it tracks whatever the installed adapters declare — including end-user
    adapters we don't ship:

    * skills route natively (:func:`native_skills_route`) -> :data:`SKILL_MODE_NATIVE`;
    * else a Codex harness (self-discovers ``.agents/skills/``) -> :data:`SKILL_MODE_CODEX_SKILLS_DIR`;
    * else ``None`` -> the runtime fails fast rather than run a skill-free trial labeled "with skill".
    """
    if native_skills_route(capability_plan):
        return SKILL_MODE_NATIVE
    if adapter_id.strip().lower() == _CODEX_ADAPTER_ID:
        return SKILL_MODE_CODEX_SKILLS_DIR
    return None


def install_skill(
    *,
    skill: AgentSkill,
    adapter_id: str,
    mode: SkillMode,
    workspace_dir: Path,
    skill_stage_dir: Path,
) -> SkillInstallation:
    """Stage ``skill`` as a ``<name>/`` bundle and wire it into the harness per ``mode``.

    Blocking file I/O — call via ``asyncio.to_thread`` from the async runtime. The bundle is always
    namespaced under ``<name>/`` so it never collides with task-seeded workspace-root files; the content
    hash is computed over the staged bytes so provenance tracks the actual skill content.

    The native branch returns the staged root for ``FabricConfig.add_skill_path``, which appends to
    whatever the base config already declares. Any preconfigured skills therefore survive injection
    without this function having to re-list them.
    """
    if mode == SKILL_MODE_NATIVE:
        skill_root = skill_stage_dir / skill.name
        _stage_bundle(skill.directory, skill_root, reserved=False)
        return SkillInstallation(
            skill_paths=[str(skill_root)],
            provenance=_provenance(skill, _hash_directory(skill_root), mode, adapter_id, str(skill_root)),
        )

    if mode == SKILL_MODE_CODEX_SKILLS_DIR:
        skill_root = workspace_dir / CODEX_SKILLS_DIR / skill.name
        _stage_bundle(skill.directory, skill_root, reserved=True)
        location = (Path(CODEX_SKILLS_DIR) / skill.name).as_posix()
        return SkillInstallation(
            skill_paths=[],
            provenance=_provenance(skill, _hash_directory(skill_root), mode, adapter_id, location),
        )

    raise SkillInjectionError(f"unknown skill injection mode {mode!r} for adapter {adapter_id!r}")


@dataclass
class SkillsInstallation:
    """Result of installing several skills for one task (see :func:`install_skills`).

    ``skill_paths`` is every staged native bundle root, in the given order, for the runtime to feed to
    ``FabricConfig.add_skill_path``; the Codex branch emits none because workspace placement is the
    delivery mechanism. ``provenances`` is one entry per skill, in the given order, stamped into trial
    metadata so a multi-skill A/B comparison is auditable.
    """

    skill_paths: list[str]
    provenances: list[SkillProvenance]


def require_unique_skill_names(skills: Sequence[AgentSkill]) -> None:
    """Raise if two skills share a name — their ``<name>/`` bundles would collide when staged.

    Each skill stages into its own ``<name>/`` directory (native stage dir or ``.agents/skills/``), so a
    repeated name would clobber (or fail to stage over) an earlier bundle. Checked up front so a
    misconfigured runtime fails before any task runs, not mid-stage on the second collision.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for skill in skills:
        if skill.name in seen and skill.name not in duplicates:
            duplicates.append(skill.name)
        seen.add(skill.name)
    if duplicates:
        raise SkillInjectionError(
            f"duplicate skill name(s) {duplicates}: each skill stages to its own '<name>/' bundle, so "
            "skill names must be unique within one runtime"
        )


@dataclass(frozen=True)
class SkillSet:
    """Immutable, name-validated collection of :class:`AgentSkill`\\s held by a Fabric runtime.

    Construction validates that skill names are unique; :meth:`with_skills` and :meth:`with_skill`
    each return a new ``SkillSet`` without modifying ``self``.
    """

    skills: tuple[AgentSkill, ...] = ()

    def __post_init__(self) -> None:
        require_unique_skill_names(self.skills)

    def with_skills(self, skills: Sequence[AgentSkill]) -> SkillSet:
        """Return a new ``SkillSet`` with ``skills`` appended; ``self`` is not modified."""
        return SkillSet((*self.skills, *skills))

    def with_skill(self, skill: AgentSkill) -> SkillSet:
        """Return a new ``SkillSet`` with ``skill`` appended; ``self`` is not modified."""
        return self.with_skills([skill])


def install_skills(
    *,
    skills: Sequence[AgentSkill],
    adapter_id: str,
    mode: SkillMode,
    workspace_dir: Path,
    skill_stage_dir: Path,
) -> SkillsInstallation:
    """Stage every skill in ``skills`` for one task and wire them all into the harness per ``mode``.

    Loops :func:`install_skill` — each skill stages into its own namespaced ``<name>/`` bundle — and
    collects the staged roots for the native mode. ``FabricConfig.add_skill_path`` appends and
    de-duplicates, so every injected skill lands alongside whatever the base config already declared,
    with no re-listing. Skill names must be unique (their ``<name>/`` bundles would otherwise collide).
    Blocking file I/O — call via ``asyncio.to_thread`` from the async runtime.

    Installation is all-or-nothing: if any skill fails to stage, the bundles already staged in this call
    are rolled back before the error propagates, so a partial skill set never lingers on disk (the caller
    raises before it ever sees provenances, so it cannot clean up itself). Only bundles this call staged
    are removed, so a reserved-path collision can never delete a pre-existing task-seeded file.
    """
    require_unique_skill_names(skills)
    provenances: list[SkillProvenance] = []
    staged_roots: list[Path] = []
    try:
        for skill in skills:
            # Register the target BEFORE staging: install_skill can raise after it has already written
            # files (a copytree failing partway, an unreadable file while hashing), and a root recorded
            # only on success would leave that partial bundle behind. A target that already exists is
            # never registered — in codex mode that is a task-seeded file install_skill refuses to
            # clobber, and rolling it back would delete task input this call did not create.
            stage_root = _skill_stage_root(skill, mode, workspace_dir, skill_stage_dir)
            if not stage_root.exists():
                staged_roots.append(stage_root)
            provenance = install_skill(
                skill=skill,
                adapter_id=adapter_id,
                mode=mode,
                workspace_dir=workspace_dir,
                skill_stage_dir=skill_stage_dir,
            ).provenance
            provenances.append(provenance)
    except Exception:
        for root in staged_roots:
            shutil.rmtree(root, ignore_errors=True)
        raise

    skill_paths: list[str] = []
    if mode == SKILL_MODE_NATIVE:
        # Each staged bundle root, order-preserved and de-duplicated (a native provenance's
        # ``location`` is its absolute staged skill root).
        skill_paths = list(dict.fromkeys(prov["location"] for prov in provenances))
    return SkillsInstallation(skill_paths=skill_paths, provenances=provenances)


def _skill_stage_root(skill: AgentSkill, mode: SkillMode, workspace_dir: Path, skill_stage_dir: Path) -> Path:
    """Absolute on-disk root :func:`install_skill` would stage ``skill`` into, computed before staging.

    Mirrors install_skill's per-mode placement so :func:`install_skills` can register a rollback target
    up front (an unknown mode raises there, not here; the returned path is simply never created, and
    rolling back a path that does not exist is a no-op)."""
    if mode == SKILL_MODE_NATIVE:
        return skill_stage_dir / skill.name
    return workspace_dir / CODEX_SKILLS_DIR / skill.name


def _render_skill_seed(
    *, skill: AgentSkill, adapter_id: str, mode: SkillMode, workspace_dir: str, skills_dir: str
) -> tuple[dict[str, str], SkillProvenance]:
    """Render one skill bundle into an in-sandbox ``{path: text}`` seed map + its provenance.

    The per-skill core of :func:`stage_skills_seed` (the containerized counterpart of :func:`install_skill`,
    which ``copytree``\\ s onto host disk): the container has no host workspace, so the bundle is read into
    memory as UTF-8 text and keyed at the harness's in-sandbox discovery path — native: ``<skills_dir>/
    <name>/``; codex: ``<workspace_dir>/.agents/skills/<name>/``. The content hash is over the source bundle
    (matching :func:`install_skill`). The caller merges these into one seed set and, for native mode, a
    single ``skills`` overlay — so no per-skill overlay is emitted here.
    """
    bundle = _read_text_bundle(skill.directory)
    skill_hash = _hash_directory(skill.directory)
    if mode == SKILL_MODE_NATIVE:
        skill_root = f"{skills_dir.rstrip('/')}/{skill.name}"
        files = {f"{skill_root}/{rel}": text for rel, text in bundle.items()}
        return files, _provenance(skill, skill_hash, mode, adapter_id, skill_root)

    if mode == SKILL_MODE_CODEX_SKILLS_DIR:
        skill_root = f"{workspace_dir.rstrip('/')}/{CODEX_SKILLS_DIR}/{skill.name}"
        files = {f"{skill_root}/{rel}": text for rel, text in bundle.items()}
        location = f"{CODEX_SKILLS_DIR}/{skill.name}"
        return files, _provenance(skill, skill_hash, mode, adapter_id, location)

    raise SkillInjectionError(f"unknown skill injection mode {mode!r} for adapter {adapter_id!r}")


def _read_text_bundle(directory: Path) -> dict[str, str]:
    """Read an agentskills bundle into a ``{posix_relpath: text}`` map (requires a top-level ``SKILL.md``).

    Every file is decoded as UTF-8: the containerized seed set (``SandboxSpec.files``) is text-only, so a
    binary file (e.g. an image under ``assets/``) raises here rather than silently corrupting the staged
    bundle — the host :func:`install_skill` path (OS-level ``copytree``) handles binary bundles instead.
    """
    src = directory.expanduser()
    if not (src / PRIMARY_SKILL_DOC).is_file():
        raise SkillInjectionError(f"skill directory {str(directory)!r} has no {PRIMARY_SKILL_DOC}")
    bundle: dict[str, str] = {}
    for path in sorted(candidate for candidate in src.rglob("*") if candidate.is_file()):
        rel = path.relative_to(src).as_posix()
        try:
            bundle[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillInjectionError(
                f"skill file {rel!r} is not UTF-8 text; containerized skill injection (via the sandbox "
                "seed set) supports text bundles only"
            ) from exc
    return bundle


@dataclass
class SkillsSeed:
    """Result of rendering several skills into one sandbox seed set (see :func:`stage_skills_seed`).

    The plural, containerized sibling of :class:`SkillsInstallation`:

    * ``files`` — the merged ``{absolute_in_sandbox_path: text}`` seed map for every staged bundle.
    * ``skill_paths`` — every staged native bundle root, in order, for the runtime to merge into the
      composed config's ``skills.paths``; the codex branch emits none.
    * ``provenances`` — one entry per skill, in the given order, for the multi-skill A/B trial metadata.
    """

    files: dict[str, str]
    skill_paths: list[str]
    provenances: list[SkillProvenance]


def stage_skills_seed(
    *,
    skills: Sequence[AgentSkill],
    adapter_id: str,
    mode: SkillMode,
    workspace_dir: str,
    skills_dir: str,
) -> SkillsSeed:
    """Render every skill in ``skills`` into one sandbox seed set for the container runtime.

    The plural, containerized sibling of :func:`install_skills`: renders each bundle (via
    :func:`_render_skill_seed`) under its own ``<name>/`` at the harness's in-sandbox discovery path and
    collects the native in-sandbox roots. The caller merges those into the composed config's
    ``skills.paths`` alongside whatever it already declared, so nothing has to be re-listed here. Skill
    names must be unique — their ``<name>/`` bundles would otherwise collide. No on-disk rollback is
    needed (unlike :func:`install_skills`): the seed set is an in-memory map, so a failure to render any
    skill just discards the accumulated map and raises, leaving nothing staged.
    """
    require_unique_skill_names(skills)
    files: dict[str, str] = {}
    provenances: list[SkillProvenance] = []
    for skill in skills:
        rendered, provenance = _render_skill_seed(
            skill=skill, adapter_id=adapter_id, mode=mode, workspace_dir=workspace_dir, skills_dir=skills_dir
        )
        files.update(rendered)
        provenances.append(provenance)

    skill_paths: list[str] = []
    if mode == SKILL_MODE_NATIVE:
        # Each staged bundle (a native provenance's ``location`` is its absolute in-sandbox skill
        # root), order-preserved and de-duplicated.
        skill_paths = list(dict.fromkeys(prov["location"] for prov in provenances))
    return SkillsSeed(files=files, skill_paths=skill_paths, provenances=provenances)


def _stage_bundle(directory: Path, skill_root: Path, *, reserved: bool) -> None:
    """Stage the skill ``directory`` as an *exact* copy at ``skill_root`` (the ``<name>/`` bundle dir).

    The staged bundle must reflect exactly the supplied directory, so provenance and behaviour track the
    real content. ``reserved`` picks the collision policy for the destination:

    * ``reserved=False`` — the evaluator-owned native stage dir: recreate it, so a reused run id can't
      leave a file that was since removed from the source bundle surviving in the stage.
    * ``reserved=True`` — the Codex workspace path (``.agents/skills/<name>``): refuse to clobber
      pre-existing content there, since it can only be a task-seeded file colliding with the reserved
      skill path.
    """
    src = directory.expanduser()
    if not (src / PRIMARY_SKILL_DOC).is_file():
        raise SkillInjectionError(f"skill directory {str(directory)!r} has no {PRIMARY_SKILL_DOC}")
    if skill_root.exists():
        if reserved:
            raise SkillInjectionError(
                f"cannot stage skill into reserved path {str(skill_root)!r}: it already exists "
                "(a task-seeded file collides with the injected skill bundle)"
            )
        shutil.rmtree(skill_root)  # evaluator-owned: recreate so the stage is an exact copy
    skill_root.parent.mkdir(parents=True, exist_ok=True)
    # OS-level copy — file contents never pass through Python memory.
    shutil.copytree(src, skill_root)


def _provenance(skill: AgentSkill, skill_hash: str, mode: SkillMode, adapter_id: str, location: str) -> SkillProvenance:
    return {
        "name": skill.name,
        "hash": skill_hash,
        "mode": mode,
        "adapter_id": adapter_id,
        "location": location,
    }


def _hash_directory(directory: Path) -> str:
    """Stable sha256 over a directory's file tree (sorted relpath + contents)."""
    digest = hashlib.sha256()
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
