# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent installer registry.

To add a new agent:
1. Create a module in agents/ implementing BaseAgentInstaller
2. Add an entry to _INSTALLERS below
"""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

import yaml
from nemo_helix_ext.cli.commands.skills.agents.claude import ClaudeInstaller
from nemo_helix_ext.cli.commands.skills.agents.codex import CodexInstaller
from nemo_helix_ext.cli.commands.skills.agents.cursor import CursorInstaller
from nemo_helix_ext.cli.commands.skills.agents.opencode import OpenCodeInstaller
from nemo_helix_ext.cli.commands.skills.base import Skill, validate_skill_name
from nemo_helix_ext.cli.commands.skills.installer import BaseAgentInstaller

logger = logging.getLogger(__name__)

# When two ``nemo.skills`` entry points share a name and point at equivalent
# content but different on-disk paths (e.g. the editable source tree and the
# vendored SDK in this monorepo), this ranking decides which distribution wins.
# Lower rank = preferred. Distributions not listed fall back to alphabetical
# order via :func:`_distribution_preference`.
_PROVIDER_PREFERENCE_RANKS: dict[str, int] = {
    "nemo-helix-ext": 0,
    "nemo-helix-sdk": 1,
    "nemo-helix": 2,
}

_INSTALLERS: dict[str, BaseAgentInstaller] = {
    "claude": ClaudeInstaller(),
    "cursor": CursorInstaller(),
    "codex": CodexInstaller(),
    "opencode": OpenCodeInstaller(),
}


class UnsupportedAgentError(Exception):
    """Raised when an unknown agent name is requested."""

    def __init__(self, agent_name: str):
        available = ", ".join(sorted(_INSTALLERS.keys()))
        super().__init__(f"Unsupported agent: '{agent_name}'. Available agents: {available}")


class DuplicateSkillError(ValueError):
    """Raised when multiple skill sources declare the same skill name."""


def get_installer(agent_name: str) -> BaseAgentInstaller:
    """Get the installer for the given agent name."""
    if agent_name not in _INSTALLERS:
        raise UnsupportedAgentError(agent_name)
    return _INSTALLERS[agent_name]


def list_agent_names() -> list[str]:
    """Return sorted list of supported agent names."""
    return sorted(_INSTALLERS.keys())


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file. Returns (metadata, body)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    metadata = yaml.safe_load(text[4:end]) or {}
    body = text[end + 5 :]  # skip past "\n---\n"
    return metadata, body


@dataclass(frozen=True)
class SkillProvider:
    """A single ``nemo.skills`` entry-point resolved to its on-disk skills root.

    ``name`` is the entry-point name (e.g. ``"agents"``, ``"platform"``).
    ``dist_name`` is the distribution / wheel name that registered the entry
    point (e.g. ``"nemo-agents-plugin"``, ``"nemo-helix-ext"``). ``dist_name``
    is what end users recognise — they pip-installed that package — so it
    powers the user-facing ``Source`` column.
    """

    name: str
    path: Path
    dist_name: str | None


def _load_skill(entry: Path, source_plugin: str | None = None, source_dist: str | None = None) -> Skill:
    skill_file = entry / "SKILL.md"
    raw = skill_file.read_text(encoding="utf-8")
    try:
        metadata, body = _parse_frontmatter(raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid frontmatter in {skill_file}: {e}") from e
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid frontmatter in {skill_file}: expected a mapping, got {type(metadata).__name__}")
    preconditions = metadata.get("preconditions", [])
    if preconditions is None:
        preconditions = []
    if isinstance(preconditions, str):
        preconditions = [preconditions]
    if not isinstance(preconditions, list) or not all(isinstance(item, str) for item in preconditions):
        raise ValueError(f"Invalid frontmatter in {skill_file}: preconditions must be a list of strings")
    skill_name = metadata.get("name", entry.name)
    if not isinstance(skill_name, str):
        raise ValueError(f"Invalid frontmatter in {skill_file}: name must be a string")
    validate_skill_name(skill_name)
    return Skill(
        name=skill_name,
        description=metadata.get("description", ""),
        version=str(metadata.get("version", "0.1")),
        content=body,
        raw=raw,
        preconditions=preconditions,
        source_dir=entry,
        source_plugin=source_plugin,
        source_dist=source_dist,
    )


def _load_skills_from_root(
    root: Path,
    source_plugin: str | None = None,
    source_dist: str | None = None,
) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.exists():
            continue
        skill = _load_skill(entry, source_plugin=source_plugin, source_dist=source_dist)
        skills[skill.name] = skill
    return skills


def _load_skills_from_provider(provider: SkillProvider) -> dict[str, Skill]:
    """Load all skills from a single provider's skills directory."""
    skills: dict[str, Skill] = {}
    for entry in sorted(provider.path.iterdir()):
        if not entry.is_dir() or not (entry / "SKILL.md").exists():
            continue
        try:
            skill = _load_skill(entry, source_plugin=provider.name, source_dist=provider.dist_name)
        except ValueError:
            logger.warning(
                "Skipping invalid skill from provider %r in %s",
                provider.name,
                entry,
                exc_info=True,
            )
            continue
        skills[skill.name] = skill
    return skills


def _raw_entry_points(group: str) -> Iterable[EntryPoint]:
    """Return every entry point registered under *group* without deduplication.

    Tests patch this single seam to inject fake entry points; production calls
    straight through to :func:`importlib.metadata.entry_points`. Going around
    ``nemo_helix_plugin.discovery.discover_entry_points`` is intentional — that helper
    collapses entries with duplicate ``name`` via a dict-comprehension, which
    is exactly the property we need to override here so we can join duplicates
    deliberately instead of letting last-wins iteration order decide.
    """
    return entry_points(group=group)


def _resolve_provider_candidate(ep: EntryPoint) -> Path | None:
    """Load and validate a single ``nemo.skills`` entry-point candidate.

    Returns the resolved skills-root path on success, or ``None`` (with a
    warning logged) for every failure mode the previous implementation
    handled: factory load fails, factory call fails, return value isn't a
    ``Path``, path doesn't exist, path isn't a directory. Returning ``None``
    keeps a sibling candidate viable — one broken provider must not poison
    the others.
    """
    provider_name = ep.name
    try:
        skills_dir_factory = ep.load()
    except Exception:
        logger.warning(
            "Failed to load 'nemo.skills' entry point %r (%s) — skipping",
            provider_name,
            ep.value,
            exc_info=True,
        )
        return None
    try:
        skills_root = skills_dir_factory()
    except Exception:
        logger.warning("Failed to resolve skills directory for provider %r", provider_name, exc_info=True)
        return None
    if not isinstance(skills_root, Path):
        logger.warning(
            "Skipping provider %r skills: expected Path from nemo.skills entry point, got %s",
            provider_name,
            type(skills_root).__name__,
        )
        return None
    if not skills_root.exists():
        logger.warning("Skipping provider %r skills: path does not exist: %s", provider_name, skills_root)
        return None
    if not skills_root.is_dir():
        logger.warning("Skipping provider %r skills: path is not a directory: %s", provider_name, skills_root)
        return None
    return skills_root


def _content_signature(skills_root: Path) -> frozenset[tuple[str, str]]:
    """Build a stable fingerprint of the SKILL.md files under *skills_root*.

    The signature is ``frozenset[(skill_dir_name, sha256(SKILL.md))]``. Two
    providers whose signatures match ship the same skills with byte-identical
    content (the common case: editable source + its vendored mirror) and can
    safely collapse to one. Signatures that differ flag a real drift between
    source and vendored copies, which we surface as a hard error.
    """
    signature: set[tuple[str, str]] = set()
    for entry in skills_root.iterdir():
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.exists():
            continue
        digest = hashlib.sha256(skill_file.read_bytes()).hexdigest()
        signature.add((entry.name, digest))
    return frozenset(signature)


def _distribution_preference(dist_name: str | None) -> tuple[int, str]:
    """Sort key for picking a winner when same-name candidates have equal content.

    Known platform distributions get explicit ranks (see
    :data:`_PROVIDER_PREFERENCE_RANKS`); everything else falls back to
    alphabetical order so the choice stays deterministic for unknown plugins
    too. Distributions without an attached ``dist_name`` sort last.
    """
    if dist_name is None:
        return (len(_PROVIDER_PREFERENCE_RANKS) + 1, "")
    explicit = _PROVIDER_PREFERENCE_RANKS.get(dist_name)
    if explicit is not None:
        return (explicit, dist_name)
    return (len(_PROVIDER_PREFERENCE_RANKS), dist_name)


def _join_same_name_candidates(name: str, candidates: list[SkillProvider]) -> SkillProvider:
    """Reduce N candidates that all claim entry-point *name* to a single provider.

    Strategy:
    1. Collapse candidates whose resolved paths are equal (after ``.resolve()``).
       This handles the trivial case where two entry points point at the same
       on-disk directory via namespace-package merging.
    2. If multiple distinct paths remain, compare their SKILL.md content
       signatures:
       a. Equal signatures → providers mirror each other; pick the preferred
          distribution via :func:`_distribution_preference`.
       b. One signature is a strict superset of all the others → the smaller
          providers ship a subset of the same content (typically a vendored
          copy that hasn't picked up newly-added source skills yet); pick the
          superset so users see every skill the source defines.
       c. Otherwise the content has genuinely diverged (an overlapping skill
          differs, or each side has skills the other lacks) — that's a real
          packaging bug, so raise :class:`DuplicateSkillError` naming the
          paths and pointing at ``make vendor`` as the usual fix.
    """
    if len(candidates) == 1:
        return candidates[0]

    by_resolved_path: dict[Path, list[SkillProvider]] = defaultdict(list)
    for candidate in candidates:
        by_resolved_path[candidate.path.resolve()].append(candidate)

    if len(by_resolved_path) == 1:
        # All candidates hit the same directory; pick one deterministically.
        return min(candidates, key=lambda c: _distribution_preference(c.dist_name))

    signatures = {path: _content_signature(path) for path in by_resolved_path}
    canonical = next(iter(signatures.values()))
    if all(sig == canonical for sig in signatures.values()):
        return min(candidates, key=lambda c: _distribution_preference(c.dist_name))

    # Pick the candidate whose content is a superset of every other candidate's.
    # If one exists it's unambiguously the most complete view of this provider's
    # skills and we prefer it (subset semantics override the ext-rank tiebreaker).
    superset_path = _strict_superset_path(signatures)
    if superset_path is not None:
        winners = [c for c in candidates if c.path.resolve() == superset_path]
        return min(winners, key=lambda c: _distribution_preference(c.dist_name))

    paths_str = ", ".join(sorted(str(p) for p in signatures))
    raise DuplicateSkillError(
        f"Conflicting 'nemo.skills' entry points named {name!r} resolve to directories "
        f"with different content: {paths_str}. This usually means a vendored copy has "
        "drifted from its source; run `make vendor` (or the equivalent for the diverging "
        "distribution) to resync."
    )


def _strict_superset_path(signatures: dict[Path, frozenset[tuple[str, str]]]) -> Path | None:
    """Return the path whose signature is a superset of every other signature,
    or ``None`` if no such single dominant candidate exists.

    "Superset" here means *every* ``(skill_name, content_hash)`` pair in the
    other candidates appears in this one — so the smaller candidates are
    strict subsets with no conflicting hashes for any shared skill name. This
    is exactly the "source has new skills not yet vendored" case.
    """
    paths = list(signatures)
    for candidate_path, candidate_sig in signatures.items():
        if all(signatures[other] <= candidate_sig for other in paths if other != candidate_path):
            return candidate_path
    return None


def _discover_skill_providers() -> dict[str, SkillProvider]:
    """Resolve all ``nemo.skills`` entry points to ``{provider_name: SkillProvider}``.

    Walks every registered entry point in the group (not the deduplicated map
    that :func:`nemo_helix_plugin.discovery.discover_entry_points` returns) so that
    multiple distributions registering the same provider name — common in this
    monorepo where ``nemo-helix-ext`` (source), ``nemo-helix-sdk``
    (vendored mirror), and ``nemo-helix`` (bundling wrapper) all register
    ``platform`` — are joined deliberately by :func:`_join_same_name_candidates`
    rather than being resolved by undefined enumeration order.

    Per-candidate validation (factory loads, returns ``Path``, path exists, is
    a directory) is delegated to :func:`_resolve_provider_candidate` so a
    single broken candidate doesn't disqualify its valid siblings.

    The plugin allowlist (``NEMO_PLUGIN_SKILLS_ALLOWLIST`` /
    ``NEMO_PLUGIN_ALLOWLIST``) is honoured: the set of allowed names is taken
    from ``nemo_helix_plugin.discovery.discover_entry_points`` and used to filter
    candidates before resolution.
    """
    candidates_by_name: dict[str, list[SkillProvider]] = defaultdict(list)

    try:
        from nemo_helix_plugin.discovery import discover_entry_points
    except ImportError:
        logger.warning("nemo-helix-plugin is unavailable; loading only bundled platform skills", exc_info=True)
    else:
        allowed_names = set(discover_entry_points("nemo.skills").keys())
        for ep in _raw_entry_points("nemo.skills"):
            if ep.name not in allowed_names:
                continue
            skills_root = _resolve_provider_candidate(ep)
            if skills_root is None:
                continue
            dist_name = ep.dist.name if ep.dist is not None else None
            candidates_by_name[ep.name].append(SkillProvider(name=ep.name, path=skills_root, dist_name=dist_name))

    if "platform" not in candidates_by_name:
        from nemo_helix_ext.skills import skills_dir

        candidates_by_name["platform"].append(
            SkillProvider(name="platform", path=skills_dir(), dist_name="nemo-helix-ext")
        )

    return {name: _join_same_name_candidates(name, candidates) for name, candidates in candidates_by_name.items()}


@lru_cache(maxsize=1)
def _load_skills_cached() -> dict[str, Skill]:
    """Discover and load all skills via the ``nemo.skills`` entry-point mechanism.

    The platform's own bundled skills are exposed through ``nemo_helix_ext.skills``
    using the same entry point that third-party plugins use, so this loader has a
    single uniform path. Two providers declaring the same skill name is a hard
    error (``DuplicateSkillError``) so accidental shadowing fails loudly.
    """
    all_skills: dict[str, Skill] = {}
    for provider in _discover_skill_providers().values():
        for skill in _load_skills_from_provider(provider).values():
            existing = all_skills.get(skill.name)
            if existing is not None:
                raise DuplicateSkillError(
                    f"Duplicate skill '{skill.name}' found in provider {provider.name}: "
                    f"{skill.source_dir}. Already defined in {existing.source_dir}."
                )
            all_skills[skill.name] = skill
    return dict(sorted(all_skills.items()))


def load_skills() -> dict[str, Skill]:
    """Load all CLI-visible skills via the ``nemo.skills`` entry-point mechanism."""
    return dict(_load_skills_cached())


def clear_cache() -> None:
    """Clear cached skill registry data so the next ``load_skills`` re-reads sources."""
    _load_skills_cached.cache_clear()
