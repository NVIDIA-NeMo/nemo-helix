# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the skill / agent registry."""

# ruff: noqa: I001 - vendoring rewrites nemo_helix_ext imports to nemo_helix,
# which changes the SDK package's preferred import ordering.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
from nemo_helix_ext.cli.commands.skills.base import Skill
from nemo_helix_ext.cli.commands.skills.registry import (
    DuplicateSkillError,
    SkillProvider,
    UnsupportedAgentError,
    _load_skill,
    _load_skills_cached,
    get_installer,
    list_agent_names,
    load_skills,
)
from nemo_helix_ext.skills import skills_dir as platform_skills_dir


def _example_plugin_skills_root() -> Path:
    relative_path = Path("plugins/example-plugin/src/nemo_example_plugin/skills")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find example plugin skills fixture: {relative_path}")


@dataclass
class _FakeDist:
    name: str


@dataclass
class _FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint`` used in tests
    that exercise ``_discover_skill_providers`` validation branches."""

    name: str
    value: str
    dist: _FakeDist | None
    _loader: Callable[[], object]

    def load(self) -> object:
        return self._loader()


def _platform_provider() -> SkillProvider:
    """The real platform provider, suitable for inclusion alongside test fakes."""
    return SkillProvider(name="platform", path=platform_skills_dir(), dist_name="nemo-helix-ext")


@pytest.fixture(autouse=True)
def clear_skill_cache():
    """Reset both registry caches around every test.

    Beyond ``_load_skills_cached``, we also drop the
    ``nemo_helix_plugin.discovery.discover_entry_points`` cache, because tests
    in this module monkeypatch that symbol and any earlier real call would have
    pinned the original cache with the un-patched result (or vice-versa).
    Clearing it on both ends keeps tests order-independent.
    """
    from nemo_helix_plugin.discovery import discover_entry_points as _discover_eps

    _load_skills_cached.cache_clear()
    _discover_eps.cache_clear()
    yield
    _load_skills_cached.cache_clear()
    _discover_eps.cache_clear()


def test_list_agent_names():
    names = list_agent_names()
    assert "claude" in names
    assert "cursor" in names
    assert "codex" in names
    assert "opencode" in names


def test_get_installer_claude():
    installer = get_installer("claude")
    assert installer.name == "claude"


def test_get_installer_cursor():
    installer = get_installer("cursor")
    assert installer.name == "cursor"


def test_get_installer_codex():
    installer = get_installer("codex")
    assert installer.name == "codex"


def test_get_installer_opencode():
    installer = get_installer("opencode")
    assert installer.name == "opencode"


def test_get_installer_unknown_raises():
    with pytest.raises(UnsupportedAgentError, match="unknown"):
        get_installer("unknown")


def test_unsupported_agent_error_lists_available():
    with pytest.raises(UnsupportedAgentError) as exc_info:
        get_installer("unknown")
    error_msg = str(exc_info.value)
    assert "claude" in error_msg
    assert "cursor" in error_msg


def test_load_skills_includes_platform_skills():
    """The platform's bundled skills must always be discoverable via the entry point."""
    skills = load_skills()
    # Canonical platform skills must remain available — adding new ones is free.
    assert {"inference"} <= skills.keys()


def test_load_skills_includes_platform_skills_without_entry_point_metadata(monkeypatch):
    """Bundled platform skills remain available even if entry-point metadata is absent."""
    monkeypatch.setattr("nemo_helix_plugin.discovery.discover_entry_points", lambda _group: {})
    monkeypatch.setattr("nemo_helix_ext.cli.commands.skills.registry._raw_entry_points", lambda _group: ())

    skills = load_skills()

    assert {"inference"} <= skills.keys()
    assert skills["inference"].source_dist == "nemo-helix-ext"


def test_load_skills_includes_example_plugin_skills(monkeypatch):
    """Adding a provider on top of the platform should expose its skills too."""
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._discover_skill_providers",
        lambda: {
            "platform": _platform_provider(),
            "example": SkillProvider(
                name="example",
                path=_example_plugin_skills_root(),
                dist_name="nemo-example-plugin",
            ),
        },
    )
    skills = load_skills()
    assert "example-hello" in skills
    assert "example-debug" in skills
    assert skills["example-hello"].source_dir == _example_plugin_skills_root() / "example-hello"
    # The distribution name is threaded through so the CLI can render it.
    assert skills["example-hello"].source_dist == "nemo-example-plugin"
    # Platform skills are still there.
    assert {"inference"} <= skills.keys()


def test_load_skills_skips_invalid_provider_frontmatter(monkeypatch, tmp_path: Path):
    root = tmp_path / "provider-skills"
    valid = root / "valid"
    invalid = root / "invalid"
    valid.mkdir(parents=True)
    invalid.mkdir(parents=True)
    (valid / "SKILL.md").write_text("---\nname: valid-provider-skill\ndescription: ok\n---\n# Valid\n")
    (invalid / "SKILL.md").write_text("---\nname: broken\ndescription: [\n---\n# Broken\n")
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._discover_skill_providers",
        lambda: {"example": SkillProvider(name="example", path=root, dist_name="example-dist")},
    )

    with patch("nemo_helix_ext.cli.commands.skills.registry.logger.warning") as mock_warning:
        skills = load_skills()

    assert "valid-provider-skill" in skills
    assert "broken" not in skills
    mock_warning.assert_called_once()
    assert "Skipping invalid skill from provider" in mock_warning.call_args.args[0]


def test_load_skill_rejects_non_mapping_frontmatter(tmp_path: Path):
    invalid = tmp_path / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text("---\n- not\n- a\n- mapping\n---\n# Broken\n")

    with pytest.raises(ValueError, match="expected a mapping"):
        _load_skill(invalid)


def test_load_skills_skips_non_mapping_frontmatter(monkeypatch, tmp_path: Path):
    root = tmp_path / "provider-skills"
    valid = root / "valid"
    invalid = root / "invalid"
    valid.mkdir(parents=True)
    invalid.mkdir(parents=True)
    (valid / "SKILL.md").write_text("---\nname: valid-provider-skill\ndescription: ok\n---\n# Valid\n")
    (invalid / "SKILL.md").write_text("---\n- not\n- a\n- mapping\n---\n# Broken\n")
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._discover_skill_providers",
        lambda: {"example": SkillProvider(name="example", path=root, dist_name="example-dist")},
    )

    with patch("nemo_helix_ext.cli.commands.skills.registry.logger.warning") as mock_warning:
        skills = load_skills()

    assert "valid-provider-skill" in skills
    assert "invalid" not in skills
    mock_warning.assert_called_once()
    assert "Skipping invalid skill from provider" in mock_warning.call_args.args[0]


def test_load_skills_duplicate_names_raise(monkeypatch, tmp_path: Path):
    """A non-platform provider that shadows a platform skill name must fail loudly."""
    root = tmp_path / "provider-skills"
    duplicate = root / "duplicate"
    duplicate.mkdir(parents=True)
    (duplicate / "SKILL.md").write_text(
        "---\nname: inference\ndescription: collides with the platform skill\n---\n# Duplicate\n"
    )
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._discover_skill_providers",
        lambda: {
            "platform": _platform_provider(),
            "example": SkillProvider(name="example", path=root, dist_name="example-dist"),
        },
    )

    with pytest.raises(DuplicateSkillError, match="Duplicate skill 'inference'"):
        load_skills()


def test_load_skills_skips_missing_provider_path_with_warning(monkeypatch, tmp_path: Path):
    """A provider whose entry-point points at a missing dir is skipped with a warning,
    and other (well-formed) providers are still loaded.

    Patches both seams ``_raw_entry_points`` (for the duplicate-aware walk) and
    ``discover_entry_points`` (for the allowlist gate) so the test exercises
    the validation branches inside ``_discover_skill_providers`` itself
    (path-exists, is-dir, isinstance Path) rather than mocking them away.
    """
    nonexistent = tmp_path / "does-not-exist"
    fake_eps = {
        "platform": _FakeEntryPoint(
            name="platform",
            value="nemo_helix_ext.skills:skills_dir",
            dist=_FakeDist(name="nemo-helix-ext"),
            _loader=lambda: platform_skills_dir,
        ),
        "ghost": _FakeEntryPoint(
            name="ghost",
            value="ghost.skills:skills_dir",
            dist=_FakeDist(name="ghost-plugin"),
            _loader=lambda: lambda: nonexistent,
        ),
    }
    monkeypatch.setattr("nemo_helix_plugin.discovery.discover_entry_points", lambda _group: fake_eps)
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._raw_entry_points",
        lambda _group: list(fake_eps.values()),
    )

    with patch("nemo_helix_ext.cli.commands.skills.registry.logger.warning") as mock_warning:
        skills = load_skills()

    # The platform's own skills must still be present even when another provider is broken.
    assert {"inference"} <= skills.keys()
    # A warning fires specifically for the missing-path ghost provider.
    warning_messages = [call.args[0] for call in mock_warning.call_args_list]
    assert any("path does not exist" in msg for msg in warning_messages)


def _write_skill_dir(root: Path, name: str, body: str = "# body\n") -> Path:
    """Create ``<root>/<name>/SKILL.md`` and return its parent directory."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: t\n---\n{body}")
    return skill_dir


def test_load_skill_rejects_path_like_frontmatter_name(tmp_path: Path):
    skill_dir = tmp_path / "safe-dir"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: ../escape\ndescription: t\n---\n# body\n")

    with pytest.raises(ValueError, match="Invalid skill name"):
        _load_skill(skill_dir)


def _fake_provider_ep(name: str, dist_name: str, path: Path) -> _FakeEntryPoint:
    return _FakeEntryPoint(
        name=name,
        value=f"{dist_name}.skills:skills_dir",
        dist=_FakeDist(name=dist_name),
        _loader=lambda path=path: lambda: path,
    )


def test_duplicate_entry_points_with_same_path_collapse(monkeypatch, tmp_path: Path):
    """When multiple distributions register the same provider name *and* their
    resolved paths are equal, they collapse to one provider. This is the
    no-op case for namespace-package merging — there's nothing to choose
    between them."""
    shared_root = tmp_path / "shared-skills"
    _write_skill_dir(shared_root, "shared-skill")
    fake_eps = [
        _fake_provider_ep("platform", "nemo-helix-ext", shared_root),
        _fake_provider_ep("platform", "nemo-helix-sdk", shared_root),
    ]
    monkeypatch.setattr(
        "nemo_helix_plugin.discovery.discover_entry_points",
        lambda _group: {ep.name: ep for ep in fake_eps},
    )
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._raw_entry_points",
        lambda _group: fake_eps,
    )

    skills = load_skills()

    assert set(skills.keys()) == {"shared-skill"}


def test_duplicate_entry_points_prefer_ext_when_content_matches(monkeypatch, tmp_path: Path):
    """When same-named providers resolve to different on-disk paths but ship
    byte-identical SKILL.md content (the source-vs-vendored case in this
    monorepo), the preferred distribution wins per
    ``_PROVIDER_PREFERENCE_RANKS``. The resulting skill's ``source_dist``
    reflects that choice."""
    ext_root = tmp_path / "ext-skills"
    sdk_root = tmp_path / "sdk-skills"
    _write_skill_dir(ext_root, "mirrored", body="# identical\n")
    _write_skill_dir(sdk_root, "mirrored", body="# identical\n")

    # SDK ordered first deliberately: a naive last-wins dedup would pick `ext`
    # by accident here; the explicit preference ranking is what makes this
    # deterministic.
    fake_eps = [
        _fake_provider_ep("platform", "nemo-helix-sdk", sdk_root),
        _fake_provider_ep("platform", "nemo-helix-ext", ext_root),
    ]
    monkeypatch.setattr(
        "nemo_helix_plugin.discovery.discover_entry_points",
        lambda _group: {ep.name: ep for ep in fake_eps},
    )
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._raw_entry_points",
        lambda _group: fake_eps,
    )

    skills = load_skills()

    assert skills["mirrored"].source_dist == "nemo-helix-ext"
    assert skills["mirrored"].source_dir == ext_root / "mirrored"


def test_duplicate_entry_points_prefer_superset_when_subset_subsumed(monkeypatch, tmp_path: Path):
    """When same-named providers differ only by one side having *additional*
    skills (the common case: source defines a new skill that hasn't been
    vendored yet), pick the superset so users see every skill the source
    declares. Subset semantics override the ext-rank tiebreaker — even an
    sdk-only superset wins over an ext subset."""
    ext_root = tmp_path / "ext-skills"
    sdk_root = tmp_path / "sdk-skills"
    _write_skill_dir(ext_root, "shared", body="# identical\n")
    _write_skill_dir(sdk_root, "shared", body="# identical\n")
    _write_skill_dir(sdk_root, "sdk-only", body="# bonus\n")

    fake_eps = [
        _fake_provider_ep("platform", "nemo-helix-ext", ext_root),
        _fake_provider_ep("platform", "nemo-helix-sdk", sdk_root),
    ]
    monkeypatch.setattr(
        "nemo_helix_plugin.discovery.discover_entry_points",
        lambda _group: {ep.name: ep for ep in fake_eps},
    )
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._raw_entry_points",
        lambda _group: fake_eps,
    )

    skills = load_skills()

    assert {"shared", "sdk-only"} <= skills.keys()
    assert skills["sdk-only"].source_dist == "nemo-helix-sdk"


def test_duplicate_entry_points_raise_on_content_drift(monkeypatch, tmp_path: Path):
    """Same-named providers whose contents genuinely diverge — neither side
    is a clean subset of the other — are a real packaging bug. Surface it as
    :class:`DuplicateSkillError` rather than silently picking a winner. Here
    each provider has a skill the other lacks AND an overlapping skill with
    different bytes, ruling out the superset-wins shortcut."""
    ext_root = tmp_path / "ext-skills"
    sdk_root = tmp_path / "sdk-skills"
    _write_skill_dir(ext_root, "drifted", body="# fresh content\n")
    _write_skill_dir(ext_root, "ext-only", body="# ext extra\n")
    _write_skill_dir(sdk_root, "drifted", body="# stale content\n")
    _write_skill_dir(sdk_root, "sdk-only", body="# sdk extra\n")
    fake_eps = [
        _fake_provider_ep("platform", "nemo-helix-ext", ext_root),
        _fake_provider_ep("platform", "nemo-helix-sdk", sdk_root),
    ]
    monkeypatch.setattr(
        "nemo_helix_plugin.discovery.discover_entry_points",
        lambda _group: {ep.name: ep for ep in fake_eps},
    )
    monkeypatch.setattr(
        "nemo_helix_ext.cli.commands.skills.registry._raw_entry_points",
        lambda _group: fake_eps,
    )

    with pytest.raises(DuplicateSkillError, match="different content"):
        load_skills()


def test_skill_class_available():
    skill = Skill(name="test", description="d", version="0.1", content="c", raw="r")
    assert skill.name == "test"
