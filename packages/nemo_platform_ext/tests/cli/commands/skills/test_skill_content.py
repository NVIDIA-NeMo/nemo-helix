# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skill content loading.

The platform's own bundled skills are exposed through the same ``nemo.skills``
entry-point mechanism that third-party plugins use, so these tests exercise the
public ``load_skills()`` API rather than any "built-in" branch.
"""

from pathlib import Path

from nemo_platform_ext.cli.commands.skills.base import Skill
from nemo_platform_ext.cli.commands.skills.registry import _load_skills_cached, load_skills

KNOWN_SKILL_PRECONDITIONS = frozenset(
    {
        "agent_config_exists",
        "agent_design_complete",
        "ethos_exists",
        "agents_plugin_available",
        "clickhouse_ready",
        "evaluator_sdk_available",
        "guardrails_plugin_available",
        "nemo_cli_available",
        "nemo_setup_complete",
        "platform_running",
        "provider_registered",
        "secrets_configured",
        "user_confirmation_required",
        "workspace_exists",
    }
)


def setup_function() -> None:
    """Drop both registry caches before every test.

    See ``test_cli.py`` and ``test_registry.py`` for the same pattern — earlier
    tests (here or elsewhere) may have pinned ``discover_entry_points``'s
    ``@cache`` with a value computed under monkeypatch, which would otherwise
    bleed into these tests and make the platform's skills appear empty.
    """
    from nemo_platform_plugin.discovery import discover_entry_points as _discover_eps

    _load_skills_cached.cache_clear()
    _discover_eps.cache_clear()


class TestSkillDataclass:
    def test_skill_has_required_fields(self):
        skill = Skill(
            name="test",
            description="A test skill",
            version="0.1",
            content="# Test",
            raw="---\nname: test\n---\n# Test",
        )
        assert skill.name == "test"
        assert skill.description == "A test skill"
        assert skill.version == "0.1"
        assert skill.content == "# Test"
        assert skill.preconditions == []

    def test_skill_has_source_dir(self):
        skill = Skill(
            name="test",
            description="A test skill",
            version="0.1",
            content="# Test",
            raw="---\nname: test\n---\n# Test",
            source_dir=Path("/fake/path"),
        )
        assert skill.source_dir == Path("/fake/path")


class TestLoadPlatformSkills:
    """The platform's bundled skills must always be loadable via ``load_skills()``."""

    def test_returns_dict_of_skills(self):
        skills = load_skills()
        assert isinstance(skills, dict)
        assert len(skills) > 0

    def test_contains_expected_skills(self):
        skills = load_skills()
        # Canonical platform skills must remain present — adding new ones is free.
        expected = {"inference"}
        assert expected <= skills.keys()

    def test_each_skill_has_valid_fields(self):
        for name, skill in load_skills().items():
            assert skill.name == name
            assert len(skill.description) > 0
            assert len(skill.version) > 0
            assert len(skill.content) > 0

    def test_content_has_frontmatter_stripped(self):
        for skill in load_skills().values():
            assert not skill.content.startswith("---")

    def test_raw_has_frontmatter(self):
        for skill in load_skills().values():
            assert skill.raw.startswith("---")

    def test_each_skill_has_source_dir(self):
        for skill in load_skills().values():
            assert skill.source_dir is not None
            assert skill.source_dir.is_dir()
            assert (skill.source_dir / "SKILL.md").exists()

    def test_platform_skills_declare_known_preconditions(self):
        skills_with_preconditions = []
        for name, skill in load_skills().items():
            if skill.source_plugin != "platform":
                continue
            if skill.preconditions:
                skills_with_preconditions.append(name)
            unknown = set(skill.preconditions) - KNOWN_SKILL_PRECONDITIONS
            assert not unknown, f"{name} has unknown preconditions: {sorted(unknown)}"
        assert skills_with_preconditions

    def test_build_agent_references_are_packaged(self):
        skill = load_skills()["nemo-build-agent"]
        assert skill.source_dir is not None
        assert (skill.source_dir / "references" / "fabric-deep-agents.md").is_file()
        assert (skill.source_dir / "references" / "testing-and-signoff.md").is_file()
        assert (skill.source_dir / "references" / "packaging.md").is_file()

    def test_build_agent_uses_agents_plugin_harness_dependency(self):
        skill = load_skills()["nemo-build-agent"]
        assert "optional NeMo Agents plugin supplies the selected harness adapter" in skill.content
        assert "Do not add a separate Deep Agents version constraint" in skill.content
        assert "If the NeMo Agents plugin and Deep Agents harness are already available" in skill.content
        assert 'uv pip install "nemo-platform[nemo-agents-plugin]"' in skill.content
        assert "uv pip install -e plugins/nemo-agents/" in skill.content

    def test_build_agent_covers_plugin_dependency_failure_paths(self):
        skill = load_skills()["nemo-build-agent"]
        assert "If the plugin is absent" in skill.content
        assert "ask for approval before" in skill.content
        assert "If the plugin is present" in skill.content
        assert "its Deep Agents adapter or runtime is absent" in skill.content
        assert "Do not install the harness independently" in skill.content

    def test_build_agent_defers_registration_until_after_local_gates(self):
        content = load_skills()["nemo-build-agent"].content
        normalized = " ".join(content.split())

        assert "do not run its `nemo agents create` registration step" in normalized
        assert "This build workflow owns registration after every pre-registration gate has passed" in normalized
        assert "do not run it earlier solely to validate the config" in normalized
        assert content.index("## Test before registration") < content.index("## Register and deploy")

    def test_build_agent_registration_uses_confirmed_workspace_and_environment(self):
        skill = load_skills()["nemo-build-agent"]
        assert skill.source_dir is not None
        packaging = (skill.source_dir / "references" / "packaging.md").read_text()
        create_command, deployments = packaging.split(".venv/bin/nemo agents deploy", maxsplit=1)
        selected_environment, no_environment = deployments.split("If no AgentEnvironment was selected", maxsplit=1)

        assert packaging.count(".venv/bin/nemo agents deploy") == 2
        assert "If an AgentEnvironment was selected" in packaging
        assert '--workspace "$WORKSPACE"' in create_command
        assert '--workspace "$WORKSPACE"' in selected_environment
        assert '--environment "$AGENT_ENVIRONMENT"' in selected_environment
        assert "AGENT_ENVIRONMENT=" not in no_environment
        assert "--environment" not in no_environment
        assert '--workspace "$WORKSPACE"' in no_environment

    def test_retrieval_recipes_references_are_packaged(self):
        skill = load_skills()["nemo-retrieval-recipes"]
        assert skill.source_dir is not None
        for relative in (
            "references/embed.md",
            "references/rerank.md",
        ):
            assert (skill.source_dir / relative).is_file(), relative
        assert "nemo data-designer create" in skill.content
        assert "retrieve-eval" in skill.content

    def test_model_selection_benchmark_cache_is_packaged(self):
        skill = load_skills()["nemo-model-selection"]
        assert skill.source_dir is not None
        assert (skill.source_dir / "references" / "benchmark_cache.json").is_file()

    def test_returns_new_dict_each_call(self):
        """Verify callers can't corrupt the cached data."""
        skills1 = load_skills()
        skills1.pop("inference")
        skills2 = load_skills()
        assert skills1 is not skills2
        assert "inference" in skills2
