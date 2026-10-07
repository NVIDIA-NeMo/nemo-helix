# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent-skill injection (pure; no nemo_fabric native stack required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.runtimes.fabric import skills as skills_module
from nhx_evals_sdk.agent_eval.runtimes.fabric.skills import (
    CODEX_SKILLS_DIR,
    SKILL_MODE_CODEX_SKILLS_DIR,
    SKILL_MODE_NATIVE,
    AgentSkill,
    SkillInjectionError,
    install_skill,
    install_skills,
    native_skills_route,
    resolve_skill_mode,
)


def _plan(*, native: bool | None) -> dict[str, object]:
    """Build a ``RunPlan.capability_plan``-shaped mapping. ``native=None`` means no skills route at all."""
    if native is None:
        return {"routes": []}
    return {"routes": [{"kind": "skills", "target": "harness_native" if native else "unsupported"}]}


_SKILL_MD = "---\nname: code-review\ndescription: Review code thoroughly.\n---\n\nBe thorough."


def _make_bundle(base: Path, name: str = "code-review", extra: dict[str, str] | None = None) -> Path:
    root = base / name
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    for rel, content in (extra or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def test_from_directory_defaults_name_from_basename(tmp_path: Path) -> None:
    skill = AgentSkill.from_directory(_make_bundle(tmp_path))
    assert skill.name == "code-review"
    assert skill.directory == (tmp_path / "code-review").resolve()


def test_from_directory_requires_skill_md(tmp_path: Path) -> None:
    src = tmp_path / "no-skill"
    src.mkdir()
    (src / "notes.md").write_text("hi", encoding="utf-8")
    with pytest.raises(SkillInjectionError):
        AgentSkill.from_directory(src)


@pytest.mark.parametrize("bad", ["Code-Review", "-pdf", "pdf-", "pdf--processing", "has space", ""])
def test_invalid_names_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        AgentSkill(name=bad, directory=Path("/skills/x"))


@pytest.mark.parametrize(
    ("capability_plan", "expected"),
    [
        ({"routes": [{"kind": "skills", "target": "harness_native"}]}, True),
        ({"routes": [{"kind": "skills", "target": "unsupported"}]}, False),
        ({"routes": [{"kind": "tools", "target": "harness_native"}]}, False),  # non-skills route ignored
        ({"routes": []}, False),
        ({}, False),  # no routes key
        ({"routes": "not-a-list"}, False),  # defensive: malformed shape
    ],
)
def test_native_skills_route(capability_plan: dict[str, object], expected: bool) -> None:
    assert native_skills_route(capability_plan) is expected


@pytest.mark.parametrize(
    ("capability_plan", "adapter_id", "expected"),
    [
        # Native routing wins regardless of adapter id (e.g. Hermes, or an end-user adapter).
        (_plan(native=True), "nvidia.fabric.hermes", SKILL_MODE_NATIVE),
        (_plan(native=True), "acme.custom", SKILL_MODE_NATIVE),
        # Not native, but the Codex adapter -> self-discovered .agents/skills dir.
        (_plan(native=False), "nvidia.fabric.codex", SKILL_MODE_CODEX_SKILLS_DIR),
        (_plan(native=None), "nvidia.fabric.codex", SKILL_MODE_CODEX_SKILLS_DIR),
        (_plan(native=False), "NVIDIA.FABRIC.CODEX", SKILL_MODE_CODEX_SKILLS_DIR),  # case-insensitive
        # Neither native nor codex -> unsupported (runtime fails fast).
        (_plan(native=False), "nvidia.fabric.hermes", None),
        (_plan(native=None), "acme.other", None),
    ],
)
def test_resolve_skill_mode(capability_plan: dict[str, object], adapter_id: str, expected: str | None) -> None:
    assert resolve_skill_mode(capability_plan=capability_plan, adapter_id=adapter_id) == expected


def test_install_native_stages_named_dir_and_overlay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    stage = tmp_path / "stage"
    workspace.mkdir()

    installation = install_skill(
        skill=AgentSkill.from_directory(_make_bundle(tmp_path / "src")),
        adapter_id="nvidia.fabric.hermes",
        mode=SKILL_MODE_NATIVE,
        workspace_dir=workspace,
        skill_stage_dir=stage,
    )

    # Bundle is staged under an isolated <name>/ dir (spec: name matches dir), not the agent workspace.
    skill_root = stage / "code-review"
    assert (skill_root / "SKILL.md").is_file()
    assert not (workspace / "SKILL.md").exists()
    assert not (workspace / ".agents").exists()

    assert installation.skill_paths == [str(skill_root)]

    prov = installation.provenance
    assert prov["name"] == "code-review"
    assert prov["mode"] == SKILL_MODE_NATIVE
    assert prov["location"] == str(skill_root)
    assert isinstance(prov["hash"], str) and prov["hash"]


def test_install_native_copies_directory_tree(tmp_path: Path) -> None:
    src = _make_bundle(tmp_path / "src", extra={"references/ref.md": "material", "scripts/run.py": "print()"})
    stage = tmp_path / "stage"

    install_skill(
        skill=AgentSkill.from_directory(src),
        adapter_id="nvidia.fabric.hermes",
        mode=SKILL_MODE_NATIVE,
        workspace_dir=tmp_path / "workspace",
        skill_stage_dir=stage,
    )

    base = stage / "code-review"
    assert (base / "SKILL.md").is_file()
    assert (base / "references" / "ref.md").read_text(encoding="utf-8") == "material"
    assert (base / "scripts" / "run.py").read_text(encoding="utf-8") == "print()"


def test_install_codex_places_under_agents_skills(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    installation = install_skill(
        skill=AgentSkill.from_directory(_make_bundle(tmp_path / "src")),
        adapter_id="nvidia.fabric.codex",
        mode=SKILL_MODE_CODEX_SKILLS_DIR,
        workspace_dir=workspace,
        skill_stage_dir=tmp_path / "stage",
    )

    # Codex discovers agentskills bundles from .agents/skills/ in its working directory.
    skill_md = workspace / ".agents" / "skills" / "code-review" / "SKILL.md"
    assert "Be thorough." in skill_md.read_text(encoding="utf-8")
    # No skills path: placement in the workspace is the delivery mechanism.
    assert installation.skill_paths == []
    assert installation.provenance["mode"] == SKILL_MODE_CODEX_SKILLS_DIR
    assert installation.provenance["location"] == f"{CODEX_SKILLS_DIR}/code-review"


def test_codex_bundle_does_not_collide_with_workspace_root(tmp_path: Path) -> None:
    # A task-seeded workspace file at the root is untouched: the skill lives under .agents/skills/.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.csv").write_text("task input", encoding="utf-8")
    src = _make_bundle(tmp_path / "src", name="collide", extra={"data.csv": "skill payload"})

    install_skill(
        skill=AgentSkill.from_directory(src),
        adapter_id="nvidia.fabric.codex",
        mode=SKILL_MODE_CODEX_SKILLS_DIR,
        workspace_dir=workspace,
        skill_stage_dir=tmp_path / "stage",
    )

    assert (workspace / "data.csv").read_text(encoding="utf-8") == "task input"
    assert (workspace / ".agents" / "skills" / "collide" / "data.csv").read_text(encoding="utf-8") == "skill payload"


def test_install_native_returns_only_the_staged_path(tmp_path: Path) -> None:
    # Preserving config-declared skills is FabricConfig.add_skill_path's job (it appends and
    # de-duplicates), so installation reports only what it staged and never re-lists prior paths.
    installation = install_skill(
        skill=AgentSkill.from_directory(_make_bundle(tmp_path / "src")),
        adapter_id="nvidia.fabric.hermes",
        mode=SKILL_MODE_NATIVE,
        workspace_dir=tmp_path / "workspace",
        skill_stage_dir=tmp_path / "stage",
    )

    assert installation.skill_paths == [str(tmp_path / "stage" / "code-review")]


def test_install_native_recreates_stale_stage(tmp_path: Path) -> None:
    # Re-staging into an existing stage (reused run id) must yield an *exact* copy of the source — a file
    # since removed from the bundle must not survive.
    stage = tmp_path / "stage"
    install_skill(
        skill=AgentSkill.from_directory(_make_bundle(tmp_path / "v1", extra={"old.md": "stale"})),
        adapter_id="nvidia.fabric.hermes",
        mode=SKILL_MODE_NATIVE,
        workspace_dir=tmp_path / "workspace",
        skill_stage_dir=stage,
    )
    assert (stage / "code-review" / "old.md").exists()

    install_skill(
        skill=AgentSkill.from_directory(_make_bundle(tmp_path / "v2")),  # no old.md
        adapter_id="nvidia.fabric.hermes",
        mode=SKILL_MODE_NATIVE,
        workspace_dir=tmp_path / "workspace",
        skill_stage_dir=stage,
    )
    assert (stage / "code-review" / "SKILL.md").is_file()
    assert not (stage / "code-review" / "old.md").exists()  # stale file recreated away


def test_install_codex_rejects_reserved_path_collision(tmp_path: Path) -> None:
    # A task seed occupying the reserved Codex skill path must not be silently clobbered/merged.
    workspace = tmp_path / "workspace"
    reserved = workspace / CODEX_SKILLS_DIR / "code-review"
    reserved.mkdir(parents=True)
    (reserved / "task_seed.txt").write_text("task file at the reserved path", encoding="utf-8")

    with pytest.raises(SkillInjectionError, match="reserved path"):
        install_skill(
            skill=AgentSkill.from_directory(_make_bundle(tmp_path / "src")),
            adapter_id="nvidia.fabric.codex",
            mode=SKILL_MODE_CODEX_SKILLS_DIR,
            workspace_dir=workspace,
            skill_stage_dir=tmp_path / "stage",
        )


def test_hash_is_content_sensitive(tmp_path: Path) -> None:
    one = tmp_path / "one" / "code-review"
    two = tmp_path / "two" / "code-review"
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    (one / "SKILL.md").write_text("one", encoding="utf-8")
    (two / "SKILL.md").write_text("two", encoding="utf-8")

    a = install_skill(
        skill=AgentSkill.from_directory(one),
        adapter_id="nvidia.fabric.hermes",
        mode=SKILL_MODE_NATIVE,
        workspace_dir=tmp_path / "wa",
        skill_stage_dir=tmp_path / "sa",
    )
    b = install_skill(
        skill=AgentSkill.from_directory(two),
        adapter_id="nvidia.fabric.hermes",
        mode=SKILL_MODE_NATIVE,
        workspace_dir=tmp_path / "wb",
        skill_stage_dir=tmp_path / "sb",
    )
    assert a.provenance["hash"] != b.provenance["hash"]


def test_install_skills_rolls_back_staged_bundles_on_failure(tmp_path: Path) -> None:
    # All-or-nothing: a later skill failing to stage rolls back the bundles already staged in this call,
    # so a partial skill set never lingers on disk.
    good = AgentSkill.from_directory(_make_bundle(tmp_path / "src", name="good"))
    (tmp_path / "missing").mkdir()
    bad = AgentSkill(name="bad", directory=tmp_path / "missing")  # no SKILL.md -> stages fail
    stage_dir = tmp_path / "stage"

    with pytest.raises(SkillInjectionError):
        install_skills(
            skills=[good, bad],
            adapter_id="nvidia.fabric.hermes",
            mode=SKILL_MODE_NATIVE,
            workspace_dir=tmp_path / "ws",
            skill_stage_dir=stage_dir,
        )

    # The first skill was staged, then rolled back when the second failed.
    assert not (stage_dir / "good").exists()


def test_install_skills_rolls_back_the_bundle_that_failed_mid_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # install_skill can raise AFTER writing files (a copytree failing partway, an unreadable file while
    # hashing). That skill's own partially staged bundle must be rolled back too, not just the bundles
    # from earlier iterations. Hashing runs once the bundle is fully copied, so failing it there puts
    # real files on disk before the error.
    good = AgentSkill.from_directory(_make_bundle(tmp_path / "src", name="good"))
    late = AgentSkill.from_directory(_make_bundle(tmp_path / "src2", name="late"))
    stage_dir = tmp_path / "stage"

    real_hash = skills_module._hash_directory

    def _fail_on_late(directory: Path) -> str:
        if directory.name == "late":
            raise OSError("unreadable file in bundle")
        return real_hash(directory)

    monkeypatch.setattr(skills_module, "_hash_directory", _fail_on_late)

    with pytest.raises(OSError):
        install_skills(
            skills=[good, late],
            adapter_id="nvidia.fabric.hermes",
            mode=SKILL_MODE_NATIVE,
            workspace_dir=tmp_path / "ws",
            skill_stage_dir=stage_dir,
        )

    assert not (stage_dir / "good").exists()  # earlier bundle rolled back, as before
    assert not (stage_dir / "late").exists()  # ...and so is the one that failed after staging files


def test_install_skills_rollback_never_deletes_preexisting_seed_file(tmp_path: Path) -> None:
    # Codex reserved-path collision: the second skill's target already holds a task-seeded file. Rollback
    # must remove only the bundle THIS call staged (the first skill), never the pre-existing seed file.
    good = AgentSkill.from_directory(_make_bundle(tmp_path / "src", name="good"))
    collide = AgentSkill.from_directory(_make_bundle(tmp_path / "src2", name="collide"))
    workspace = tmp_path / "ws"
    seeded = workspace / CODEX_SKILLS_DIR / "collide"  # a task-seeded file on the reserved codex path
    seeded.mkdir(parents=True)
    (seeded / "seed.txt").write_text("task data", encoding="utf-8")

    with pytest.raises(SkillInjectionError):
        install_skills(
            skills=[good, collide],
            adapter_id="nvidia.fabric.codex",
            mode=SKILL_MODE_CODEX_SKILLS_DIR,
            workspace_dir=workspace,
            skill_stage_dir=tmp_path / "stage",
        )

    # The staged first bundle is rolled back...
    assert not (workspace / CODEX_SKILLS_DIR / "good").exists()
    # ...but the pre-existing task-seeded file is untouched.
    assert (seeded / "seed.txt").read_text(encoding="utf-8") == "task data"


class _Config:
    """The slice of FabricConfig the relocation touches: ``skills.paths`` plus add/remove."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = list(paths)

    @property
    def skills(self) -> object | None:
        return None if not self.paths else type("Skills", (), {"paths": list(self.paths)})()

    def add_skill_path(self, path: str | Path) -> None:
        if str(path) not in self.paths:
            self.paths.append(str(path))

    def remove_skill_path(self, path: str | Path) -> None:
        self.paths = [p for p in self.paths if p != str(path)]


def test_workspace_rooted_skills_names_deepagents_only() -> None:
    assert skills_module.workspace_rooted_skills("nvidia.fabric.langchain.deepagents")
    assert skills_module.workspace_rooted_skills(" NVIDIA.Fabric.LangChain.DeepAgents ")
    assert not skills_module.workspace_rooted_skills("nvidia.fabric.hermes")
    assert not skills_module.workspace_rooted_skills("nvidia.fabric.codex")


def test_relocate_copies_each_bundle_under_agents_skills_and_rewrites_the_config(tmp_path: Path) -> None:
    base = tmp_path / "agent"
    _make_bundle(base / "skills", "arithmetic")
    absolute = _make_bundle(tmp_path / "elsewhere", "style", extra={"notes.md": "n"})
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = _Config(["skills/arithmetic", str(absolute)])

    locations = skills_module.relocate_skills_into_workspace(config, base_dir=base, workspace_dir=workspace)

    assert locations == [".agents/skills/arithmetic", ".agents/skills/style"]
    assert (workspace / ".agents/skills/arithmetic/SKILL.md").read_text() == _SKILL_MD
    assert (workspace / ".agents/skills/style/notes.md").read_text() == "n"
    assert config.paths == ["/.agents/skills"]


def test_relocate_leaves_a_config_without_skills_alone(tmp_path: Path) -> None:
    config = _Config([])
    assert skills_module.relocate_skills_into_workspace(config, base_dir=tmp_path, workspace_dir=tmp_path) == []
    assert config.paths == []


def test_relocate_refuses_two_bundles_with_one_name(tmp_path: Path) -> None:
    _make_bundle(tmp_path / "a/skills", "review")
    _make_bundle(tmp_path / "b/skills", "review")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = _Config([str(tmp_path / "a/skills/review"), str(tmp_path / "b/skills/review")])

    with pytest.raises(SkillInjectionError, match="both stage as .agents/skills/review"):
        skills_module.relocate_skills_into_workspace(config, base_dir=None, workspace_dir=workspace)


def test_relocate_refuses_a_path_without_a_skill_doc(tmp_path: Path) -> None:
    (tmp_path / "agent/skills/empty").mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = _Config(["skills/empty"])

    with pytest.raises(SkillInjectionError, match="has no SKILL.md"):
        skills_module.relocate_skills_into_workspace(config, base_dir=tmp_path / "agent", workspace_dir=workspace)
    assert config.paths == ["skills/empty"]  # nothing rewritten on failure


def test_relocate_will_not_clobber_a_seeded_workspace_file(tmp_path: Path) -> None:
    _make_bundle(tmp_path / "agent/skills", "review")
    workspace = tmp_path / "ws"
    (workspace / ".agents/skills/review").mkdir(parents=True)
    (workspace / ".agents/skills/review/SKILL.md").write_text("task-seeded")
    config = _Config(["skills/review"])

    with pytest.raises(SkillInjectionError, match="reserved path"):
        skills_module.relocate_skills_into_workspace(config, base_dir=tmp_path / "agent", workspace_dir=workspace)
