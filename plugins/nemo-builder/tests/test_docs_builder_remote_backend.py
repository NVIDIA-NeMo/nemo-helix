# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the operator's guide to the remote backend on the builder's settings, its step driver and what it answers.

Offline: nothing here renders the chart or contacts a deployment.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from functools import cache
from pathlib import Path
from typing import Any, get_args

import yaml
from nemo_builder_plugin.config import BuilderConfig, RemoteConfig
from nemo_builder_plugin.remote import CANCEL_MARGIN_SECONDS, STEP_BACKENDS
from nemo_builder_plugin.run.remote import Stage, step_main
from nemo_helix_plugin.jobs.execution_profiles import (
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
)
from pydantic.fields import FieldInfo

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/set-up/helm/builder-remote-backend.mdx"
HARBOR_GUIDE = ROOT / "docs/set-up/helm/harbor-sandbox-profiles.mdx"
README = ROOT / "plugins/nemo-builder/README.md"
#: Where the sentences the guide quotes are spelt: the plugin, and the Jobs backend the step runs on, whose
#: message for a step that ran out of time a failed image's row repeats.
SOURCES = [
    *sorted((ROOT / "plugins/nemo-builder/src/nemo_builder_plugin").rglob("*.py")),
    ROOT / "services/core/jobs/src/nhx/core/jobs/controllers/backends/kubernetes/kubernetes_job.py",
]

SETTINGS = "## Select the backend"
STEP_IMAGE = "## Build the step image"
PROFILE = "## Add the Jobs profile"
CREDENTIAL = "## Give the step its credential"
TROUBLESHOOTING = "## Troubleshooting"
#: From here on the guide quotes evals' errors, which the builder's source doesn't spell.
EVALS_ERRORS = "### An evals build fails after a switch"

#: The `remote` keys whose value in the recipe is an example: none has a default a deployment could build with.
EXAMPLES = {"registry", "repository", "image", "credential_secret"}
#: The shortest string of the sources that is taken for part of a quoted sentence.
MIN_SPELT = 12
#: How much of a quoted sentence the sources must spell. Not all of it: where the code fills in a value the
#: guide may name one, such as `kubernetes_job` or `RuntimeError`, and a quoted value keeps its quotation marks.
SPELT_SHARE = 2 / 3
PLACEHOLDER = re.compile(r"<[^<>]+>")


def _text(path: Path = GUIDE) -> str:
    return path.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """One second-level section of the guide, from its heading to the next."""
    return heading + _text().split(heading, 1)[1].split("\n## ", 1)[0]


def _yaml_blocks(text: str) -> list[dict[str, Any]]:
    return [yaml.safe_load(block) for block in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)]


def _slug(heading: str) -> str:
    return re.sub(r"\s+", "-", re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip())


def _headings(path: Path) -> set[str]:
    return {_slug(heading) for heading in re.findall(r"^#{2,4} (.+)$", _text(path), re.MULTILINE)}


def _settings() -> dict[str, FieldInfo]:
    """Every key the guide's settings table documents: `backend`, and each `remote` key."""
    remote = {f"remote.{name}": field for name, field in RemoteConfig.model_fields.items()}
    return {"backend": BuilderConfig.model_fields["backend"], **remote}


def _backend_recipe() -> dict[str, Any]:
    (recipe,) = _yaml_blocks(_section(SETTINGS))
    return recipe["platformConfig"]["builder"]


@cache
def _spelt() -> list[tuple[str | None, ...]]:
    """Every string the sources spell, in its parts: None where an f-string fills in a value."""
    spelt: list[tuple[str | None, ...]] = []
    for path in SOURCES:
        tree = ast.parse(_text(path))
        formatted = [node for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)]
        # A part of an f-string is not a string of its own.
        within = {id(part) for node in formatted for part in node.values}
        for node in ast.walk(tree):
            parts: tuple[str | None, ...]
            if isinstance(node, ast.JoinedStr):
                parts = tuple(
                    part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else None
                    for part in node.values
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in within:
                parts = (node.value,)
            else:
                continue
            if sum(len(part) for part in parts if part) >= MIN_SPELT:
                spelt.append(parts)
    return spelt


def _spelt_share(sentence: str) -> float:
    """How much of ``sentence``, its placeholders aside, the sources spell: in one string, or in the several
    that a message is put together from."""
    spelt = [False] * len(sentence)
    for parts in _spelt():
        # Cheap, and most strings fail it: one whose longest part isn't in the sentence can't be.
        if max([part for part in parts if part], key=len) not in sentence:
            continue
        pattern = re.compile("".join(".+?" if part is None else f"({re.escape(part)})" for part in parts))
        for found in pattern.finditer(sentence):
            for group in range(1, pattern.groups + 1):
                start, end = found.span(group)
                spelt[start:end] = [True] * (end - start)
    placeholders = {position for found in PLACEHOLDER.finditer(sentence) for position in range(*found.span())}
    fixed = [position for position in range(len(sentence)) if position not in placeholders]
    return sum(spelt[position] for position in fixed) / len(fixed)


def _quoted_sentences() -> list[str]:
    """Each sentence a troubleshooting table quotes: a cell that is one code span, and not a code or a status."""
    tables = _section(TROUBLESHOOTING).split(EVALS_ERRORS, 1)[0]
    return [cell for cell in re.findall(r"\| `([^`]+)` (?=\|)", tables) if " " in cell]


def test_the_settings_table_has_every_key_at_its_default() -> None:
    rows = dict(re.findall(r"^\| `([a-z_.]+)` \| ([^|]+) \|", _section(SETTINGS), re.MULTILINE))
    fields = _settings()
    assert rows.keys() == fields.keys()
    for key, cell in rows.items():
        token = re.search(r"`([^`]+)`", cell)
        assert token or cell.strip() == "unset", f"{key}: no default in {cell!r}"
        documented = yaml.safe_load(token.group(1)) if token else None
        default = fields[key].get_default(call_default_factory=True)
        assert documented == default, f"{key}: the guide says {cell.strip()!r}, the model says {default!r}"


def test_the_backend_recipe_validates_and_shows_the_other_keys_at_their_defaults() -> None:
    recipe = _backend_recipe()
    assert BuilderConfig.model_validate(recipe).backend == "remote"
    shown, defaults = recipe["remote"], RemoteConfig()
    assert shown.keys() == RemoteConfig.model_fields.keys()
    for key in shown.keys() - EXAMPLES:
        assert shown[key] == getattr(defaults, key), key


def test_the_profile_recipe_validates_and_can_run_the_backend_recipe() -> None:
    remote = BuilderConfig.model_validate(_backend_recipe()).remote
    (recipe,) = _yaml_blocks(_section(PROFILE))
    (executor,) = recipe["platformConfig"]["jobs"]["executors"]
    assert executor["config"].keys() <= KubernetesJobExecutionProfileConfig.model_fields.keys()
    profile = KubernetesJobExecutionProfile.model_validate(executor)
    assert (profile.provider, profile.profile) == ("cpu", remote.profile)
    assert profile.backend in STEP_BACKENDS
    # What a submit holds the profile's limit to: see `RemoteBackend._check_deployment`.
    floor = remote.build_timeout_seconds * remote.max_specs_per_set + CANCEL_MARGIN_SECONDS
    assert profile.config.ttl_seconds_active >= floor


def test_the_profile_table_names_real_keys_and_their_defaults() -> None:
    rows = dict(re.findall(r"^\| `config\.([a-z_]+)` \| ([^|]+) \|", _section(PROFILE), re.MULTILINE))
    fields = KubernetesJobExecutionProfileConfig.model_fields
    assert rows and rows.keys() <= fields.keys()
    for key in ("max_active_jobs", "ttl_seconds_active", "ttl_seconds_before_active"):
        assert f"If unset, {fields[key].default}" in rows[key], key


def test_the_mounted_credential_recipe_validates_and_is_where_the_step_program_reads() -> None:
    (recipe,) = _yaml_blocks(_section(CREDENTIAL))
    config = recipe["platformConfig"]
    assert RemoteConfig.model_validate(config["builder"]["remote"]).credential_secret is None
    (executor,) = config["jobs"]["executors"]
    storage = KubernetesJobExecutionProfile.model_validate(executor).config.storage
    (volume,) = storage.additional_volumes
    (mount,) = storage.additional_volume_mounts
    assert volume.secret is not None and mount.name == volume.name and mount.read_only
    assert f'Path("{mount.mount_path}/' in _section(STEP_IMAGE)


def test_the_step_program_parses_and_uses_what_the_plugin_has() -> None:
    (program,) = re.findall(r"```python\n(.*?)```", _section(STEP_IMAGE), re.DOTALL)
    imports = [node for node in ast.walk(ast.parse(program)) if isinstance(node, ast.ImportFrom)]
    names = [(node.module or "", alias.name) for node in imports for alias in node.names]
    ours = [(module, name) for module, name in names if module.startswith("nemo_builder_plugin")]
    assert ours, "the step program imports nothing from the plugin"
    for module, name in ours:
        assert hasattr(importlib.import_module(module), name), f"{module} has no {name}"
    assert "step_main(make_service, read_credential=read_credential)" in program
    assert "read_credential" in inspect.signature(step_main).parameters


def test_the_guide_lists_the_stages_the_driver_knows() -> None:
    listed = re.search(r"reports one of \w+ stages: ([^.]+)\.", _section(STEP_IMAGE))
    assert listed, "the guide doesn't list the stages"
    assert tuple(re.findall(r"`([a-z]+)`", listed.group(1))) == get_args(Stage)


def test_each_sentence_the_troubleshooting_tables_quote_is_spelt_by_the_sources() -> None:
    """An operator searches the guide for the sentence an answer or a row shows, so the two must read the same."""
    sentences = _quoted_sentences()
    assert len(sentences) >= 40, "the troubleshooting tables are not read as they were"
    shares = {sentence: _spelt_share(sentence) for sentence in sentences}
    unspelt = {sentence: f"{share:.0%}" for sentence, share in shares.items() if share < SPELT_SHARE}
    assert not unspelt, f"the guide quotes what the sources don't say: {unspelt}"


def test_links_name_existing_headings() -> None:
    guide = _text()
    for target, link in (
        (GUIDE, r"\]\(#([a-z0-9-]+)\)"),
        (HARBOR_GUIDE, r"harbor-sandbox-profiles#([a-z0-9-]+)"),
        (README, r"nemo-builder/README\.md#([a-z0-9-]+)"),
    ):
        anchors = set(re.findall(link, guide))
        assert anchors, f"the guide has no link into {target.name}"
        assert anchors <= _headings(target), f"{target.name} has no heading for {anchors - _headings(target)}"
