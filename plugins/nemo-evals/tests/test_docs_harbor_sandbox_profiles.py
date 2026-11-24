# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Harbor sandbox profiles admin guide on the ``evals.runs`` config contract and the task image build rule.

Offline: nothing here renders the chart or contacts a deployment.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from nemo_builder_plugin.config import BuilderConfig
from nemo_evals.config import (
    HarborSandboxProfileConfig,
    HarborTaskJobsConfig,
    RunControllerConfig,
    TasksetRunsConfig,
    expected_builder_backend,
)
from nemo_evals.harbor.remote.profile import OpenSandboxClientSettings
from nemo_evals.image_builds.runner import backends_disagree
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/set-up/helm/harbor-sandbox-profiles.mdx"
LIMITS = "## Run limits and controller"
IMAGE_BUILDS = "## Task image builds"
LINKING = [
    ROOT / "docs/evals/agent-eval/harbor-kubernetes.mdx",
    ROOT / "skills/nemo-evals-plugin/references/harbor-kubernetes.md",
    ROOT / "skills/nemo-evals-plugin/references/troubleshooting.md",
]
# Keys the guide leaves out on purpose: admins keep the default.
UNDOCUMENTED = {
    "dispatcher_profiles",
    # Accepted, but no effect in this release.
    "image_reference_mode",
    "scoring_reserve_s",
    "stale_attempt_cleanup_timeout_s",
    "unschedulable_after_s",
}
API_HOST = "<api-service>.<namespace>.svc.cluster.local"
# The chart's evals.runs.harbor (k8s/helm/values.yaml), trimmed: what the recipes merge over.
CHART_HARBOR = {
    "sandbox_profiles": {
        "harbor-standard": {
            "agent_channel": [API_HOST],
            "agent_setup_egress": ["registry.npmjs.org", "pypi.org"],
            "egress_allow": [API_HOST],
        }
    },
}


def _fields() -> dict[str, FieldInfo]:
    """Every documented key: the harbor keys, the profile keys, and ``connection.*`` in place of ``connection``."""
    fields = dict(HarborTaskJobsConfig.model_fields)
    for name, field in HarborSandboxProfileConfig.model_fields.items():
        if name == "connection":
            fields |= {f"connection.{sub}": f for sub, f in OpenSandboxClientSettings.model_fields.items()}
        else:
            fields[name] = field
    return fields


def _run_fields() -> dict[str, FieldInfo]:
    """Every documented ``evals.runs`` key outside ``harbor``: the budgets, and ``controller.*`` in place of ``controller``."""
    fields = {name: f for name, f in TasksetRunsConfig.model_fields.items() if name not in {"harbor", "controller"}}
    return fields | {f"controller.{name}": f for name, f in RunControllerConfig.model_fields.items()}


def _section(heading: str) -> str:
    """One second-level section of the guide, from its heading to the next."""
    return heading + GUIDE.read_text().split(heading, 1)[1].split("\n## ", 1)[0]


def _limits_section() -> str:
    return _section(LIMITS)


def _image_builds_section() -> str:
    return _section(IMAGE_BUILDS)


def _rows(text: str) -> dict[str, str]:
    """Each table row's key and its Default cell."""
    return dict(re.findall(r"^\| `([a-z_.]+)` \| ([^|]+) \|", text, re.MULTILINE))


def _profile_rows() -> dict[str, str]:
    return _rows(GUIDE.read_text().replace(_limits_section(), ""))


def _yaml_blocks(text: str) -> list[dict[str, Any]]:
    return [yaml.safe_load(block) for block in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)]


def _slug(heading: str) -> str:
    return re.sub(r"\s+", "-", re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip())


def _merge(base: Any, override: Any) -> Any:
    """Helm's ``mergeOverwrite``: maps merge key by key; lists and scalars are replaced."""
    if isinstance(base, dict) and isinstance(override, dict):
        return {**base, **{key: _merge(base.get(key), value) for key, value in override.items()}}
    return override


def _assert_defaults(rows: dict[str, str], fields: dict[str, FieldInfo]) -> None:
    for key, cell in rows.items():
        field = fields[key]
        if cell.startswith("chart:"):
            assert field.is_required() and "`harbor-standard`" in cell, key
            continue
        token = re.search(r"`([^`]+)`", cell)
        documented = yaml.safe_load(token.group(1)) if token else None
        if isinstance(documented, list):
            documented = tuple(documented)
        assert field.default is not PydanticUndefined, key
        assert documented == field.default, f"{key}: page says {cell.strip()!r}, model says {field.default!r}"


def _recipes() -> list[dict[str, Any]]:
    """Every ``evals.runs`` recipe: all of the guide's YAML outside the image builds section, which sets none."""
    runs = _yaml_blocks(GUIDE.read_text().replace(_image_builds_section(), ""))
    return [recipe["platformConfig"]["evals"]["runs"] for recipe in runs]


def test_every_config_key_is_documented_once() -> None:
    rows = _profile_rows()
    assert not rows.keys() & UNDOCUMENTED, "a key is both in a table and UNDOCUMENTED"
    assert rows.keys() | UNDOCUMENTED == _fields().keys()


def test_documented_defaults_match_the_models() -> None:
    _assert_defaults(_profile_rows(), _fields())


def test_every_run_limit_is_documented_with_its_default() -> None:
    rows = _rows(_limits_section())
    assert rows.keys() == _run_fields().keys()
    _assert_defaults(rows, _run_fields())


def test_the_backend_switch_recipe_sets_builders_key_alone_and_evals_expects_that_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admin who follows the recipe changes one key. Evals, reading the same configuration, follows it."""
    recipes = _yaml_blocks(_image_builds_section())
    assert recipes, "no backend switch recipe"
    for recipe in recipes:
        config = recipe["platformConfig"]
        assert config.keys() == {"builder"}, "the recipe sets a key outside builder"
        builder = BuilderConfig.model_validate(config["builder"])
        monkeypatch.setattr(BuilderConfig, "get", classmethod(lambda cls, builder=builder: builder))
        assert expected_builder_backend() == builder.backend == "remote"


def test_the_guide_quotes_the_error_of_a_build_whose_backends_differ() -> None:
    """An admin searches the guide for the error a failed build shows, so the two must read the same."""
    quoted = re.search(r"`(Builder builds with [^`]+)`", _image_builds_section())
    assert quoted, "the guide doesn't quote the error"
    assert quoted.group(1) == backends_disagree("remote", "execution")


def test_run_limit_recipes_validate() -> None:
    limits = [recipe for recipe in _recipes() if recipe.keys() - {"harbor"}]
    assert limits, "no run limits recipe"
    for recipe in limits:
        TasksetRunsConfig.model_validate(_merge({"harbor": CHART_HARBOR}, recipe))


def test_recipes_in_order_keep_profiles_valid_and_connected() -> None:
    harbor: dict[str, Any] = CHART_HARBOR
    for recipe in (r for r in _recipes() if "harbor" in r):
        harbor = _merge(harbor, recipe["harbor"])
        config = HarborTaskJobsConfig.model_validate(harbor)
        for name, profile in config.sandbox_profiles.items():
            assert API_HOST in profile.agent_channel, f"{name} can't reach the gateway from no-network tasks"
            assert profile.agent_setup_egress, f"{name} can't install agents"
    assert config.default_sandbox_profile == "harbor-web"
    assert config.sandbox_profiles["harbor-standard"].max_cpus == 8  # the ceiling check in "Apply and verify"


def test_links_to_the_guide_name_existing_headings() -> None:
    headings = {_slug(h) for h in re.findall(r"^#{2,4} (.+)$", GUIDE.read_text(), re.MULTILINE)}
    for path in LINKING:
        anchors = re.findall(r"harbor-sandbox-profiles#([a-z0-9-]+)", path.read_text())
        assert anchors or "harbor-sandbox-profiles" in path.read_text(), f"{path.name} doesn't link the guide"
        assert set(anchors) <= headings, f"{path.name} links missing headings {set(anchors) - headings}"


def test_links_from_the_guide_name_existing_user_guide_headings() -> None:
    user_guide = LINKING[0].read_text()
    headings = {_slug(h) for h in re.findall(r"^#{2,4} (.+)$", user_guide, re.MULTILINE)}
    anchors = set(re.findall(r"harbor-kubernetes#([a-z0-9-]+)", GUIDE.read_text()))
    assert anchors and anchors <= headings, f"missing user guide headings {anchors - headings}"
