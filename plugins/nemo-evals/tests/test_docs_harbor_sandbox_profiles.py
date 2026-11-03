# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Harbor sandbox profiles admin guide on the ``evals.runs.harbor`` config contract.

Offline: nothing here renders the chart or contacts a deployment.
"""

import re
from pathlib import Path
from typing import Any

import yaml
from nemo_evals.config import HarborSandboxProfileConfig, HarborTaskJobsConfig
from nemo_evals.harbor.remote.profile import OpenSandboxClientSettings
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/set-up/helm/harbor-sandbox-profiles.mdx"
LINKING = [
    ROOT / "docs/evals/agent-eval/harbor-kubernetes.mdx",
    ROOT / "skills/nemo-evals-plugin/references/harbor-kubernetes.md",
    ROOT / "skills/nemo-evals-plugin/references/troubleshooting.md",
]
API_HOST = "<FULLNAME>-api.<namespace>.svc.cluster.local"
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


def _rows() -> dict[str, str]:
    """Each table row's key and its Default cell."""
    return dict(re.findall(r"^\| `([a-z_.]+)` \| ([^|]+) \|", GUIDE.read_text(), re.MULTILINE))


def _reserved() -> set[str]:
    section = GUIDE.read_text().split("### Reserved", 1)[1].split("\n#", 1)[0]
    return set(re.findall(r"^- `([a-z_]+)`$", section, re.MULTILINE))


def _slug(heading: str) -> str:
    return re.sub(r"\s+", "-", re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip())


def _merge(base: Any, override: Any) -> Any:
    """Helm's ``mergeOverwrite``: maps merge key by key; lists and scalars are replaced."""
    if isinstance(base, dict) and isinstance(override, dict):
        return {**base, **{key: _merge(base.get(key), value) for key, value in override.items()}}
    return override


def test_every_config_key_is_documented_once() -> None:
    rows, reserved = _rows(), _reserved()
    assert not rows.keys() & reserved, "a key is both in a table and Reserved"
    assert rows.keys() | reserved == _fields().keys()


def test_documented_defaults_match_the_models() -> None:
    fields = _fields()
    for key, cell in _rows().items():
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


def test_recipes_in_order_keep_profiles_valid_and_connected() -> None:
    recipes = [yaml.safe_load(block) for block in re.findall(r"```yaml\n(.*?)```", GUIDE.read_text(), re.DOTALL)]
    harbor: dict[str, Any] = CHART_HARBOR
    for recipe in recipes:
        harbor = _merge(harbor, recipe["platformConfig"]["evals"]["runs"]["harbor"])
        config = HarborTaskJobsConfig.model_validate(harbor)
        for name, profile in config.sandbox_profiles.items():
            assert API_HOST in profile.agent_channel, f"{name} can't reach the gateway from no-network tasks"
            assert profile.agent_setup_egress, f"{name} can't install agents"
    assert config.default_sandbox_profile == "harbor-web"
    assert config.sandbox_profiles["harbor-web"].max_cpus == 8


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
