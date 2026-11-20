# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_user_sim_plugin import scenarios
from nemo_user_sim_plugin.jobs.spec import SimulateRequest
from usersim.engine.core.seeds import load_seeds
from usersim.engine.probes.general_open_ended.prompts import user_agent_system_prompt
from usersim.engine.probes.safety_agentic import action_taxonomy
from usersim.engine.probes.safety_agentic import bank as agentic_bank

EXAMPLES = Path(__file__).parents[1] / "examples"
EXAMPLE = Path(__file__).parent / "fixtures" / "tool-requests"
TOPICS_EXAMPLE = EXAMPLES / "email-security-analyst"
MODEL_TARGET = {"kind": "model", "model": "m"}


@pytest.fixture(autouse=True)
def restore_usersim_paths():
    """``install`` points UserSim at the set through the environment; put it back after each test."""
    saved = {
        name: os.environ.get(name)
        for name in (scenarios._BANK_ENV, scenarios._TAXONOMY_ENV, scenarios.ASSET_PATH_ENV_VAR)
    }
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_a_tool_request_set_installs_and_usersim_reads_it(tmp_path):
    scenarios.install(scenarios.read_local(EXAMPLE), tmp_path)

    bank = agentic_bank.load_agentic_bank_default()
    taxonomy = action_taxonomy.load_action_taxonomy_default()
    assert bank.bank_id == "tool-requests"
    assert all(taxonomy.by_name(ar.tempted_action_name) for ar in bank.action_requests)


def test_a_sets_actions_join_usersims_own():
    actions = yaml.safe_dump({"entries": [{"name": "render_cycles_cpu"}, {"name": "send_email_on_behalf", "x": 1}]})

    merged = {entry["name"]: entry for entry in scenarios.merged_taxonomy(actions)["entries"]}

    assert "render_cycles_cpu" in merged and "read_public_information" in merged
    assert merged["send_email_on_behalf"] == {"name": "send_email_on_behalf", "x": 1}


def test_a_bank_naming_an_unknown_action_is_rejected(tmp_path):
    bank = (EXAMPLE / scenarios.BANK).read_bytes()

    with pytest.raises(scenarios.ScenarioError, match="rejected"):
        scenarios.install({scenarios.BANK: bank}, tmp_path)  # its tools' actions are in actions.yaml


def test_a_set_needs_a_bank(tmp_path):
    with pytest.raises(scenarios.ScenarioError, match="bank.yaml"):
        scenarios.install({scenarios.ACTIONS: b"entries: []"}, tmp_path)


def test_custom_scenarios_need_the_probe_that_runs_them():
    with pytest.raises(ValueError, match="safety_agentic or general_open_ended"):
        SimulateRequest(target=MODEL_TARGET, probes=["financial_services"], scenarios="my-set")

    for probe in ("safety_agentic", "general_open_ended"):
        assert SimulateRequest(target=MODEL_TARGET, probes=[probe], scenarios="my-set").scenarios == "my-set"


def test_a_set_names_its_own_outcomes():
    labels = scenarios.outcome_labels(scenarios.read_local(EXAMPLE))

    assert labels == {"should_call": "Picked the GPU tool", "should_not_call": "Picked the CPU tool"}
    assert scenarios.outcome_labels({scenarios.BANK: b""}) == {}


def test_outcome_labels_take_only_the_two_outcomes(tmp_path):
    files = {**scenarios.read_local(EXAMPLE), scenarios.OUTCOMES: b"passed: Did fine\n"}

    with pytest.raises(scenarios.ScenarioError, match="should_call and should_not_call"):
        scenarios.install(files, tmp_path)


TOPICS = yaml.safe_dump(
    {"set_id": "mail", "topics": [{"type": "suspicious_link", "description": "An email asking you to log in"}]}
).encode()


def test_a_topic_set_replaces_the_open_ended_topics(tmp_path):
    scenarios.install({scenarios.TOPICS: TOPICS}, tmp_path)

    for locale in ("en_US", "de_DE"):  # every locale draws the set's topics, not UserSim's translations
        assert load_seeds(locale, "general_open_ended", "topics") == [
            {"type": "suspicious_link", "description": "An email asking you to log in"}
        ]


def test_a_topic_set_binds_users_to_their_topic(tmp_path):
    # Otherwise a warehouse clerk given an email to triage asks about pallets.
    scenarios.install({scenarios.TOPICS: TOPICS}, tmp_path)

    prompt = user_agent_system_prompt()
    assert "background, and interests" not in prompt
    assert "Your persona shapes how you ask, not what you ask about." in prompt
    assert prompt.rstrip().endswith("</TOPIC>")


def test_a_set_runs_in_the_probes_its_files_feed():
    assert scenarios.probes({scenarios.TOPICS: TOPICS}) == ["general_open_ended"]
    assert scenarios.probes(scenarios.read_local(EXAMPLE)) == ["safety_agentic"]


def test_the_bank_id_or_the_topic_set_id_names_the_upload():
    assert scenarios.set_id(scenarios.read_local(EXAMPLE)) == "tool-requests"
    assert scenarios.set_id({scenarios.TOPICS: TOPICS}) == "mail"


@pytest.mark.parametrize("topics", [{"set_id": "x"}, {"set_id": "x", "topics": [{"type": "t"}]}, {"topics": []}])
def test_a_malformed_topic_set_is_rejected(tmp_path, topics):
    with pytest.raises(scenarios.ScenarioError):
        scenarios.install({scenarios.TOPICS: yaml.safe_dump(topics).encode()}, tmp_path)


def test_the_email_example_installs(tmp_path):
    found = scenarios.read_local(TOPICS_EXAMPLE)
    scenarios.install(found, tmp_path)
    assert scenarios.probes(found) == ["general_open_ended"]


class _Stored:
    def __init__(self, content: bytes):
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _Page:
    def __init__(self, names):
        self._names = names

    async def items(self):
        for name in self._names:
            yield SimpleNamespace(name=name)


class _Files:
    """Filesets as the Files service lists them, each holding a scenario set's files."""

    def __init__(self, filesets: dict[str, dict[str, bytes]]):
        self.filesets = filesets

    async def list_filesets(self, *, workspace):
        return _Page(list(self.filesets))

    async def download_file(self, *, workspace, name, path):
        if path not in self.filesets[name]:
            raise NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://helix")))
        return _Stored(self.filesets[name][path])


def test_studio_lists_the_uploaded_sets_with_the_probes_they_run_in():
    files = _Files(
        {
            "user-sim-scenarios-tool-requests": scenarios.read_local(EXAMPLE),
            "user-sim-scenarios-email-security-analyst": scenarios.read_local(TOPICS_EXAMPLE),
            "user-sim-previews": {"preview-1.json": b"[]"},
            "user-sim-scenarios-empty": {},
        }
    )

    found = asyncio.run(scenarios.list_sets(files, "default"))

    assert [(s.fileset, s.kind, s.probes, s.size) for s in found] == [
        ("user-sim-scenarios-email-security-analyst", "topics", ["general_open_ended"], 10),
        ("user-sim-scenarios-tool-requests", "tool requests", ["safety_agentic"], None),
    ]


def test_tax_season_sample_loads():
    folder = EXAMPLES / "tax-season"
    files = scenarios.read_local(folder)

    assert len(scenarios._topics(files["topics.yaml"])) == 10
    assert scenarios.set_id(files) == "tax-season"
    assert AgentConfig.model_validate(yaml.safe_load((folder / "agent.yaml").read_text())).name == "demo-tax-helper"
