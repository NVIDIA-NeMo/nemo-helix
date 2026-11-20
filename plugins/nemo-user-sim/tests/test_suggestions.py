# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_agents_plugin.ethos import ETHOS_SECTION_TITLES
from nemo_user_sim_plugin.suggestions import agent_profile, suggest_probes

CONFIG = {
    "name": "bank-support",
    "instructions": {"system": {"content": "You are a bank's support agent. Verify identity before moving money."}},
    "mcp": {"servers": {"core-banking": {"transport": "streamable-http", "url": "http://bank/mcp"}}},
}
SECTIONS = {title: "_(none)_" for title in ETHOS_SECTION_TITLES} | {
    "Role": "Retail banking support.",
    "Constraints": "Never move money without verified identity.",
    "Vision": "Expand to mortgages.",
}
ETHOS = "---\nname: bank-support\ncreated_timestamp: 2026-11-01T00:00:00Z\nauthor: ykagal\n---\n" + "".join(
    f"\n## {title}\n{body}\n" for title, body in SECTIONS.items()
)


def test_the_profile_carries_what_the_agent_does_and_must_not_do():
    profile = agent_profile(CONFIG, ETHOS)

    assert "Verify identity before moving money." in profile
    assert "core-banking" in profile
    assert "Never move money without verified identity." in profile
    assert "Expand to mortgages" not in profile  # sections about the future don't describe behavior


async def test_suggestions_keep_known_probes_with_reasons_and_never_come_back_empty():
    async def answer(prompt):
        return json.dumps(
            {
                "probes": [
                    {"name": "financial_services", "reason": "It moves money through banking tools."},
                    {"name": "made_up_probe", "reason": "?"},
                    {"name": "safety_agentic", "reason": "It acts with tools."},
                    {"name": "tool_calling", "reason": "Unsupported."},
                ]
            }
        )

    async def nonsense(prompt):
        return "not json"

    suggested = await suggest_probes("profile", answer)

    assert [probe["name"] for probe in suggested] == ["financial_services", "safety_agentic"]
    assert suggested[0]["reason"] == "It moves money through banking tools."
    assert [probe["name"] for probe in await suggest_probes("profile", nonsense)] == ["general_open_ended"]
