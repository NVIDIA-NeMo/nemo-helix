# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the fixed agent versions from agent.yaml, so each differs from v1 only by its fix."""

import copy
from pathlib import Path
from typing import Any

import yaml

AGENT_DIR = Path(__file__).resolve().parent.parent
VARIANTS_DIR = AGENT_DIR / "variants"
FIXED_TOOL_DEFINITION = "generate_data_designer_job_request.post-3857.json"


def _variant(base: dict[str, Any], *, name: str, description: str, system_prompt: str, tool_definition: str | None):
    config = copy.deepcopy(base)
    config["name"] = name
    config["description"] = description
    config["instructions"]["system"]["content"] = system_prompt
    if tool_definition is not None:
        config["mcp"]["servers"]["data_designer"]["args"].append(tool_definition)
    endpoint = config["telemetry"]["opentelemetry"]["endpoints"][0]
    endpoint["service_name"] = name
    endpoint["resource_attributes"]["gen_ai.agent.name"] = name
    return config


def main() -> None:
    base = yaml.safe_load((AGENT_DIR / "agent.yaml").read_text())
    models = (VARIANTS_DIR / "models.txt").read_text().strip()
    fixed_prompt = (VARIANTS_DIR / "system-prompt.post-3857.txt").read_text().strip()
    variants = {
        "agent-v2-prompt.yaml": _variant(
            base,
            name="data-designer-toolcall-v2-prompt",
            description="v1 plus a prompt that names the platform's only model and provider.",
            system_prompt=f"{base['instructions']['system']['content'].strip()}\n\n{models}",
            tool_definition=None,
        ),
        "agent-v3-schema.yaml": _variant(
            base,
            name="data-designer-toolcall-v3-schema",
            description="v2 plus the post-#3857 tool schema and system prompt.",
            system_prompt=f"{fixed_prompt}\n\n{models}",
            tool_definition=FIXED_TOOL_DEFINITION,
        ),
    }
    for filename, config in variants.items():
        (AGENT_DIR / filename).write_text(yaml.safe_dump(config, sort_keys=False, width=120, allow_unicode=True))
        print(f"wrote {filename}")


if __name__ == "__main__":
    main()
