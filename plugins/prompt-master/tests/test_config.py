# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from prompt_master_plugin.config import PromptMasterConfigError, load_prompt_master_config


def test_prompt_master_example_is_valid() -> None:
    config_path = Path(__file__).parents[1] / "examples" / "prompt-master.yaml"

    config = load_prompt_master_config(config_path)

    assert config.model.provider == "nvidia"
    assert config.model.model == "nvidia-nemotron-3-5-lightning-30b-a3b"


def test_loads_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-master.yaml"
    config_path.write_text(
        """
model:
  provider: nvidia
  model: nvidia-nemotron-3-nano-30b-a3b
  temperature: 0.1
prompt_override: |
  Help users debug their Python code.
timeout_seconds: 90
""".lstrip(),
        encoding="utf-8",
    )

    config = load_prompt_master_config(config_path)

    assert config.model.provider == "nvidia"
    assert config.model.model == "nvidia-nemotron-3-nano-30b-a3b"
    assert config.model.temperature == 0.1
    assert config.prompt_override == "Help users debug their Python code.\n"
    assert config.timeout_seconds == 90


def test_rejects_empty_config(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-master.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(PromptMasterConfigError, match="root must be a mapping"):
        load_prompt_master_config(config_path)


def test_prompt_override_is_optional(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-master.yaml"
    config_path.write_text(
        """
model:
  provider: openai
  model: gpt-5.6
""".lstrip(),
        encoding="utf-8",
    )

    assert load_prompt_master_config(config_path).prompt_override is None


def test_rejects_blank_prompt_override(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-master.yaml"
    config_path.write_text(
        """
model:
  provider: openai
  model: gpt-5.6
prompt_override: "   "
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptMasterConfigError, match="prompt_override"):
        load_prompt_master_config(config_path)


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-master.yaml"
    config_path.write_text(
        """
model:
  provider: openai
  model: gpt-5.6
prompt_override: Be concise.
unexpected: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptMasterConfigError, match="unexpected"):
        load_prompt_master_config(config_path)


@pytest.mark.parametrize(
    "field",
    ["base_url: https://integrate.api.nvidia.com/v1", "api_key_env: NVIDIA_API_KEY"],
)
def test_rejects_a_model_endpoint_or_key(tmp_path: Path, field: str) -> None:
    """Inference always goes through the platform's Inference Gateway, so neither is configurable."""
    config_path = tmp_path / "prompt-master.yaml"
    config_path.write_text(
        f"""
model:
  provider: nvidia
  model: nvidia-nemotron-3-nano-30b-a3b
  {field}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptMasterConfigError, match=field.split(":")[0]):
        load_prompt_master_config(config_path)
