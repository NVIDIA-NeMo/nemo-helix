# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract continuity: compile_automodel_config import path and optional snapshot check."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_DIR = Path(__file__).resolve().parent / "contract"
GENERATE_SCRIPT = CONTRACT_DIR / "generate_configs.py"

# v1 excludes embedding SFT until product expands scope.
EMBEDDING_CONFIG_STEMS = {"embed_1b_lora", "embed_1b_full_sft"}


@pytest.mark.skipif(not CONTRACT_DIR.is_dir(), reason="contract fixtures not in tree")
def test_generate_configs_import_path() -> None:
    """generate_configs.py must import compile_automodel_config from backends.config."""
    text = GENERATE_SCRIPT.read_text()
    assert "backends.config import compile_automodel_config" in text
    assert "backends.automodel.config" not in text


@pytest.mark.skipif(not CONTRACT_DIR.is_dir(), reason="contract fixtures not in tree")
@pytest.mark.parametrize(
    "config_name",
    [
        "llama_3_2_1b_lora",
        "llama_3_2_1b_lora_packing",
        "nemotron_nano_lora_packing",
    ],
)
def test_contract_input_parses_as_training_step_config(config_name: str) -> None:
    from nhx.automodel.tasks.training.schemas import TrainingStepConfig

    input_path = CONTRACT_DIR / "input_configs" / "llama-3.2-1b" / f"{config_name}.json"
    if config_name.startswith("nemotron"):
        input_path = CONTRACT_DIR / "input_configs" / "nemotron-nano" / f"{config_name}.json"
    if not input_path.exists():
        pytest.skip(f"missing {input_path}")

    raw = json.loads(input_path.read_text())
    raw.pop("backend", None)
    TrainingStepConfig.model_validate(raw)


@pytest.mark.skipif(not CONTRACT_DIR.is_dir(), reason="contract fixtures not in tree")
def test_contract_output_configs_up_to_date_excluding_embedding() -> None:
    """Run generate_configs --check when nemo_automodel is available in the environment."""
    pytest.importorskip("nemo_automodel")
    if not GENERATE_SCRIPT.is_file():
        pytest.skip("generate_configs.py missing")

    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "services" / "automodel" / "src")

    result = subprocess.run(
        [sys.executable, str(GENERATE_SCRIPT), "--check", "--exclude", ",".join(sorted(EMBEDDING_CONFIG_STEMS))],
        cwd=CONTRACT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        combined = result.stdout + result.stderr
        # The embedding configs are excluded by the generator itself, above, so a
        # failure reaching here is a real one. It used to be inferred instead --
        # any output mentioning an excluded stem skipped the whole check, which
        # meant a change touching every config switched the guard off silently.
        if "nemo_automodel" in combined and "ModuleNotFoundError" in combined:
            pytest.skip("nemo_automodel not installed in test env (run in training image CI)")
        pytest.fail(f"contract configs out of date:\n{combined}")


def test_the_check_excludes_configs_rather_than_forgiving_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real mismatch must fail even when an excluded config mismatches too.

    This is the guard that was switched off. The exclusion used to be inferred
    from the *output*: any run whose failure text mentioned an embedding config
    skipped entirely. Adding a key to every compiled config made every config
    mismatch, so the text always mentioned one, so the check silently stopped
    running -- with no failure and no skip message anyone would look twice at.

    Stubs go in via monkeypatch so they are torn down with the test; a
    module-scope `nemo_automodel` stub outlives the file that installed it and
    leaks into whatever else shares the xdist worker.
    """
    import importlib.util
    from unittest.mock import MagicMock

    for name in ("nemo_automodel", "nemo_automodel._transformers", "nemo_automodel._transformers.registry"):
        monkeypatch.setitem(sys.modules, name, MagicMock())

    spec = importlib.util.spec_from_file_location("_contract_gen_under_test", GENERATE_SCRIPT)
    assert spec and spec.loader
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    excluded, real = "embed_1b_lora", "gpt_oss_lora"
    monkeypatch.setattr(
        gen,
        "_discover_model_groups",
        lambda: [("g", [tmp_path / f"{excluded}.json", tmp_path / f"{real}.json"])],
    )
    monkeypatch.setattr(
        gen,
        "_process_model_group",
        lambda group, paths, write=False: [(tmp_path / f"{p.stem}.yaml", {"x": 1}) for p in paths],
    )
    monkeypatch.setattr(gen, "_dump_yaml", lambda compiled: "MISMATCH\n")

    with pytest.raises(SystemExit):
        gen._check_all({excluded})

    # ...and excluding everything that can mismatch leaves nothing to fail on,
    # which is what makes the assertion above about the exclusion and not about
    # the stub always failing.
    gen._check_all({excluded, real})
