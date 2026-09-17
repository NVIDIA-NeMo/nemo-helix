# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the help RlContributor contributes to the customization CLI."""

from __future__ import annotations

import pytest
from nemo_rl_plugin.contributor import RlContributor


@pytest.fixture
def contributor() -> RlContributor:
    return RlContributor()


def test_cli_summary_states_what_it_trains_and_where_it_runs(contributor: RlContributor) -> None:
    summary = contributor.get_cli_summary()
    assert summary is not None
    assert "DPO" in summary.trains and "GRPO" in summary.trains
    assert "kubernetes_job backend only" in summary.runs_on
    assert summary.command == "nemo customization rl submit job.json"


def test_cli_summary_fits_the_rendered_width(contributor: RlContributor) -> None:
    """The router prints the blurb through an 80-column Rich console."""
    summary = contributor.get_cli_summary()
    assert summary is not None
    rendered = summary.render("rl")
    assert [line for line in rendered.splitlines() if len(line) > 80] == []


def test_backend_help_goes_deeper_than_the_top_level_blurb(contributor: RlContributor) -> None:
    summary = contributor.get_cli_summary()
    assert summary is not None
    blurb = summary.render(contributor.name)
    help_text = contributor.cli_help

    assert len(help_text) > len(blurb)
    # Payload and escape-hatch detail belongs on the backend, not in the overview.
    for detail in ("RlJobInput", "training.type", "finetuning_type", "explain"):
        assert detail in help_text, detail
        assert detail not in blurb, detail


def test_backend_help_names_the_backend_to_use_for_sft(contributor: RlContributor) -> None:
    """Someone who landed here wanting SFT needs to be sent somewhere, not just refused."""
    help_text = contributor.cli_help
    assert "kubernetes_job backend only" in help_text
    assert "use the automodel or unsloth backend" in help_text


def test_submit_help_explains_the_job_json(contributor: RlContributor) -> None:
    cli = contributor.get_cli()
    submit = next(cmd for cmd in cli.registered_commands if cmd.name == "submit")
    assert submit.help is not None
    assert "RlJobInput" in submit.help
    assert "nemo customization rl explain" in submit.help
