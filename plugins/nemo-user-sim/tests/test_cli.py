# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
from types import SimpleNamespace

import httpx
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_user_sim_plugin import cli
from nemo_user_sim_plugin.cli import UserSimCLI, _print_report, interview, rerun_request, run_target
from nemo_user_sim_plugin.report import conversations, failure_rates, session_link
from typer.testing import CliRunner

RUNNER = CliRunner()


def test_run_needs_exactly_one_target_and_never_takes_a_base_url():
    app = UserSimCLI().get_cli()

    both = RUNNER.invoke(app, ["run", "--agent", "bank-support", "--model", "m"])
    help_text = RUNNER.invoke(app, ["run", "--help"]).output

    assert both.exit_code != 0 and "exactly one of --agent or --model" in both.output
    assert "--base-url" not in help_text


def test_report_rows_join_scores_and_rank_segments_by_failure():
    summary = {
        "conversations": [
            {"trajectory_id": "a", "user_interaction_style": "impatient", "status": "ok"},
            {"trajectory_id": "b", "user_interaction_style": "impatient", "status": "ok"},
            {"trajectory_id": "c", "user_interaction_style": "cooperative", "status": "ok"},
            {"trajectory_id": "d", "user_interaction_style": "cooperative", "status": "failed"},
        ]
    }
    scores = {"a": {"passed": False, "reason": "judge_fail"}, "b": {"passed": False}, "c": {"passed": True}}

    rows = conversations(summary, scores)

    assert [row["passed"] for row in rows] == [False, False, True, None]
    assert rows[0]["reason"] == "judge_fail"
    assert failure_rates(rows, "user_interaction_style") == {"impatient": 1.0, "cooperative": 0.0}


def test_transcript_links_only_for_conversations_published_to_intake():
    published = {"trajectory_id": "a", "publication": "published"}
    skipped = {"trajectory_id": "b", "publication": "skipped"}

    link = session_link(published, "http://helix", "default")
    assert link is not None and link.endswith("/intake/sessions/a")
    assert session_link(skipped, "http://helix", "default") is None


def test_the_interview_turns_answers_into_a_run(monkeypatch):
    answers = iter(["agent", "2", "3,5", "15"])
    monkeypatch.setattr(cli, "prompt", lambda *args, **kwargs: next(answers))

    choice = interview(agents=["legal-intake", "bank-support"], models=["m"])

    assert choice == {
        "agent": "bank-support",
        "model": None,
        "probes": ["financial_services", "safety_chat_pressure"],
        "users": 15,
    }


def test_a_run_the_judge_couldnt_score_says_why(capsys):
    results = {
        "summary": {"totals": {"conversations": 1, "completed": 1, "failed": 0}, "conversations": []},
        "scoring_error": {"message": "Scoring failed, so no conversation was judged. Model 'm' is down."},
    }

    class Client:
        def download_result(self, *, workspace, collection, job, name):
            if name not in results:
                raise NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://helix")))
            return io.BytesIO(json.dumps(results[name]).encode())

    _print_report(Client(), "default", "user-sim.simulate-x", "http://helix")  # ty: ignore[invalid-argument-type]

    assert "Scoring failed, so no conversation was judged. Model 'm' is down." in capsys.readouterr().out


def test_waiting_on_a_run_reports_progress_as_it_goes(monkeypatch, capsys):
    def status(state, step=None):
        steps = [SimpleNamespace(name=step, status=SimpleNamespace(value="active"))] if step else []
        return SimpleNamespace(data=lambda: SimpleNamespace(status=SimpleNamespace(value=state), steps=steps))

    polls = [
        status("pending"),
        status("active", "simulate"),
        status("active", "simulate"),
        status("active", "simulate"),
        status("active"),
        status("active", "evaluate"),
        status("completed"),
    ]
    summaries = [None, {"totals": {"conversations": 20, "completed": 6, "failed": 1}}]

    class Client:
        def get_job_status(self, *, workspace, collection, name):
            return polls.pop(0)

        def download_result(self, *, workspace, collection, job, name):
            summary = summaries.pop(0) if len(summaries) > 1 else summaries[0]
            if summary is None:
                raise NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://helix")))
            return io.BytesIO(json.dumps(summary).encode())

    monkeypatch.setattr(cli, "sleep", lambda seconds: None)

    ended = cli._wait(Client(), "default", "user-sim.simulate-x", 60, progress=True)  # ty: ignore[invalid-argument-type]

    assert ended == "completed"

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Waiting for the job to start.")
    assert lines[1].startswith("Simulating: starting the conversations.")
    assert lines[2].startswith("Simulating: 7 of 20 conversations finished, 1 couldn't run.")
    assert lines[3].startswith("Simulation done. The judges are scoring the conversations.")
    assert len(lines) == 4


def test_run_sends_use_agent_tools_for_an_agent():
    assert run_target("a", None, use_agent_tools=True) == {"kind": "agent", "agent": "a", "use_agent_tools": True}
    assert run_target("a", None, use_agent_tools=False) == {"kind": "agent", "agent": "a"}
    assert run_target(None, "m", use_agent_tools=False) == {"kind": "model", "model": "m"}


def test_run_rejects_use_agent_tools_with_a_model():
    app = UserSimCLI().get_cli()

    result = RUNNER.invoke(app, ["run", "--model", "m", "--use-agent-tools", "--no-wait"])

    assert result.exit_code != 0 and "--use-agent-tools needs --agent" in result.output


def test_a_menu_asks_again_for_a_number_it_doesnt_offer(monkeypatch, capsys):
    answers = iter(["0", "3", "x", "2"])
    monkeypatch.setattr(cli, "prompt", lambda *args, **kwargs: next(answers))

    assert cli._pick("Pick one", ["a", "b"]) == ["b"]
    assert capsys.readouterr().out.count("Enter a number from 1 to 2.") == 3


def test_a_rerun_copies_the_source_runs_settings():
    source = {
        "target": {"kind": "agent", "agent": "demo-tax-helper", "use_agent_tools": True},
        "probes": ["general_open_ended"],
        "users": 20,
        "locale": "en_US",
        "seed": 7,
        "max_turns": 5,
        "scenarios": "user-sim-scenarios-tax-season",
        "publish_to_intake": False,
        "preview": "abc",
        "exclude_personas": ["p9"],
        "assistant_model": "m",
        "fabric_config": {"name": "demo-tax-helper"},
    }

    assert rerun_request(source, "run-1", ["a", "b"]) == {
        "target": {"kind": "agent", "agent": "demo-tax-helper", "use_agent_tools": True},
        "probes": ["general_open_ended"],
        "locale": "en_US",
        "max_turns": 5,
        "scenarios": "user-sim-scenarios-tax-season",
        "publish_to_intake": False,  # a run kept out of Intake stays out when it's rerun
        "rerun": {"job": "run-1", "conversations": ["a", "b"]},
    }
