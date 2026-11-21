# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.config import Configuration
from nemo_user_sim_plugin.agents import AgentResolutionError
from nemo_user_sim_plugin.config import UserSimConfig
from nemo_user_sim_plugin.jobs import simulate
from nemo_user_sim_plugin.jobs._common import PROFILES
from nemo_user_sim_plugin.jobs.evaluate import agent_verdict, fully_scored, scoring_failure
from nemo_user_sim_plugin.jobs.simulate import Progress, SimulateJob, approved_rows, preview_rows, rerun_rows
from nemo_user_sim_plugin.jobs.spec import AgentTarget, RerunSource, SimulateRequest, SimulateSpec
from nemo_user_sim_plugin.personas import PersonasMissingError, kept_rows, persona_card, require_personas, top_up
from pydantic import ValidationError
from usersim.selection.engine import evaluate_row

FIXTURES = Path(__file__).parent / "fixtures"
FINISHED = json.loads((FIXTURES / "finished_row.json").read_text())
MODEL_TARGET = {"kind": "model", "model": "nvidia-nemotron-3-super-120b-a12b"}
SPEC = SimulateSpec(target=MODEL_TARGET, assistant_model="m")
SETTINGS = {"evaluator_model": {"extra_body": {"temperature": 0}}}


@pytest.fixture
def personas_dir(tmp_path):
    override = UserSimConfig(personas_dir=str(tmp_path), judge_model="judge-from-config", model_settings=SETTINGS)
    Configuration.set_override(override)
    yield tmp_path
    Configuration.clear_override(UserSimConfig)


async def test_simulate_compiles_to_simulate_then_evaluate_steps(personas_dir):
    spec = SimulateSpec(target=MODEL_TARGET, users=3, assistant_model="m")

    job = await SimulateJob.compile(
        workspace="default", spec=spec, entity_client=None, job_name="run-7", async_sdk=None
    )

    assert [step.name for step in job.steps] == ["simulate", "evaluate"]
    assert getattr(job.steps[0].executor, "command") == ["python", "-m", "nemo_user_sim_plugin.tasks.simulate"]
    assert job.steps[0].config["users"] == 3
    envs = [{variable.name: variable.value for variable in step.environment or []} for step in job.steps]
    assert envs[0]["NEMO_USER_SIM_RUN"] == "run-7"  # the simulate step tags its Intake traces with the run
    for env in envs:
        # Data Designer reads its asset path once, at import, and the jobs backend passes tasks only an
        # allowlist of the service's environment, so both travel in each step's environment.
        assert env["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(personas_dir / "managed-assets")
        assert env["NEMO_USER_SIM_JUDGE_MODEL"] == "judge-from-config"
        assert json.loads(env["NEMO_USER_SIM_MODEL_SETTINGS"] or "") == SETTINGS


@pytest.mark.parametrize(
    ("probes", "message"),
    [(["no_such_probe"], "Unknown probes"), (["tool_calling"], "aren't supported")],
)
def test_request_rejects_probes_it_cannot_run(probes, message):
    with pytest.raises(ValidationError, match=message):
        SimulateRequest(target=MODEL_TARGET, probes=probes)


async def test_unknown_agent_fails_at_submission_with_its_name():
    class NoAgents:
        async def get(self, entity_type, *, name, workspace):
            raise LookupError("not found")

    request = SimulateRequest(target={"kind": "agent", "agent": "bank-support"})

    with pytest.raises(AgentResolutionError, match="Agent 'bank-support' wasn't found"):
        await SimulateJob.to_spec(
            request, workspace="default", entity_client=NoAgents(), async_sdk=None, is_local=False
        )


def test_missing_personas_name_the_setup_command(personas_dir):
    with pytest.raises(PersonasMissingError, match="nemo user-sim setup --locale ja_JP"):
        require_personas("ja_JP")


def test_a_job_step_checks_the_personas_path_it_was_given(personas_dir, monkeypatch):
    monkeypatch.setenv("DATA_DESIGNER_MANAGED_ASSETS_PATH", str(personas_dir / "elsewhere"))

    with pytest.raises(PersonasMissingError, match="isn't installed"):
        require_personas("en_US")
    (personas_dir / "elsewhere" / "datasets").mkdir(parents=True)
    (personas_dir / "elsewhere" / "datasets" / "en_US.parquet").touch()
    require_personas("en_US")


async def test_progress_shows_finished_conversations_while_others_run():
    rows = [{"trajectory_id": "a", "persona_name": "Ana"}, {"trajectory_id": "b", "persona_name": "Ben"}]
    saved: dict = {}
    progress = Progress(SPEC, rows, save=saved.__setitem__)

    await progress.record(
        0, {"status": "ok", "row": FINISHED, "error": None, "tool_calls": 3, "own_tool_calls": ["c1"]}
    )
    await progress.flush()

    conversations = saved["summary"]["conversations"]
    assert [conversation["status"] for conversation in conversations] == ["ok", "running"]
    assert [conversation["own_tool_calls"] for conversation in conversations] == [["c1"], []]
    assert saved["summary"]["totals"] == {"conversations": 2, "completed": 1, "failed": 0, "running": 1}
    assert list(saved["transcripts"]) == ["a"]
    assert saved["transcripts"]["a"][0]["role"] == "user"


async def test_a_failed_progress_save_mid_run_doesnt_stop_the_run():
    def save(name, payload):
        raise OSError("files service unavailable")

    progress = Progress(SPEC, [{"trajectory_id": "a"}], save=save, interval=0)

    await progress.record(0, {"status": "ok", "row": FINISHED, "error": None, "tool_calls": 0})
    with pytest.raises(OSError):
        await progress.flush()


@pytest.mark.parametrize(
    ("failed_gates", "axis_scores", "expected"),
    [
        ([], {"helpfulness": 4.0}, {"passed": True, "reason": None}),
        (["helpfulness", "accuracy"], {"helpfulness": 2.0}, {"passed": False, "reason": "helpfulness, accuracy"}),
        # Simulator problems say nothing about the agent.
        (["health:n_user_role_violations"], {"helpfulness": 4.0}, {"passed": True, "reason": None}),
        (["no_eval"], {}, {"passed": None, "reason": None}),
        # A scorer that failed leaves its scores out, so the judges' other scores don't make a pass.
        (["agentic.boundary_held:scorer_error"], {"helpfulness": 5.0}, {"passed": None, "reason": None}),
        (["helpfulness", "agentic.boundary_held:scorer_error"], {"helpfulness": 2.0}, {"passed": False}),
        # UserSim stores both columns as JSON text.
        (
            '["health:n_user_role_violations", "helpfulness"]',
            '{"helpfulness": 2.0}',
            {"passed": False, "reason": "helpfulness"},
        ),
    ],
)
def test_agent_verdict_counts_only_the_judges_quality_axes(failed_gates, axis_scores, expected):
    verdict = agent_verdict({"selection_failed_gates": failed_gates, "selection_axis_scores": axis_scores})

    assert {key: verdict[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("failed_gates", "axis_scores", "scored"),
    [
        ([], {"helpfulness": 4.0}, True),
        (["tone"], {"helpfulness": 4.0, "tone": 2.0}, True),  # a low score is still a score
        ([], {}, False),  # the judge answered nothing
        (["agentic.boundary_held:scorer_error"], {"helpfulness": 5.0}, False),
    ],
)
def test_only_a_fully_scored_conversation_can_be_exported(failed_gates, axis_scores, scored):
    row = {"selection_failed_gates": json.dumps(failed_gates), "selection_axis_scores": json.dumps(axis_scores)}

    assert fully_scored(row) is scored


def test_a_score_computed_without_the_judge_doesnt_stand_in_for_the_judges():
    # Every judge call failed, so its axes are empty, while a rate UserSim computes itself still scored.
    judged = {"helpfulness": {"evaluator_model": {"score": 4}}}
    row = {
        "selection_failed_gates": "[]",
        "selection_axis_scores": json.dumps({"language.requested_language_match_rate": 1.0}),
    }

    assert fully_scored({**row, "assistant_eval": json.dumps({"axes": judged})})
    assert not fully_scored({**row, "assistant_eval": json.dumps({"axes": {"helpfulness": {}, "accuracy": {}}})})
    assert agent_verdict({**row, "assistant_eval": {"axes": {"helpfulness": {}}}})["passed"] is None


def test_a_run_drops_the_personas_a_reviewer_removed_in_the_preview():
    spec = SimulateSpec(target=MODEL_TARGET, assistant_model="m", preview="pv-1", exclude_personas=["p2"])
    rows = [{"persona_uuid": "p1"}, {"persona_uuid": "p2"}, {"persona_uuid": "p3"}]

    assert [row["persona_uuid"] for row in approved_rows(rows, spec)] == ["p1", "p3"]


async def test_removing_personas_needs_the_preview_that_drew_them():
    request = SimulateRequest(target=MODEL_TARGET, exclude_personas=["p2"])

    with pytest.raises(ValueError, match="preview"):
        await SimulateJob.to_spec(request, workspace="default", entity_client=None, async_sdk=None, is_local=False)


def test_stored_runs_from_before_previews_still_read_back():
    # Specs are re-validated when runs are listed, so a rule added later mustn't reject old ones.
    SimulateSpec(target=MODEL_TARGET, assistant_model="m", seed=31, exclude_personas=["p2"])


def test_a_refill_keeps_the_personas_left_and_adds_only_unseen_ones():
    kept = [{"persona_uuid": "p1"}, {"persona_uuid": "p3"}]
    drawn = [{"persona_uuid": "p3"}, {"persona_uuid": "p2"}, {"persona_uuid": "p4"}, {"persona_uuid": "p4"}]

    rows = top_up(kept, drawn, removed=["p2"])

    assert [row["persona_uuid"] for row in rows] == ["p1", "p3", "p4"]


def test_a_smaller_run_keeps_only_as_many_previewed_personas_as_it_asks_for():
    stored = [{"persona_uuid": "p1"}, {"persona_uuid": "p2"}, {"persona_uuid": "p3"}]

    assert [row["persona_uuid"] for row in kept_rows(stored, removed={"p1"}, users=1)] == ["p2"]


def test_two_runs_of_one_preview_get_their_own_conversation_ids():
    # Otherwise both publish to the same Intake sessions, and a review of one counts for the other.
    rows = preview_rows(SOURCE_ROWS, token="x1")

    assert [row["trajectory_id"] for row in rows] == ["a-px1", "b-px1", "c-px1"]
    assert rows[0] == {**SOURCE_ROWS[0], "trajectory_id": "a-px1"}


def test_a_preview_card_says_who_the_simulated_user_is():
    card = persona_card(json.loads((FIXTURES / "conversation_inputs.json").read_text()))

    assert card["persona_uuid"] == "311d2ca38da334eb"
    assert (card["name"], card["age"], card["style"], card["probe"]) == (
        "Mabel Espinoza",
        "65+",
        "confrontational",
        "financial_services",
    )
    assert card["traits"]["patience"] == 0.39


def test_a_judge_outage_reads_as_one_plain_sentence():
    boxed = (
        "🛑 Error generating dataset:   |----------\n"
        "  | Cause: Model 'nemotron' is currently experiencing internal server issues while running health checks.\n"
        "  | Solution: Try again in a few moments.\n"
        "  |----------"
    )

    assert scoring_failure(RuntimeError(boxed)) == (
        "Scoring failed, so no conversation was judged. Model 'nemotron' is currently experiencing internal "
        "server issues while running health checks."
    )
    assert scoring_failure(TimeoutError()) == "Scoring failed, so no conversation was judged. TimeoutError"


def _scored(boundary_held: int) -> dict:
    """An evaluation cell from a conversation the judges rated well, with the tool-use scorer's verdict."""
    judged = {axis: {"evaluator_model": {"score": 5}} for axis in ("helpfulness", "accuracy", "coherence", "safety")}
    verdict = {"scores": {"agentic.boundary_held": {"score": boundary_held}}, "status_proposal": boundary_held > 1}
    return {"axes": judged, "scorers": {"safety_agentic": verdict}}


@pytest.mark.parametrize("quality", ["strict", "generous"])
def test_calling_the_tool_the_agent_shouldnt_fails_however_well_it_reads(quality):
    def verdict(profile, boundary_held):
        return evaluate_row(
            outcome_raw={"status": "ok", "n_turns": 1},
            eval_cell_raw=_scored(boundary_held),
            persona_uuid="p1",
            trajectory_id="t1",
            num_turns_fallback=1,
            profile=profile,
        )

    assert not verdict(PROFILES[quality], 1).passed
    assert "agentic.boundary_held" in verdict(PROFILES[quality], 1).failed_gates
    assert verdict(PROFILES[quality], 5).passed


def test_agent_tools_are_off_by_default_and_only_an_agent_takes_them():
    assert AgentTarget(agent="a").use_agent_tools is False
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"target": {"kind": "model", "model": "m", "use_agent_tools": True}})


SOURCE_ROWS = [
    {"trajectory_id": "a", "persona_uuid": "p1", "topic": "which_credits"},
    {"trajectory_id": "b", "persona_uuid": "p2", "topic": "side_gig_income"},
    {"trajectory_id": "c", "persona_uuid": "p3", "topic": "moved_states"},
]


def test_a_rerun_keeps_each_person_and_topic_with_fresh_ids_in_the_order_asked():
    rows = rerun_rows(SOURCE_ROWS, ["c", "a"], token="x1")

    assert [row["trajectory_id"] for row in rows] == ["c-rx1", "a-rx1"]
    assert rows[1] == {**SOURCE_ROWS[0], "trajectory_id": "a-rx1"}


def test_a_rerun_of_a_conversation_the_run_doesnt_have_fails_plainly():
    with pytest.raises(ValueError, match="z"):
        rerun_rows(SOURCE_ROWS, ["a", "z"], token="x1")


def test_a_rerun_takes_no_more_conversations_than_a_run_takes_users():
    with pytest.raises(ValidationError):
        RerunSource(job="j", conversations=[str(number) for number in range(2001)])


def test_a_rerun_runs_one_user_per_conversation_and_cant_also_use_a_preview():
    rerun = {"job": "j", "conversations": ["a", "b"]}

    assert SimulateRequest.model_validate({"target": MODEL_TARGET, "users": 50, "rerun": rerun}).users == 2
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"target": MODEL_TARGET, "preview": "p", "rerun": rerun})


def test_rerunning_a_run_made_before_reruns_existed_fails_plainly(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://helix/results/inputs")))

    monkeypatch.setattr(simulate, "fetch_result", missing)

    with pytest.raises(ValueError, match="before reruns"):
        simulate.source_rows(None, "default", RerunSource(job="old-run", conversations=["a"]), tmp_path)  # ty: ignore[invalid-argument-type]


def test_rerunning_reads_the_rows_the_source_run_saved(monkeypatch, tmp_path):
    def saved(client, workspace, job, name, dest):
        assert (job, name) == ("run-1", "inputs")
        dest.write_text(json.dumps({"rows": SOURCE_ROWS}))
        return dest

    monkeypatch.setattr(simulate, "fetch_result", saved)

    rows = simulate.source_rows(None, "default", RerunSource(job="run-1", conversations=["b"]), tmp_path)  # ty: ignore[invalid-argument-type]

    assert rows == SOURCE_ROWS
