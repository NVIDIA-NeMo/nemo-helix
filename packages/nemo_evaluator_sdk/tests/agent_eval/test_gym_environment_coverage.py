# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage of the Gym runner across a representative sample of Gym environments.

The other Gym tests (``test_gym_runtime.py``, ``test_gym_aggregate_scores.py``) replay recorded
fixtures through the adapter and never invoke the CLI, so they encode whatever ``mcqa`` happens to do
and keep passing for every other environment. This module goes the other way: it drives the real
``gym`` CLI against several environments chosen to span Gym's dependency and wiring categories.

Gym ships 105 ``resources_servers`` — 68 with two or fewer dependencies, 28 with multi-dependency CPU
stacks, 6 heavy/GPU, and 10 requiring Docker — and ``mcqa`` is the easiest of all of them, so "the
runner works" has only ever been established for the least demanding case.

**Two layers, deliberately.** ``gym env validate`` merges configs, flags, and overrides and reports
unset ``???`` values, bad paths, and dangling cross-references *without Ray, servers, a model
endpoint, or credentials*. It is fast, offline, and answers most of what this sweep wants to know, so
every environment gets a validate test. The rollout tests then start the servers and collect for
real, against a built-in stub endpoint (:class:`_StubPolicyServer`) so they need no model of their
own; set ``NEMO_GYM_POLICY_BASE_URL`` to run them against a real one instead.

That split matters because model wiring is not standardized. ``mcqa``/``gpqa_diamond``/
``wmt_translation`` reference a ``policy_model`` server fed by the global ``policy_base_url`` /
``policy_api_key`` / ``policy_model_name`` keys; ``gdpval`` wants a judge server named
``gdpval_judge_model`` that no shipped config defines; ``legal_agent_bench`` takes a raw
``judge_base_url``; and ``wmt_translation`` also loads a local COMET model that is not an endpoint at
all. Validate surfaces those differences as concrete errors instead of as a startup timeout.

Prerequisites, in the order the skips report them:

* **NeMo Gym installed in this environment** (``pip install nemo-gym``). Environments ship in the
  wheel, so no checkout is required.
* **The target environment's own dependencies** — each ``resources_server`` ships a
  ``requirements.txt``; install it from that directory so its ``-e nemo-gym[dev] @ ../../`` resolves.
* **Docker**, for environments whose agent works inside containers.

A model endpoint is *not* a prerequisite: the stub supplies one. ``NEMO_GYM_POLICY_BASE_URL`` is an
override for running against a real model, not a gate.

Tracked by AALGO-485; the CI and GPU-runner story is AALGO-494.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig, discover_gym_tasks
from nemo_evaluator_sdk.agent_eval.runtimes.gym.config import _flatten_overrides
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: Opt-in override pointing the rollout tests at a *real* model endpoint instead of the built-in
#: stub. Unset (the normal case) means the stub serves the rollouts, which is what lets these tests
#: run unattended. Gym reads its own credentials from a gitignored ``env.yaml``; this only decides
#: which endpoint the runner is configured against.
POLICY_BASE_URL_ENV = "NEMO_GYM_POLICY_BASE_URL"

_DEFAULT_AGENT = "simple_agent"
_DEFAULT_AGENT_CONFIG = "responses_api_agents/simple_agent/configs/simple_agent.yaml"

#: Keep runs small — this establishes that an environment works at all, not how well it scores.
_TASK_LIMIT = 2

#: What the stub answers when a case defines no gradeable answer. Deliberately not empty: an empty
#: response is indistinguishable from an agent that never ran (see `_agent_never_ran`), so a canned
#: answer keeps "the plumbing works" separable from "the agent produced nothing".
_CANNED_ANSWER = "This is a stub response from the NeMo Helix test suite."

#: Stands in for a per-run absolute directory inside a case's `hydra_params`. A committed
#: absolute path would be wrong on every machine, so the run substitutes a real one.
_ABS_ASSETS_PLACEHOLDER = "_ABS_ASSETS"


class _StubPolicyServer:
    """An OpenAI-compatible ``/v1/chat/completions`` endpoint that answers from a prompt-keyed map.

    **Why a stub at all.** Collecting rollouts needs a reachable model, and requiring one turned the
    whole rollout half of this module into a permanent skip — the runner's `validate -> env start ->
    eval run` sequence had never executed end to end. A stub removes the only prerequisite these
    tests could not satisfy for themselves.

    **Why prompt-keyed rather than a canned reply.** The oracle answers each prompt with *that row's*
    own expected answer, so a passing run means every task received the answer belonging to it. Gym
    attributes rollouts back to tasks through an index this runner assigns, and an off-by-one there
    would still produce a full set of trials — just with the answers paired to the wrong tasks. Under
    a canned reply that bug scores identically to a correct run; under the oracle every reward drops
    to 0. A queue-based mock cannot express this either, since rollouts are collected concurrently
    and arrival order is not the dataset order.

    Stdlib rather than ``pytest-httpserver``: Gym reaches this endpoint from *subprocesses*, so it
    has to be a real socket, and `packages/nemo_evaluator_sdk` does not depend on `nhx-testing`.
    ``plugins/nemo-guardrails/tests/unit/benchmarks/test_shim.py`` sets the same precedent.
    """

    def __init__(self) -> None:
        self._answers: dict[str, str] = {}
        self._unmatched: list[str] = []
        self.request_count = 0
        server_self = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    body = {}
                content = server_self._answer_for(body)
                payload = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model") or "stub-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    # Non-zero on purpose: the runner reads token usage to tell a real attempt from
                    # an agent that never called the model.
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base signature
                """Silence per-request logging to stderr; failures are asserted on, not read."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @staticmethod
    def _prompt_text(body: Mapping[str, Any]) -> str:
        """Every message's text, concatenated — matching is against what the model actually saw."""
        parts: list[str] = []
        for message in body.get("messages") or ():
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, Sequence):
                parts.extend(
                    block.get("text", "") for block in content if isinstance(block, Mapping) and block.get("text")
                )
        return "\n".join(parts)

    def _answer_for(self, body: Mapping[str, Any]) -> str:
        self.request_count += 1
        text = self._prompt_text(body)
        for needle, answer in self._answers.items():
            if needle in text:
                return answer
        # Recorded rather than raised: a handler exception would surface as an opaque 500 inside
        # Gym. The test asserts on this list, which names what went unmatched.
        self._unmatched.append(text[-200:])
        return _CANNED_ANSWER

    def expect(self, prompt_fragment: str, answer: str) -> None:
        """Answer any prompt containing ``prompt_fragment`` with ``answer``."""
        self._answers[prompt_fragment] = answer

    @property
    def unmatched(self) -> list[str]:
        return list(self._unmatched)

    def __enter__(self) -> _StubPolicyServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"


@dataclass(frozen=True)
class GymEnvironmentCase:
    """One environment in the sample, with the prerequisites it adds beyond an installed Gym."""

    resources_server: str
    category: str
    #: Why this environment is in the sample rather than another from the same category.
    rationale: str
    needs_docker: bool = False
    needs_gpu: bool = False
    agent: str = _DEFAULT_AGENT
    agent_config: str = _DEFAULT_AGENT_CONFIG
    #: Mirrors ``GymRuntimeConfig.bind_resources_server``. False where the environment registers its
    #: resources-server under a name other than its own, so the caller must bind it explicitly.
    bind_resources_server: bool = True
    #: Nested config overrides this environment needs beyond the runner's automatic binding. Empty
    #: where the standard `policy_model` wiring is enough; populated as the sweep discovers what an
    #: environment actually requires.
    hydra_params: Mapping[str, Any] = field(default_factory=dict)
    #: Row field holding this environment's ground truth, and how it wants an answer spelled. When
    #: set, the stub answers each prompt with that row's own correct answer and the rollout test can
    #: assert a perfect score — which is what makes per-task attribution testable (see
    #: :class:`_StubPolicyServer`). Left None where no answer is mechanically derivable (a judged
    #: rubric, a Docker-executed verifier), and those cases assert only that trials completed.
    answer_key: str | None = None
    answer_format: Callable[[str], str] | None = None
    #: Startup budget when the runner's 240s default is not enough. Set per environment rather than
    #: raised globally: a long default would turn a genuinely wedged server into a long wait for
    #: every environment, and the environments that need it are identifiable up front.
    startup_timeout_s: float | None = None

    @property
    def id(self) -> str:
        return f"{self.resources_server}[{self.category}]"


CASES: tuple[GymEnvironmentCase, ...] = (
    GymEnvironmentCase(
        resources_server="mcqa",
        category="baseline",
        rationale=(
            "The environment every existing test is built on. Included so a failure elsewhere is "
            "attributable to that environment rather than to the runner or the local setup."
        ),
        # Graded `strict_single_letter_boxed`, per the row's own `grading_mode`.
        answer_key="expected_answer",
        answer_format=lambda letter: f"\\boxed{{{letter}}}",
    ),
    GymEnvironmentCase(
        resources_server="gpqa_diamond",
        category="tiny",
        rationale=(
            "Control. Same multiple-choice shape as mcqa with comparably small dependencies, so a "
            "failure here means we encoded mcqa's specifics rather than the category's."
        ),
        # Same multiple-choice shape as mcqa, different spelling — its prompt asks for
        # `Answer: LETTER`, not a boxed letter. Encoding both is the point of the control.
        answer_key="expected_answer",
        answer_format=lambda letter: f"Answer: {letter}",
    ),
    GymEnvironmentCase(
        resources_server="gdpval",
        category="heavy-cpu-judge",
        rationale=(
            "Seven dependencies including binary wheels (PyMuPDF, pdfminer.six, python-docx), no "
            "GPU, and it ships a prompt config so it exercises the data-prep path mcqa skips. Also "
            "the clearest case of model wiring that is not standardized — see the overrides below."
        ),
        # Two Gym conventions this environment does not follow, both of which the caller has to know
        # about and neither of which is discoverable without reading its YAML or failing:
        #
        # 1. Its resources-server is registered as `gdpval_resources_server`, not `gdpval`. The
        #    runner's `bind_resources_server` binds the *environment* name (the simple_agent
        #    convention), so it must be disabled and bound by hand here.
        # 2. It references a judge model server named `gdpval_judge_model` that no shipped config
        #    defines — `responses_api_models/` provides model *types*, not this block — so the caller
        #    must construct it.
        #
        # We deliberately do not paper over either in the runner: they are Gym's inconsistencies, and
        # hiding them behind guesswork would make the runner wrong for environments that follow the
        # convention. Recorded here so the cost to a caller is visible.
        bind_resources_server=False,
        hydra_params={
            "simple_agent": {
                "responses_api_agents": {"simple_agent": {"resources_server": {"name": "gdpval_resources_server"}}}
            },
            "gdpval_judge_model": {
                "responses_api_models": {
                    "inference_provider": {
                        "entrypoint": "app.py",
                        # Interpolations, not literals: the judge shares the policy endpoint rather
                        # than needing a second one configured.
                        "base_url": "${policy_base_url}",
                        "api_key": "${policy_api_key}",
                        "model": "${policy_model_name}",
                    }
                }
            },
        },
    ),
    GymEnvironmentCase(
        resources_server="legal_agent_bench",
        category="docker",
        rationale=(
            "Harbor executes the task-local verifier inside Docker; needs a running daemon, ~10 GB "
            "of space, and a multi-minute first image build. Takes a raw `judge_base_url` rather "
            "than a model-server reference. Also runnable through the SDK's Harbor and Fabric "
            "examples, so it is the one environment comparable across three runners."
        ),
        needs_docker=True,
        # Measured: ~8 minutes from a cold cache. Startup installs this environment's dependencies
        # and then prepares the Harbor task tree (`ensure_assets` + `hydrate_runtime_tasks`) before
        # the server reports ready, which the 240s default does not cover. Observed as exactly the
        # timeout this change made legible: "3 of 4 server(s) started; still waiting on:
        # legal_agent_bench".
        startup_timeout_s=1800.0,
        # LAB's three asset directories default to paths *relative to the working directory*. The
        # resources-server prepares the Harbor task tree under them and the agent reads it back via
        # `local_dataset_path: ${...harbor_tasks_dir}` — the same key, so one override moves both
        # sides — but a relative value only resolves if every server shares the cwd the data was
        # written under, which is not guaranteed. Left relative, the agent dies with
        # `FileNotFoundError: resources_servers/legal_agent_bench/data/runtime/harbor_tasks/...`,
        # returns an empty response, and Gym records a rollout scoring 0.0 with no failure — a
        # silent zero. Absolute paths remove the assumption entirely.
        #
        # `_ABS_ASSETS` is a placeholder: the rollout test substitutes a real tmp_path, since a
        # committed absolute path would be wrong on every machine.
        hydra_params={
            "legal_agent_bench": {
                "resources_servers": {
                    "legal_agent_bench": {
                        "harbor_tasks_dir": f"{_ABS_ASSETS_PLACEHOLDER}/runtime/harbor_tasks/legal_agent_bench",
                        "harbor_tasks_cache_dir": f"{_ABS_ASSETS_PLACEHOLDER}/cache/harbor_tasks/legal_agent_bench",
                        "harness_skills_dir": f"{_ABS_ASSETS_PLACEHOLDER}/cache/harness/skills",
                    }
                }
            }
        },
    ),
    GymEnvironmentCase(
        resources_server="wmt_translation",
        category="gpu",
        rationale=(
            "torch, torchvision, and unbabel-comet, plus a local COMET model (Unbabel/XCOMET-XXL) "
            "that is not an endpoint at all. Expected to fail without a GPU — the point of running "
            "it anyway is to check the failure is legible."
        ),
        needs_gpu=True,
    ),
)


def _gym_cli() -> str:
    """The `gym` CLI, or skip. Everything here needs it; nothing here needs a checkout."""
    resolved = shutil.which("gym")
    if resolved is None:
        pytest.skip("NeMo Gym is not installed in this environment (`pip install nemo-gym`)")
    return resolved


def _bounded_probe(argv: list[str]) -> bool:
    """True when the command exists and exits zero. Bounded: a wedged daemon must not stall the run."""
    if shutil.which(argv[0]) is None:
        return False
    try:
        return subprocess.run(argv, capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _environment_dir(gym: str, case: GymEnvironmentCase) -> Path:
    """Locate an environment's directory by asking the CLI, or skip.

    Deliberately not ``importlib.resources``: nemo-helix bans Ray by constraint
    (``"ray; sys_platform == 'never'"`` in the root pyproject, for an unfixed CVE) while Gym imports
    Ray at module load, so Gym cannot be installed in this SDK's environment. It lives in its own,
    reachable only through the ``gym`` binary on PATH — which means ``resources_servers.<name>`` is
    not importable here even when the environment is perfectly available.

    ``gym list resources-servers <name>`` reports the environment's config path and exits non-zero
    for one it cannot resolve, so it answers both questions across the venv boundary.
    """
    # `gym list` is a Hydra entry point too, so it also wants somewhere to put its run directory —
    # without this it drops an `outputs/<date>/<time>/` into whatever directory pytest ran from.
    with tempfile.TemporaryDirectory(prefix="gym-list-hydra-") as hydra_dir:
        proc = subprocess.run(
            [gym, "list", "resources-servers", case.resources_server, f"hydra.run.dir={hydra_dir}"],
            capture_output=True,
            text=True,
            timeout=60,
        )

    if proc.returncode != 0:
        pytest.skip(
            f"Gym cannot resolve resources-server {case.resources_server!r}; install its own "
            "requirements.txt from the environment's directory (its `-e nemo-gym[dev] @ ../../` is "
            "relative to that directory, so it must be installed from there)"
        )
    config_line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("config:")), None)
    if config_line is None:
        pytest.skip(f"`gym list resources-servers {case.resources_server}` reported no config path to locate it by")
    # <env>/configs/<env>.yaml -> <env>
    return Path(config_line.removeprefix("config:").strip()).parent.parent


def _require_environment(gym: str, case: GymEnvironmentCase) -> Path:
    """Skip with one specific reason unless this case's prerequisites are present.

    The reasons are the deliverable: ``pytest -ra`` over this module is the coverage map, so each one
    names the single missing thing rather than a generic "prerequisites not met".
    """
    environment_dir = _environment_dir(gym, case)

    if case.needs_docker and not _bounded_probe(["docker", "info"]):
        pytest.skip(f"{case.resources_server} runs its verifier in Docker; no working daemon found")

    if case.needs_gpu and not _bounded_probe(["nvidia-smi"]):
        pytest.skip(f"{case.resources_server} scores with GPU models; no working nvidia-smi found")

    return environment_dir


def _selection(case: GymEnvironmentCase, hydra_dir: Path, hydra_params: Mapping[str, Any] | None = None) -> list[str]:
    """The selection arguments the runner passes to both `gym env validate` and `gym env start`.

    ``hydra.run.dir`` mirrors what the runner does: Gym is a Hydra app and writes a timestamped run
    directory per invocation, defaulting to ``outputs/`` under the current directory. Left alone, a
    sweep drops one of those into the repo for every environment it checks.
    """
    argv = [
        "--config",
        case.agent_config,
        "--model-type",
        "inference_provider",
        "--resources-server",
        case.resources_server,
    ]
    if case.bind_resources_server:
        argv.append(f"+{case.agent}.responses_api_agents.{case.agent}.resources_server.name={case.resources_server}")
    # Resolved overrides when the caller has them, so validate checks the *same* values the rollout
    # would run — an unresolved `_ABS_ASSETS` placeholder would validate a config nothing ever uses.
    argv.extend(_flatten_overrides(case.hydra_params if hydra_params is None else hydra_params))
    argv.append(f"hydra.run.dir={hydra_dir}")
    return argv


def _task_prompt(task: AgentEvalTask) -> str:
    """The prompt text Gym will send for a task, recovered from the source row we materialize.

    Read from ``inputs['gym_row']`` rather than ``intent``: for Gym rows ``intent`` is a dataset
    label (``"Gym row from example.jsonl"``), identical across every row in a file, so keying an
    oracle on it would collapse all tasks onto one entry.
    """
    row = task.inputs.get("gym_row") or {}
    parts: list[str] = []
    for item in row.get("input") or ():
        content = item.get("content") if isinstance(item, Mapping) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, Sequence):
            parts.extend(block.get("text", "") for block in content if isinstance(block, Mapping) and block.get("text"))
    return "\n".join(parts)


#: How much of a prompt's tail identifies it. The head is shared boilerplate ("answer with a letter
#: inside \boxed{}"), so a prefix would collide across rows; the tail carries the question and its
#: options.
_PROMPT_KEY_CHARS = 160


def _prime_stub(stub: _StubPolicyServer, case: GymEnvironmentCase, tasks: Sequence[AgentEvalTask]) -> None:
    """Teach the stub each task's own correct answer, for cases where one is derivable."""
    if case.answer_key is None or case.answer_format is None:
        return
    keys: set[str] = set()
    for task in tasks:
        expected = task.metadata.get("gym_row_extras", {}).get(case.answer_key)
        if expected is None:
            pytest.skip(f"{case.resources_server}: rows carry no {case.answer_key!r} to build an oracle from")
        key = _task_prompt(task)[-_PROMPT_KEY_CHARS:]
        assert key, f"{case.resources_server}: recovered no prompt text for task {task.id}"
        keys.add(key)
        stub.expect(key, case.answer_format(str(expected)))
    # A collision would silently answer one task with another's ground truth, which is precisely the
    # bug this oracle exists to detect — so it must not be able to cause a false pass.
    assert len(keys) == len(tasks), (
        f"{case.resources_server}: prompt tails collide across tasks ({len(keys)} keys for "
        f"{len(tasks)} tasks); raise _PROMPT_KEY_CHARS"
    )


def _case_overrides(case: GymEnvironmentCase, tmp_path: Path) -> dict[str, Any]:
    """The case's own overrides, with the `_ABS_ASSETS` placeholder resolved to a real directory.

    `legal_agent_bench`'s asset directories have to be absolute (see its case comment), but the
    absolute path is per-run, so the table carries a placeholder the run substitutes.
    """
    assets = tmp_path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(case.hydra_params).replace(_ABS_ASSETS_PLACEHOLDER, str(assets))
    return json.loads(rendered)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_gym_environment_config_validates(case: GymEnvironmentCase, tmp_path: Path) -> None:
    """`gym env validate` accepts the config the runner would run.

    No Ray, no servers, no model endpoint, no credentials — so this runs anywhere Gym is installed
    and is where most of the sweep's signal comes from. A failure here is a concrete, readable
    report: an unset ``???``, a bad path, or a cross-reference to a server that is not defined.
    """
    gym = _gym_cli()
    # Only the environment itself is required. Docker and GPU are *execution* prerequisites, and
    # gating on them here would throw away the coverage this test exists for — `wmt_translation`'s
    # config would go unvalidated on every machine without an NVIDIA card, which is all of them
    # until AALGO-494 lands the GPU runner.
    _environment_dir(gym, case)

    proc = subprocess.run(
        [gym, "env", "validate", *_selection(case, tmp_path / "hydra", _case_overrides(case, tmp_path))],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"{case.resources_server}: `gym env validate` rejected the runner's config.\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
@pytest.mark.asyncio
async def test_gym_environment_runs_end_to_end(case: GymEnvironmentCase, tmp_path: Path) -> None:
    """Collect real rollouts and confirm the runner turns them into scored trials.

    Deliberately shallow: this establishes that a category works, not how well it scores. Reward
    values are Gym's own and are not asserted on — that would be testing Gym, not the runner.
    """
    gym = _gym_cli()
    environment_dir = _require_environment(gym, case)

    dataset = environment_dir / "data" / "example.jsonl"
    if not dataset.exists():
        pytest.skip(f"{case.resources_server} ships no data/example.jsonl at {dataset}")

    tasks = discover_gym_tasks(dataset)
    assert tasks, f"{case.resources_server}: discover_gym_tasks found no rows in {dataset}"
    tasks = tasks[:_TASK_LIMIT]

    with _StubPolicyServer() as stub:
        # An explicitly configured endpoint wins, so this suite can be pointed at a real model to
        # produce meaningful scores. Unset, the stub serves the rollouts and the run needs nothing
        # beyond an installed Gym.
        policy_base_url = os.environ.get(POLICY_BASE_URL_ENV) or stub.base_url
        using_stub = policy_base_url == stub.base_url
        if using_stub:
            _prime_stub(stub, case, tasks)

        runner = GymAgentTaskRunner(
            config=GymRuntimeConfig(
                agent=case.agent,
                agent_config=case.agent_config,
                resources_server=case.resources_server,
                num_repeats=1,
                bind_resources_server=case.bind_resources_server,
                # Omit rather than pass None, so the runner's own default stays the single source of
                # truth for every environment that does not need more.
                **({"startup_timeout_s": case.startup_timeout_s} if case.startup_timeout_s else {}),
                hydra_params={
                    "policy_base_url": policy_base_url,
                    "policy_api_key": "stub-key",
                    "policy_model_name": "stub-model",
                    **_case_overrides(case, tmp_path),
                },
            )
        )

        result = await AgentEvaluator().run(
            tasks=tasks,
            target=runner,
            config=AgentEvalRunConfig(work_dir=tmp_path / "gym_run", parallelism=1),
        )

    assert result.summary.task_count == len(tasks)
    assert len(result.trials) == len(tasks)

    failed = [trial for trial in result.trials if trial.status is not AgentEvalTrialStatus.COMPLETED]
    assert not failed, f"{case.resources_server}: trials did not complete: {[(t.task_id, t.status) for t in failed]}"

    assert runner.runner_info().name == "gym"
    assert result.scores, f"{case.resources_server}: no metric scores were produced"

    if not using_stub:
        return
    # The model was genuinely called. Guards against a rollout path that "succeeds" without ever
    # reaching the endpoint — the legal_agent_bench failure mode, where empty responses scored 0.0
    # and read as a normal run.
    assert stub.request_count > 0, f"{case.resources_server}: the stub was never called, so no agent ran"
    if case.answer_key is None:
        return
    # Every prompt matched its own row, and every row therefore scored. A rollout attributed to the
    # wrong task would have received another row's answer and scored 0.
    assert not stub.unmatched, (
        f"{case.resources_server}: {len(stub.unmatched)} prompt(s) matched no task; the runner sent "
        f"content the oracle did not recognise: {stub.unmatched}"
    )
    rewards = [trial.metadata.get("reward") for trial in result.trials]
    assert all(reward == 1.0 for reward in rewards), (
        f"{case.resources_server}: every task was answered with its own ground truth, so each should "
        f"score 1.0; got {rewards}. Unequal rewards here mean rollouts were attributed to the wrong "
        "tasks rather than that the model was wrong."
    )
