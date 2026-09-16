# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same Harbor run as ``run_harbor_gpa.py``, scored by TruLens's Agent GPA feedback functions.

TruLens is not a dependency of this repository; install it first::

    uv pip install trulens-core trulens-feedback trulens-providers-openai
    export NVIDIA_API_KEY=...
    uv run plugins/nemo-evaluator/examples/harbor_gpa/run_harbor_gpa_trulens.py
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from gpa_metrics import attach_gpa
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborAgentTaskRunner,
    HarborRuntimeConfig,
    discover_harbor_tasks,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig
from trulens_metrics import nvidia_provider, trulens_gpa_metrics

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b", help="Model the agent runs on.")
parser.add_argument("--judge-model", default="nvidia/nemotron-3.5-lightning-30b-a3b", help="Model TruLens judges with.")
parser.add_argument("--jobs-dir", type=Path, default=Path("./harbor-jobs"), help="Where Harbor writes results.")
parser.add_argument("--work-dir", type=Path, default=Path("./harbor-gpa-out"), help="Where the run bundle lands.")
args = parser.parse_args()

if not os.environ.get("NVIDIA_API_KEY"):
    raise SystemExit("NVIDIA_API_KEY is not set; both the agent and the judge read it from the environment.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# 1. The tasks, with TruLens's feedback functions wrapped as the nine GPA metrics.
tasks = attach_gpa(
    discover_harbor_tasks(Path(__file__).parent / "gpa_dataset"),
    trulens_gpa_metrics(nvidia_provider(args.judge_model)),
)

# 2. The agent: identical to run_harbor_gpa.py.
runner = HarborAgentTaskRunner(
    config=HarborRuntimeConfig(
        jobs_dir=args.jobs_dir,
        agent_import_path="nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent:NemoFabricAgent",
        agent_kwargs={
            "fabric_adapter_id": "nvidia.fabric.langchain.deepagents",
            "fabric_package": "nemo-fabric[deepagents,relay]==0.3.0b1",
            "fabric_workspace": "/app",
            "fabric_telemetry": "relay",
        },
        agent_model_name=args.model,
        agent_env_from_host=["NVIDIA_API_KEY"],
        n_concurrent_trials=3,
        agent_setup_timeout_multiplier=8.0,
        agent_timeout_multiplier=5.0,
        quiet=False,
    )
)

# 3. Run and score.
result = asyncio.run(
    AgentEvaluator().run(tasks=tasks, target=runner, config=AgentEvalRunConfig(work_dir=args.work_dir, parallelism=3))
)

# 4. Report.
print("\nAggregates:")
for aggregate in result.summary.scores.scores:
    mean = "unmeasured" if aggregate.mean is None else f"{aggregate.mean:.3f}"
    print(f"  {aggregate.name:<32} {mean:>10}  measured={aggregate.count} unmeasured={aggregate.nan_count}")

print("\nPer trial:")
for trial in result.trials:
    print(f"  {trial.task_id} ({trial.status.value})")
    for score in result.scores:
        if score.trial_id == trial.id:
            outputs = {o.name: o.value for o in score.outputs}
            reason = str(outputs.get("reason", ""))[:200]
            print(f"    {score.metric_type}: {outputs.get('score', outputs.get('reward', 'unmeasured'))}  {reason}")
