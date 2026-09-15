# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a NeMo Fabric deepagents harness on a Nemotron model inside Harbor.

Harbor owns the sandbox and the verifier; NeMo Fabric owns the agent harness. The bridge is
:class:`~nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent.NemoFabricAgent`, selected with
``agent_import_path`` and configured with ``agent_kwargs`` -- the same two fields a platform
``HarborRunnerTarget`` carries. The model API key never appears in the config: ``agent_env_from_host``
names it, Harbor resolves it from this process's environment when it creates the agent, and the job
directory's ``config.json`` records only ``${NVIDIA_API_KEY}``.

Requires ``harbor`` installed, a running Docker daemon, and the model provider's API key exported
(``NVIDIA_API_KEY`` from https://build.nvidia.com for the default model). Run it as a module from the
repository root::

    uv run python -m packages.nemo_evaluator_sdk.examples.harbor.run_fabric_deepagents_example
    uv run python -m packages.nemo_evaluator_sdk.examples.harbor.run_fabric_deepagents_example \\
        --model nvidia/nemotron-3.5-lightning-30b-a3b
    uv run python -m packages.nemo_evaluator_sdk.examples.harbor.run_fabric_deepagents_example \\
        --model openai/gpt-5.4   # forwards OPENAI_API_KEY instead
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent import DEFAULT_API_KEY_ENV
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborRuntimeConfig, run_harbor_eval
from pydantic import JsonValue

logger = logging.getLogger(__name__)

#: hello-world on a `python:3.12-slim` image: the Fabric agent installs itself into the task container.
FABRIC_HELLO_WORLD_DATASET_DIR = Path(__file__).resolve().parent / "fabric_hello_world_dataset"
NEMO_FABRIC_AGENT = "nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent:NemoFabricAgent"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def api_key_env_for(model: str, override: str | None) -> str:
    """The environment variable holding the credential for ``model``'s provider."""
    if override:
        return override
    provider = model.split("/", maxsplit=1)[0] if "/" in model else "openai"
    try:
        return DEFAULT_API_KEY_ENV[provider]
    except KeyError:
        raise SystemExit(f"no default credential variable for provider {provider!r}; pass --api-key-env") from None


async def _main(jobs_dir: Path, *, model: str, api_key_env: str, job_name: str | None) -> None:
    agent_kwargs: dict[str, JsonValue] = {
        "fabric_adapter_id": "nvidia.fabric.langchain.deepagents",
        "fabric_package": "nemo-fabric[deepagents]==0.3.0b1",
        # The task image's working directory; Fabric's default `/testbed` does not exist there.
        "fabric_workspace": "/app",
    }
    if api_key_env != DEFAULT_API_KEY_ENV.get(model.split("/", maxsplit=1)[0]):
        agent_kwargs["fabric_model_api_key_env"] = api_key_env
    config = HarborRuntimeConfig(
        jobs_dir=jobs_dir,
        job_name=job_name,
        agent_import_path=NEMO_FABRIC_AGENT,
        agent_kwargs=agent_kwargs,
        # `provider/model`: the provider picks the credential variable and endpoint (nvidia ->
        # NVIDIA_API_KEY, https://integrate.api.nvidia.com/v1) and the full id is kept.
        agent_model_name=model,
        agent_env_from_host=[api_key_env],
        n_concurrent_trials=1,
        # Installing Fabric and the deepagents harness in the container takes a few minutes the
        # first time; Harbor's default agent-setup timeout is tuned for prebuilt agents.
        agent_setup_timeout_multiplier=8.0,
        agent_timeout_multiplier=5.0,
        quiet=False,
    )
    result = await run_harbor_eval(config, FABRIC_HELLO_WORLD_DATASET_DIR)

    print(f"run_id: {result.run_id}  tasks: {result.summary.task_count}  trials: {result.summary.trial_count}")
    for aggregate in result.summary.scores.scores:
        print(f"  {aggregate.name}: mean={aggregate.mean}")
    for score in result.scores:
        reward = score.outputs[0].value if score.outputs else None
        print(f"  {score.task_id}: reward={reward} status={score.status.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs-dir", type=Path, default=Path("./harbor-jobs"), help="Where Harbor writes results.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="`provider/model` slug for the harness.")
    parser.add_argument("--job-name", default=None, help="Pin a job name to reuse its directory as a cache.")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable holding the model API key. Defaults per provider: "
        + ", ".join(f"{provider} -> {name}" for provider, name in DEFAULT_API_KEY_ENV.items()),
    )
    args = parser.parse_args()
    api_key_env = api_key_env_for(args.model, args.api_key_env)
    if not os.environ.get(api_key_env):
        raise SystemExit(f"{api_key_env} is not set; the agent forwards it into the task container.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_main(args.jobs_dir, model=args.model, api_key_env=api_key_env, job_name=args.job_name))


if __name__ == "__main__":
    main()
