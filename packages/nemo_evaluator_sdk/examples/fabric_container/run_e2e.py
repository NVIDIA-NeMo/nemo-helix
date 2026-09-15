# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live end-to-end: run a real Fabric task inside a Docker sandbox via FabricContainerRuntime.

Constructs the runtime directly (the plugin `_resolve_target` wiring is not yet in place — see
AALGO-321) and drives one task through the hermes harness. The sandbox image is provisioned opaquely
(build-if-missing) on first run from published wheels — no Dockerfile to write and no source checkout.
Requires a running Docker daemon and NVIDIA_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.runtimes.fabric.container_runtime import FabricContainerRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.providers.docker import DockerSandboxProvider
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus
from nemo_evaluator_sdk.values.common import SecretRef

# nemo_fabric is a dev-installed native dep (see script/dev-install-fabric.sh); this example uses its
# typed config. A plain mapping works too — that's what the plugin passes — if you'd rather not import it.
from nemo_fabric import FabricConfig, HarnessConfig, MetadataConfig, RuntimeConfig  # ty: ignore[unresolved-import]


async def main() -> int:
    model = os.environ.get("FABRIC_MODEL", "nvidia/nemotron-3-nano-30b-a3b")

    # The default LocalSecretResolver reads the model credential from the process env. Take it from the
    # env var only (no key-file option): a key file is easy to drop into the repo and commit by accident.
    if not os.environ.get("NVIDIA_API_KEY"):
        print("Set NVIDIA_API_KEY", file=sys.stderr)
        return 2

    # The runner only *declares* the secret it needs (the env var the hermes adapter reads via its
    # requirements.env); it resolves via the default LocalSecretResolver (from the host env) and injects
    # the value into the container. No raw credential on the API surface.
    runtime = FabricContainerRuntime(
        # Typed Hermes agent config: model-only (no codex/node). The harness is chosen by
        # harness.adapter_id, never inferred from the model; Fabric owns its execution mechanism.
        FabricConfig(
            metadata=MetadataConfig(name="hermes-eval"),
            harness=HarnessConfig(adapter_id="nvidia.fabric.hermes", resolution="preinstalled"),
            models={"default": {"provider": "nvidia", "model": model}},
            runtime=RuntimeConfig(input_schema="chat", output_schema="message"),
        ),
        provider=DockerSandboxProvider(),
        secrets={"NVIDIA_API_KEY": SecretRef(root="NVIDIA_API_KEY")},
    )

    task = AgentEvalTask(
        id="hello",
        intent="Greet and add two numbers",  # eval-side metadata — never shown to the agent
        inputs={"instruction": "Reply with a one-sentence greeting and the sum of 2 and 2."},
    )

    output_dir = Path(os.environ.get("FABRIC_OUTPUT_DIR", "/tmp/fabric-container-e2e"))
    (trial,) = await runtime.run_tasks([task], AgentEvalRunConfig(work_dir=output_dir))

    print("=== TRIAL ===")
    print("status:", trial.status)
    print("output_text:", trial.output.output_text if trial.output else None)
    print("evidence keys:", trial.evidence.names() if trial.evidence else [])
    print("metadata:", json.dumps({key: str(value) for key, value in trial.metadata.items()}, indent=2))
    print("evidence under:", output_dir / "evidence" / "fabric_container")
    if trial.status != AgentEvalTrialStatus.COMPLETED:
        print("error:", trial.metadata.get("error"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
