# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric agent-eval runtime: config shapes for different harnesses.

The same :class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime` targets
different agent harnesses purely via the Fabric config — the harness is selected by its
``harness.adapter_id``, never inferred from a model. Across harnesses the shape differs mainly in that
``adapter_id``, optional input/output schemas, and any harness-specific ``harness.settings``:

* **Codex** (``nvidia.fabric.codex``) takes codex-specific ``harness.settings`` such as sandbox and
  approval modes.
* **Hermes** (``nvidia.fabric.hermes``) declares its ``input``/``output`` schemas.

Fabric owns each adapter's execution mechanism; callers select the adapter rather than configuring a
CLI-versus-library transport.

An optional ``model=`` slug (e.g. ``"openai/gpt-5.4"``) can be passed to ``FabricAgentRuntime`` to
apply the model to each task config, mirroring Fabric's own Harbor integration. Passing
``sandbox=<SandboxProvider>`` runs the same config inside a sandbox instead of on the host (see
``examples/fabric_container/run_e2e.py``).

The configs are built from ``nemo_fabric``'s typed config objects (``FabricConfig`` etc.), which
validate structure at construction. That makes this module — like any real Fabric use — require the
optional native stack (``nemo-fabric[codex,relay]`` + the ``nemo-relay`` gateway, plus the ``codex``
CLI for the Codex harness); see ``script/dev-install-fabric.sh``. Run it (``python -m ...`` or
directly) to print each harness config.
"""

from __future__ import annotations

import json

from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_fabric import (
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RuntimeConfig,
)

#: Codex harness with codex-specific settings.
CODEX_CLI_CONFIG = FabricConfig(
    metadata=MetadataConfig(name="codex-eval"),
    harness=HarnessConfig(
        adapter_id="nvidia.fabric.codex",
        settings={"sandbox": "read-only"},
    ),
    models={"default": ModelConfig(provider="openai", model="gpt-5.4")},
)

#: Hermes harness with explicit chat/message schemas.
HERMES_SDK_CONFIG = FabricConfig(
    metadata=MetadataConfig(name="hermes-eval"),
    harness=HarnessConfig(adapter_id="nvidia.fabric.hermes", resolution="preinstalled"),
    models={"default": ModelConfig(provider="nvidia", model="qwen2.5-coder-32b")},
    runtime=RuntimeConfig(input_schema="chat", output_schema="message"),
)

#: Named Fabric configs, one per harness, keyed by a short label.
HARNESS_CONFIGS: dict[str, FabricConfig] = {
    "codex-cli": CODEX_CLI_CONFIG,
    "hermes-sdk": HERMES_SDK_CONFIG,
}


def build_runtime(harness: str, *, model: str | None = None, work_root: str | None = None) -> FabricAgentRuntime:
    """Build a :class:`FabricAgentRuntime` for a named harness (see :data:`HARNESS_CONFIGS`)."""
    return FabricAgentRuntime(config=HARNESS_CONFIGS[harness].to_mapping(), model=model, work_root=work_root)


def main() -> None:
    """Print each harness's Fabric config to show how the structure differs."""
    for harness, config in HARNESS_CONFIGS.items():
        build_runtime(harness)  # verify the config constructs a runtime
        print(f"# {harness}  (adapter_id={config.harness.adapter_id})")
        print(json.dumps(config.to_mapping(), indent=2))
        print()


if __name__ == "__main__":
    main()
