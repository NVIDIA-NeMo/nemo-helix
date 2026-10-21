<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Legacy NAT ReAct example

`react-agent.yml` is an existing NAT workflow retained for compatibility examples.
It is **not supported by Fabric `config-tuning` optimization**. The historical
optimizer input has been moved from `react-optimize.yml` to
[`legacy-nat/react-optimize.yml.legacy`](legacy-nat/react-optimize.yml.legacy).
That file is reference material, not a runnable bundle; its paths still describe
its original location. Do not submit it to `prepare-fileset` or `run-strategy`.

Changing command names, adding `--agent react-agent`, or changing only the schema
version cannot migrate the NAT workflow. Fabric optimization requires a supported
Fabric agent, `optimizer.search_space` entries targeting its config, `eval.fabric`,
and objectives that the Evals SDK metrics actually emit. It does not translate
NAT `llms`, per-LLM `optimizable_params`, or NAT evaluator configuration.

For supported configuration tuning, follow the shipped
[Hermes optimize runbook](../../../nemo-optimization/examples/hermes-optimize/README.md).
Its chat-only inline bundle is `optimize-chatonly.yaml`; its stored-agent variant
uses `optimize-chatonly-via-agent.yaml` with `--agent hermes-optimize-chatonly`
on both prepare and submit, after registering `agents/chatonly/agent.yaml`.
These are different agents, **not** migrations of ReAct's Wikipedia/clock tools.
The runbook covers harness installation, credentials, gateway models, and job
execution requirements. Offline `prepare-fileset --dry-run --no-check-models`
validates bundle inputs only, not model access or successful optimization trials.
