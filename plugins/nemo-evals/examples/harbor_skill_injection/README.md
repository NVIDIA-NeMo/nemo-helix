<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor skill injection demo

## Codex with GPT-6 Luna on Kubernetes

Use a Helix deployment whose task worker images include this checkout. In the [DeepSWE notebook](../deepswe_k8s_demo/deepswe_k8s_demo.ipynb), use its connection and model setup (section 1) to get a gateway model ref for `azure/openai/gpt-6-luna`. Use an existing Luna gateway model ref if your deployment already serves it. The script uses the notebook's built-in `codex` agent, pinned at `0.162.1`, with hosted web search disabled.

Run from the repository root with `nemo` and `jq` on `PATH`.

```bash
CONTEXT=your-cli-context WORKSPACE=your-workspace MODEL_REF=your-workspace/luna-gateway \
  bash plugins/nemo-evals/examples/harbor_skill_injection/run_codex_demo.sh
```

With platform auth on, also set `INFERENCE_KEY_SECRET` to the workspace secret name from the notebook. Set `SANDBOX_PROFILE` if your deployment uses a custom profile for Codex setup.

The script builds one task image, submits a baseline and a skill-enabled run against the same pinned taskset, and prints their results. Both runs write a Python function that returns the nth positive odd integer. The verifier awards `1.0` when the function returns the correct values, uses a `supercool_` name, and takes a single positional argument named `enieme`. Harbor injects a bundled copy of the [Supercool Coding Guidelines example](../../../../packages/nhx_evals_sdk/examples/skill_eval/skills/supercool-guidelines/SKILL.md).

Download each task bundle to compare the generated `solution.py` files and verifier logs.

```bash
nemo --context "$CONTEXT" jobs results download agent-eval-results \
  --workspace "$WORKSPACE" --job "$RUN_ID-task-0" --output-file "$RUN_ID-task.tar.gz"
```

## Deterministic smoke test

Run this from the repository root with the Helix workspace installed and Docker running.

```bash
uv run python plugins/nemo-evals/examples/harbor_skill_injection/run_demo.py
```

The deterministic agent writes `nth_odd(n)` for the baseline and `supercool_impair(enieme)` when it reads the bundled `SKILL.md` from Harbor's `skills_dir`. Both functions produce the same values. The expected rewards are `0.0` and `1.0`. The demo requires no model credentials. Harbor job logs are under `temp/harbor-skill-demo/`.

On a Helix deployment built from this checkout, use the same task directory and set this `spec.target` for the skill variant in a subprocess job or a Kubernetes taskset run.

```json
{
  "kind": "harbor",
  "source": {"import_path": "nemo_evals.examples.harbor_skill_demo:SupercoolGuidelinesAgent"},
  "env_vars": {"NHX_HARBOR_MOCK_SKILL": "1"}
}
```

Omit `env_vars` for the baseline. The run option selects the bundled skill on the worker, where Harbor handles the sandbox upload.
