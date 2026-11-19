<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor skill injection demo

## Codex with GPT-6 Luna on Kubernetes

Use a Helix deployment whose task worker images include this checkout. In the [DeepSWE notebook](../deepswe_k8s_demo/deepswe_k8s_demo.ipynb), use its connection and model setup (section 1) to get a gateway model ref for `azure/openai/gpt-6-luna`. Use an existing Luna gateway model ref if your deployment already serves it. The script uses the notebook's built-in `codex` agent, pinned at `0.162.1`, with hosted web search disabled.

Run from the repository root with `nemo`, `jq`, and Python 3.12 or newer on `PATH`.

```bash
CONTEXT=your-cli-context WORKSPACE=your-workspace MODEL_REF=your-workspace/luna-gateway \
  bash plugins/nemo-evals/examples/harbor_skill_injection/run_codex_demo.sh
```

With platform auth on, also set `INFERENCE_KEY_SECRET` to the workspace secret name from the notebook. Set `SANDBOX_PROFILE` if your deployment uses a custom profile for Codex setup.

The script publishes the Supercool Coding Guidelines skill as `$WORKSPACE/supercool-guidelines` and uses the publication response to pin its revision. Both runs use the same pinned Taskset. The treatment gets `spec.target.skills: ["$WORKSPACE/supercool-guidelines#<digest>"]` and the baseline gets `spec.target.skills: []`. Both runs write a Python function that returns the nth positive odd integer. The verifier awards `1.0` when the function returns the correct values, uses a `supercool_` name, and takes a single positional argument named `enieme`. [Supercool Coding Guidelines](../../../../packages/nhx_evals_sdk/src/nhx_evals_sdk/agent_eval/example_skills/supercool-guidelines/SKILL.md)

Submission pins each named skill to its current revision. Each task worker downloads that revision's archive, checks its hash, and passes the extracted directory to Harbor, which uploads it into the trial sandbox.

Each arm must produce one error-free scored trial. Both `0` and `1` are valid scores. The launcher also checks that the two task Jobs used the same effective images and agent-layer setting.

The launcher prints `RUN_DIR`, which defaults to a new `temp/harbor-codex-demo.*` directory. It saves both request files and idempotency keys under `requests/`, downloads each task Job bundle, and writes `comparison.json` after validating the pair. To resume an interrupted run, set `RUN_DIR` to the same directory and repeat the original invocation. Use a new directory for a fresh comparison.

For the Evals skill example, pass both `TASKSET_REF` and `SKILL_REF` as `workspace/name#<64-hex digest>` refs. Publish and build those resources before running the launcher. This path requires a pinned separate verifier image and the plain task image with its baked Codex installation.

Download a task Job's bundle directly to recover its `solution.py`, verifier output, or startup logs.

```bash
nemo --context "$CONTEXT" jobs results download agent-eval-results \
  --workspace "$WORKSPACE" --job "$RUN_ID-task-0" --output-file "$RUN_ID-task.tar.gz"
```

## Deterministic smoke test

Run this from the repository root with the Helix workspace installed and Docker running.

```bash
uv run python plugins/nemo-evals/examples/harbor_skill_injection/run_demo.py
```

The script runs the same task twice through `HarborRuntimeConfig`: once with `skills=[]` and once with `skills=[<path to supercool-guidelines>]`. The deterministic agent writes `nth_odd(n)` for the baseline and `supercool_impair(enieme)` when it reads the injected `SKILL.md` from Harbor's `skills_dir`. Both functions produce the same values. The expected rewards are `0.0` and `1.0`. The demo requires no model credentials. Harbor job logs are under `temp/harbor-skill-demo/`.

On a Helix deployment built from this checkout, publish the skill first, then use the same task directory and set this `spec.target` in a subprocess job or a Kubernetes taskset run.

```bash
nemo agents skills publish supercool-guidelines \
  --from-dir packages/nhx_evals_sdk/src/nhx_evals_sdk/agent_eval/example_skills/supercool-guidelines \
  --workspace "$WORKSPACE"
```

```json
{
  "kind": "harbor",
  "source": {"import_path": "nemo_evals.examples.harbor_skill_demo:SupercoolGuidelinesAgent"},
  "skills": ["your-workspace/supercool-guidelines"]
}
```

Set `"skills": []` for the baseline. Omitting `skills` leaves skills to the task and agent.
