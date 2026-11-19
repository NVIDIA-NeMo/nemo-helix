<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluate skill injection with Harbor

| Harbor task | Workload | Published skill |
| --- | --- | --- |
| `nemo-evals-plugin-001` | Offline dataset evaluation | `nemo-evals-plugin` |
| `nemo-evals-plugin-002` | Live Taskset evaluation with a deterministic loopback agent | `supercool-agent-eval` |

Publish either task directory as a single-task Taskset. For the Taskset agent-evaluation
example, see [Supercool run aliases](#supercool-run-aliases).

Run one pinned Taskset twice with the same Codex model and sandbox images. The baseline has no
injected skills. The treatment receives a published NeMo Evals skill revision through
`spec.target.skills`. The task asks Codex to submit a two-row offline exact-match evaluation
to NeMo Helix and retrieve its results. A separate Harbor verifier checks the live job evidence.
The scorer replays saved trials with Harbor reward, skill activation, token efficiency,
command outcomes, and expected behavior. It scores package readiness once per skill archive.

Run the commands from the checkout root in Bash with `jq`, Python 3.12 or newer, and the
checkout's NeMo CLI and SDK available. Use a Kubernetes deployment with Builder, OpenSandbox and
Scoped Access Keys enabled. Its API and workers need the `skills-evaluation-hackathon` changes
for managed Harbor skill injection and the skill metric types. Set these values for an authenticated
workspace editor with read and inference access to both models.

```bash
export CONTEXT=dev WORKSPACE=default SANDBOX_PROFILE=harbor-standard
export MODEL_REF="$WORKSPACE/codex-gateway" JUDGE_MODEL_REF="$WORKSPACE/judge"
export INFERENCE_KEY_SECRET="$WORKSPACE/evals-inference-key"
export RUN_DIR="$PWD/temp/harbor-evals-skill-run"
example=plugins/nemo-evals/examples/harbor_evals_skill
nhx=(uv run --frozen nemo --context "$CONTEXT")
mkdir -p "$RUN_DIR"
```

`INFERENCE_KEY_SECRET` is the name of a secret containing an inference-scoped NeMo access key.
The Harbor worker issues the effective key for each trial. The standard sandbox profile permits
the platform API on its agent channel.

## Publish and check with Oracle

The task image derives from a pinned `nhx-api:0.10.0` image and uses a separate task-owned
Python verifier image. Edit [skill-removals.toml](dataset/nemo-evals-plugin-001/environment/skill-removals.toml)
before publishing to change which baked-in skill directories are removed. Entries must identify
existing skill directories beneath the image's approved skill roots. Builder needs HTTPS access
to `github.com` and `release-assets.githubusercontent.com` to fetch Codex. To use an approved
mirror, change the Dockerfile's `CODEX_RELEASE_URL` default and retain the checksums.

```bash
TASKSET_REF=$("${nhx[@]}" evals tasksets create --harbor-dir "$example/dataset/nemo-evals-plugin-001" \
    --workspace "$WORKSPACE" --json | jq -er '.taskset_ref')
export TASKSET_REF
"${nhx[@]}" evals tasksets build-images "$TASKSET_REF" --wait
"${nhx[@]}" evals runs submit --taskset "$TASKSET_REF" --workspace "$WORKSPACE" \
    --agent oracle --sandbox-profile "$SANDBOX_PROFILE"
```

Copy the printed Oracle run ID into `ORACLE_RUN_ID`.

```bash
"${nhx[@]}" evals runs status "$ORACLE_RUN_ID" --workspace "$WORKSPACE" --watch
"${nhx[@]}" evals runs results "$ORACLE_RUN_ID" --workspace "$WORKSPACE" --json
TASK_JOB=$("${nhx[@]}" evals runs tasks "$ORACLE_RUN_ID" --workspace "$WORKSPACE" --json \
    | jq -er '.data | select(length == 1) | .[0].job')
```

Require one completed trial with Harbor reward `1` and inner mean `0.5`. Download from the
task Job to inspect the verifier evidence and `startup.log`, including when startup failed and
the run has no merged bundle.

```bash
"${nhx[@]}" jobs results download agent-eval-results --job "$TASK_JOB" \
    --workspace "$WORKSPACE" --output-file "$RUN_DIR/oracle.tar.gz"
```

The [reference solution](dataset/nemo-evals-plugin-001/solution/solve.sh) runs the inner
`nemo evals evaluate --spec-file`, polls `nemo jobs get-status`, and downloads the service's
`aggregate-scores` result using `nemo jobs results download`. The inner service is loopback-only
and auth-disabled for this cooperative benchmark.

## Publish the skill and check the trial credential

```bash
skill=$("${nhx[@]}" agents skills publish nemo-evals-plugin \
    --from-dir skills/nemo-evals-plugin --taskset "$TASKSET_REF" \
    --workspace "$WORKSPACE" --output-format json)
skill_digest=$(jq -er '.content_hash' <<< "$skill")
export SKILL_REF="$WORKSPACE/nemo-evals-plugin#$skill_digest"
```

For the unscored smoke, confirm the editor can read the pinned revision and the archive it
references. Supply those two paths only to this trial. The standard profile exposes the full
platform API at the same origin as the Codex gateway.

```bash
revision=$("${nhx[@]}" agents skills get nemo-evals-plugin --revision "$skill_digest" \
    --workspace "$WORKSPACE" --output-format json)
archive_ref=$(jq -er '.archive.archive_ref' <<< "$revision")
fileset_ref=${archive_ref%%#*}
archive_path=${archive_ref#*#}
fileset_workspace=${fileset_ref%%/*}
fileset_name=${fileset_ref#*/}
"${nhx[@]}" files download "$fileset_name" --workspace "$fileset_workspace" \
    --remote-path "$archive_path" -o "$RUN_DIR/smoke-skill.tar"
jq -n --arg model "$MODEL_REF" --arg secret "$INFERENCE_KEY_SECRET" \
    --arg profile "$SANDBOX_PROFILE" --arg key "$(python3 -c 'import uuid; print(uuid.uuid4())')" \
    --arg revision "/apis/agents/v2/workspaces/$WORKSPACE/skills/nemo-evals-plugin/revisions/$skill_digest" \
    --arg archive "/apis/files/v2/workspaces/$fileset_workspace/filesets/$fileset_name/-/$archive_path" '
    {
      idempotency_key: $key,
      spec: {target: {
        kind: "harbor", source: {name: "codex", model: {ref: $model}},
        agent_kwargs: {version: "0.162.1", disable_web_search: true},
        env_secrets: {OPENAI_API_KEY: $secret}, n_attempts: 1, skills: [],
        env_vars: {NHX_SCOPE_PROBE: "1", NHX_SCOPE_PROBE_SKILL_PATH: $revision,
                   NHX_SCOPE_PROBE_ARCHIVE_PATH: $archive}
      }},
      execution: {dispatcher_profile: "default", sandbox: {profile: $profile}}
    }' > "$RUN_DIR/scope-smoke-request.json"
"${nhx[@]}" evals runs submit --taskset "$TASKSET_REF" --workspace "$WORKSPACE" \
    --spec-file "$RUN_DIR/scope-smoke-request.json"
```

Watch the smoke run and fetch its task Job bundle as for Oracle. Require a successful Codex
model response and `agent/scope-probe.json` with `enabled: true` and both `skill_status` and
`archive_status` equal to `403`. In `trial.log`, look for
`Codex is already available at the requested version` and, when the profile enables a setup
window, `Closed agent setup egress`.

## Compare and score

```bash
uv run --frozen bash plugins/nemo-evals/examples/harbor_skill_injection/run_codex_demo.sh
uv run --frozen python "$example/score.py" "$RUN_DIR"
```

Resume an interrupted pair with the same inputs and `RUN_DIR` to reuse the saved request keys.
The launcher checks equal effective agent and verifier digests and one error-free binary reward
per arm. Either reward and ties are valid. The scorer reads the raw Codex catalog and
`skill-inventory.json`, then checks that the treatment discovered the pinned skill and the
baseline catalog excluded it. A successful registered-skill read establishes activation.
Missing activation or usage evidence produces unmeasured coverage.

The output `$RUN_DIR/skill-eval-summary.json` records the live rewards, image digests, scoring
Job IDs, metrics, views, coverage and diagnostics. The scoring Jobs use a separate pinned
Taskset revision from the live Taskset. Each arm's original bundle and the scoring artifacts
are retained in the run directory.

### Inspect the comparison in Studio

Open **Evaluations → Skill Comparisons** in Studio and import
`$RUN_DIR/skill-eval-summary.json`. The view includes paired metric bars, a detailed
comparison table, token usage, and a task/trial heatmap. Select a metric or heatmap
cell to inspect the stored scorer records and diagnostics. **Package Checks** shows
archive checks separately from trial results; **Evidence** lists revision pins,
effective images, local bundle paths, and links to scoring Jobs.

Import reads JSON in the current browser tab. It does not upload results, start
runs, or fetch local bundles. Refreshing the page clears the report. Job links
open the current Helix deployment, so use the deployment matching the report's
source context and workspace. The page follows the existing Evaluations feature
flag and does not require the Tasks/Tasksets flag.

Missing measurements remain unmeasured. Differences are withheld when scoring
is incomplete, metric coverage is partial, or the arms' images/discovery do not
match the expected comparison. One trial per arm does not support significance
claims or confidence intervals. New reports retain inclusive prompt, cached
input, and completion token measurements; older reports may only retain the
uncached prompt plus completion metric.

The presentation adapts the paired-result, coverage, and token-accounting patterns
from [SkillEvaluator](https://gitlab-master.nvidia.com/ai_tools/nvevalcollection/skill-evaluator).
It uses Helix metric names and evidence contracts. It does not reinterpret those
metrics as native SkillEvaluator dimensions or publish data to Eval Studio.

The viewer also offers **Load Saved Runs** for completed Helix executions in the
current workspace. Select a baseline without skills and a treatment with the same
pinned taskset and one skill. The initial loader supports one task and one trial
per arm, searches the latest 100 executions/scoring jobs, and uses the latest
completed scoring replay with the exact trial ID. Its scoring job is identified
in Evidence. Imported reports remain available for choosing other scoring
configurations. No new evaluation is launched. Saved-run loading does not
confirm runtime skill discovery from the live rollout, so differences remain
withheld until that evidence is available.

Set `EXPECT_SETUP_EGRESS=0` when using a profile with an empty `agent_setup_egress` list.
The stored scoring Taskset follows the documented Harbor replay and metric extension APIs.
[Custom metrics and views](https://docs.nvidia.com/nemo-helix/v0.10/documentation/evaluate-models/agent-eval/harbor-runner#custom-metrics-and-views)

The command metric reports success among root Codex `exec_command` calls with a
matched exit status. Missing outcomes remain unknown. Replay sends outcome flags
and marks the trace partial.

To see sample trace, scan, and bundled-test evidence and scores, run the offline
fixture with the Evals SDK installed:

```bash
uv run --no-sync python plugins/nemo-evals/examples/harbor_evals_skill/score_evidence_example.py
```

## Supercool run aliases

The `002` task gives Codex a spec for `nemo evals agent-evaluate` against a pinned
one-task Taskset in the sandbox's Helix service. The deterministic loopback agent
returns the expected answer in both arms. The [Supercool skill](skills/supercool-agent-eval/SKILL.md)
supplies the required suffix for the job's `labels.run_alias`. The separate verifier
reads the alias and the evaluation result directly from Helix.

Use the connection and Codex model settings above, with a fresh run directory.
For auth-disabled deployments, leave `INFERENCE_KEY_SECRET` unset. The replay
uses deterministic scoring and works without `JUDGE_MODEL_REF`.

```bash
export RUN_DIR="$PWD/temp/harbor-supercool-agent-eval-run"
mkdir -p "$RUN_DIR"
TASKSET_REF=$("${nhx[@]}" evals tasksets create \
    --harbor-dir "$example/dataset/nemo-evals-plugin-002" --workspace "$WORKSPACE" --json \
    | jq -er '.taskset_ref')
export TASKSET_REF
"${nhx[@]}" evals tasksets build-images "$TASKSET_REF" --wait
"${nhx[@]}" evals runs submit --taskset "$TASKSET_REF" --workspace "$WORKSPACE" \
    --agent oracle --sandbox-profile "$SANDBOX_PROFILE"
```

Check the Oracle as above. Require one completed trial, Harbor reward `1`,
and inner exact-match mean `1.0` in the collected job snapshot. Then publish the
skill and run the pair.

```bash
skill=$("${nhx[@]}" agents skills publish supercool-agent-eval \
    --from-dir "$example/skills/supercool-agent-eval" --taskset "$TASKSET_REF" \
    --workspace "$WORKSPACE" --output-format json)
skill_digest=$(jq -er '.content_hash' <<< "$skill")
export SKILL_REF="$WORKSPACE/supercool-agent-eval#$skill_digest"
uv run --frozen bash plugins/nemo-evals/examples/harbor_skill_injection/run_codex_demo.sh
uv run --frozen python "$example/score.py" "$RUN_DIR"
```

The live and replayed Harbor rewards distinguish the successful inner evaluation
from the naming convention.

| Score | Baseline | With skill |
| --- | ---: | ---: |
| `harbor_reward.inner_eval_passed` | 1 | 1 |
| `harbor_reward.suffix_applied` | 0 | 1 |
| `harbor_reward.reward` | 0 | 1 |
| `skill_activation.activated` | Unmeasured | 1 |
| `view.verified_skill_use` | Unmeasured | 1 |

The replay adds the observed activation, token efficiency, and the
`verified_skill_use` view. The view requires both a passing verifier reward and
a confirmed read of the pinned skill. On auth-disabled dev deployments, the pair
demonstrates injection and use while archive read isolation remains unverified.
