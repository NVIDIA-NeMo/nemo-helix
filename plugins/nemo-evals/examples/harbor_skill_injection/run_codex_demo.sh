#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
umask 077
: "${CONTEXT:?Set CONTEXT to your Nemo CLI context}"
: "${WORKSPACE:?Set WORKSPACE to the workspace for the evaluation}"
: "${MODEL_REF:?Set MODEL_REF to the workspace/name of your GPT-6 Luna gateway model}"

demo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$demo_dir/../../../.." && pwd)
nemo_cli=(nemo --context "$CONTEXT")
variants=(baseline with-skill)

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

pinned_ref='^[[:alnum:]_.-]+/[[:alnum:]_.-]+#[0-9a-f]{64}$'
taskset=${TASKSET_REF:-}
skill=${SKILL_REF:-}
using_overrides=false
if [[ -n "$taskset" || -n "$skill" ]]; then
    [[ "$taskset" =~ $pinned_ref && "$skill" =~ $pinned_ref ]] || \
        die 'Set TASKSET_REF and SKILL_REF together as workspace/name#<64 lowercase hex characters>.'
    using_overrides=true
fi

if [[ -z ${RUN_DIR:-} ]]; then
    mkdir -p "$repo_root/temp"
    RUN_DIR=$(mktemp -d "$repo_root/temp/harbor-codex-demo.XXXXXX")
fi
mkdir -p -- "$RUN_DIR"
run_dir=$(cd -- "$RUN_DIR" && pwd)
printf 'Comparison directory: %s\n' "$run_dir"

# Hold the directory lock from before publication through the final comparison.
exec 9>>"$run_dir/.launcher.lock"
python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)' || \
    die "Could not lock $run_dir. Check that no other launcher is using it."
scratch=$(mktemp -d "$run_dir/.launcher.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT

inputs=$(jq -n --arg context "$CONTEXT" --arg workspace "$WORKSPACE" --arg model "$MODEL_REF" \
    --arg profile "${SANDBOX_PROFILE:-}" --arg secret "${INFERENCE_KEY_SECRET:-}" \
    --arg taskset "$taskset" --arg skill "$skill" '
    {context: $context, workspace: $workspace, model: $model, profile: $profile, secret_ref: $secret,
     input_taskset_ref: $taskset, input_skill_ref: $skill}
')

write_spec() {
    local variant=$1 submission_key
    submission_key=$(python3 -c 'from uuid import uuid4; print(uuid4())')
    jq -n --arg variant "$variant" --arg model "$MODEL_REF" --arg skill "$skill" \
        --arg secret "${INFERENCE_KEY_SECRET:-}" --arg profile "${SANDBOX_PROFILE:-}" \
        --arg submission_key "$submission_key" '
        {
            idempotency_key: $submission_key,
            spec: {target: {
                kind: "harbor",
                source: {name: "codex", model: {ref: $model}},
                agent_kwargs: {version: "0.162.1", disable_web_search: true},
                n_attempts: 1
            }},
            execution: {dispatcher_profile: "default"}
        }
        | .spec.target.skills = (if $variant == "with-skill" then [$skill] else [] end)
        | if $secret != "" then .spec.target.env_secrets = {OPENAI_API_KEY: $secret} else . end
        | if $profile != "" then .execution.sandbox.profile = $profile else . end
    '
}

requests="$run_dir/requests"
if [[ ! -d "$requests" ]]; then
    if [[ "$using_overrides" == false ]]; then
        skill_dir="$repo_root/packages/nhx_evals_sdk/src/nhx_evals_sdk/agent_eval/example_skills/supercool-guidelines"
        taskset=$("${nemo_cli[@]}" evals tasksets create --workspace "$WORKSPACE" \
            --harbor-dir "$demo_dir/dataset" --json | jq -er '.taskset_ref')
        [[ "$taskset" =~ $pinned_ref ]] || die 'Taskset publication did not return a pinned ref.'
        "${nemo_cli[@]}" evals tasksets build-images "$taskset" --wait
        skill_digest=$("${nemo_cli[@]}" agents skills publish supercool-guidelines --from-dir "$skill_dir" \
            --workspace "$WORKSPACE" --output-format json | jq -er '.content_hash | select(test("^[0-9a-f]{64}$"))')
        skill="$WORKSPACE/supercool-guidelines#$skill_digest"
    fi

    # Commit both specifications together before either arm can be submitted.
    mkdir "$scratch/requests"
    jq -n --argjson inputs "$inputs" --arg taskset "$taskset" --arg skill "$skill" \
        '{inputs: $inputs, taskset_ref: $taskset, skill_ref: $skill}' >"$scratch/requests/pins.json"
    for variant in "${variants[@]}"; do
        write_spec "$variant" >"$scratch/requests/$variant.json"
    done
    mv -- "$scratch/requests" "$requests"
else
    jq -e --argjson inputs "$inputs" '.inputs == $inputs' "$requests/pins.json" >/dev/null || \
        die "The inputs differ from the saved requests in $run_dir. Use the original inputs or a fresh RUN_DIR."
    taskset=$(jq -er '.taskset_ref' "$requests/pins.json")
    skill=$(jq -er '.skill_ref' "$requests/pins.json")
fi
printf 'Taskset: %s\nSkill: %s\n' "$taskset" "$skill"

submit() {
    local variant=$1 output run_id
    local run_file="$run_dir/$variant/run-id"
    mkdir -p "$run_dir/$variant" || return
    if [[ -f "$run_file" ]]; then
        cat "$run_file"
        return
    fi

    output=$("${nemo_cli[@]}" evals runs submit --workspace "$WORKSPACE" \
        --taskset "$taskset" --spec-file "$requests/$variant.json") || {
        printf 'Resume with the same inputs and RUN_DIR=%q to reuse the saved %s request.\n' "$run_dir" "$variant" >&2
        return 1
    }
    printf '%s\n' "$output" >&2
    run_id=$(awk '/^Run run-[0-9a-f]+ / {print $2; exit}' <<< "$output")
    [[ "$run_id" =~ ^run-[0-9a-f]{16}$ ]] || die "No run ID in the $variant submission output. Resume with RUN_DIR=$run_dir."
    printf '%s\n' "$run_id" >"$scratch/$variant-run-id" || return
    mv -- "$scratch/$variant-run-id" "$run_file" || return
    cat "$run_file"
}

download_bundle() {
    local variant=$1 job=$2
    local bundle="$run_dir/$variant/agent-eval-results"
    [[ -f "$bundle/trials.jsonl" ]] && return 0
    [[ ! -e "$bundle" ]] || return 1
    "${nemo_cli[@]}" jobs results download agent-eval-results --workspace "$WORKSPACE" \
        --job "$job" --output-file "$scratch/$variant.tar.gz" || return
    python3 - "$scratch/$variant.tar.gz" "$scratch/$variant" <<'PY' || return
import sys
import tarfile

with tarfile.open(sys.argv[1]) as bundle:
    members = bundle.getmembers()
    if any(not (member.isfile() or member.isdir()) for member in members):
        raise ValueError("Job bundle contains a link or special file")
    bundle.extractall(sys.argv[2], members=members, filter="data")
PY
    [[ -f "$scratch/$variant/agent-eval-results/trials.jsonl" ]] || return 1
    mv -- "$scratch/$variant/agent-eval-results" "$bundle"
}

collect_arm() {
    local variant=$1 run_id=$2 job reward images failed=0
    printf '\n%s %s\n' "$variant" "$run_id"
    "${nemo_cli[@]}" evals runs status "$run_id" --workspace "$WORKSPACE" --watch || return
    if ! job=$("${nemo_cli[@]}" evals runs tasks "$run_id" --workspace "$WORKSPACE" --json | jq -er '
        .data | select(length == 1) | .[0].job | select(type == "string" and length > 0)
    '); then
        printf '%s (%s): expected one named task Job\n' "$variant" "$run_id" >&2
        return 1
    fi
    if ! download_bundle "$variant" "$job"; then
        printf '%s (%s): could not extract the bundle from task Job %s\n' "$variant" "$run_id" "$job" >&2
        failed=1
    fi
    if reward=$("${nemo_cli[@]}" evals runs results "$run_id" --workspace "$WORKSPACE" --json | jq -er '
        select(.run.phase == "completed" and .run.result_state == "ready")
        | .tasks
        | select(length == 1)
        | .[0].result.summary
        | select(.trial_count == 1 and .error_count == 0)
        | [.scores.scores[]? | select(.name == "harbor_reward.reward")]
        | select(length == 1)
        | .[0]
        | select(.count == 1 and .nan_count == 0 and (.mean == 0 or .mean == 1))
        | .mean
    '); then
        printf 'harbor_reward.reward: %s\n' "$reward"
    else
        printf '%s (%s): expected one completed, error-free trial with a Harbor reward of 0 or 1\n' \
            "$variant" "$run_id" >&2
        failed=1
    fi

    if ! images=$("${nemo_cli[@]}" jobs get "$job" --workspace "$WORKSPACE" -f json | jq -ce '
        def image_ref: type == "string" and length > 0;
        .spec.environment
        | select(.agent.image_ref | image_ref)
        | select((.agent_preinstalled | type) == "boolean")
        | select(.verifier == null or (.verifier.image_ref | image_ref))
        | {agent_image_ref: .agent.image_ref, verifier_image_ref: .verifier.image_ref, agent_preinstalled}
    '); then
        printf '%s (%s): unexpected effective images or agent layer in task Job %s\n' "$variant" "$run_id" "$job" >&2
        return 1
    fi
    printf '%s\n' "$images" >"$run_dir/$variant/job-environment.json" || return
    [[ "$failed" == 0 ]] || return 1

    jq -n --arg run "$run_id" --arg job "$job" --argjson reward "$reward" \
        --arg bundle "$variant/agent-eval-results" --argjson images "$images" '
        {run_id: $run, job_name: $job, reward: $reward, bundle: $bundle} + ($images | del(.agent_preinstalled))
    ' >"$scratch/$variant.json"
}

baseline=$(submit baseline)
with_skill=$(submit with-skill)

comparison_failed=0
collect_arm baseline "$baseline" || comparison_failed=1
collect_arm with-skill "$with_skill" || comparison_failed=1
[[ "$comparison_failed" == 0 ]] || exit 1

jq -ne --slurpfile baseline "$run_dir/baseline/job-environment.json" \
    --slurpfile treatment "$run_dir/with-skill/job-environment.json" '$baseline == $treatment' >/dev/null || \
    die "Task Jobs used different effective images or agent-layer settings. See $run_dir/*/job-environment.json."
if [[ "$using_overrides" == true ]]; then
    jq -e '
        def pinned: (. // "") | test("^[^@\\s]+@sha256:[0-9a-f]{64}$");
        (.agent_image_ref | pinned) and (.verifier_image_ref | pinned) and .agent_preinstalled == false
    ' "$run_dir/baseline/job-environment.json" >/dev/null || \
        die "Expected both Jobs to use a digest-pinned task image without an agent layer and a digest-pinned separate verifier."
fi

jq -n --arg taskset "$taskset" --arg skill "$skill" --arg workspace "$WORKSPACE" --arg context "$CONTEXT" \
    --slurpfile baseline "$scratch/baseline.json" --slurpfile treatment "$scratch/with-skill.json" '
    {taskset_ref: $taskset, skill_ref: $skill, workspace: $workspace, context: $context,
     arms: {baseline: $baseline[0], "with-skill": $treatment[0]}}
' >"$scratch/comparison.json"
mv -- "$scratch/comparison.json" "$run_dir/comparison.json"
printf '\nComparison: %s\n' "$run_dir/comparison.json"
