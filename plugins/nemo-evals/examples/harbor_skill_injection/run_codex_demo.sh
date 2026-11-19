#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${CONTEXT:?Set CONTEXT to your Nemo CLI context}"
: "${WORKSPACE:?Set WORKSPACE to the workspace for the evaluation}"
: "${MODEL_REF:?Set MODEL_REF to the workspace/name of your GPT-6 Luna gateway model}"

demo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
nemo_cli=(nemo --context "$CONTEXT")

taskset=$("${nemo_cli[@]}" evals tasksets create --workspace "$WORKSPACE" \
    --harbor-dir "$demo_dir/dataset" --json | jq -er '.taskset_ref')
"${nemo_cli[@]}" evals tasksets build-images "$taskset" --wait

submit() {
    local variant=$1 output run_id
    output=$(jq -n --arg variant "$variant" --arg model "$MODEL_REF" \
        --arg key "${INFERENCE_KEY_SECRET:-}" --arg profile "${SANDBOX_PROFILE:-}" '
        {
            spec: {target: {
                kind: "harbor",
                source: {name: "codex", model: {ref: $model}},
                agent_kwargs: {version: "0.162.1", disable_web_search: true},
                n_attempts: 1
            }},
            execution: {dispatcher_profile: "default"}
        }
        | if $variant == "with-skill" then .spec.target.env_vars = {NHX_HARBOR_MOCK_SKILL: "1"} else . end
        | if $key != "" then .spec.target.env_secrets = {OPENAI_API_KEY: $key} else . end
        | if $profile != "" then .execution.sandbox.profile = $profile else . end
    ' | "${nemo_cli[@]}" evals runs submit --workspace "$WORKSPACE" \
        --taskset "$taskset" --spec-file /dev/stdin) || return
    printf '%s\n' "$output" >&2
    run_id=$(awk '/^Run run-[0-9a-f]+ / {print $2; exit}' <<< "$output")
    if [[ -z "$run_id" ]]; then
        printf '%s\n' "No run ID in the $variant submission output" >&2
        return 1
    fi
    printf '%s\n' "$run_id"
}

baseline=$(submit baseline)
with_skill=$(submit with-skill)

for variant in baseline with-skill; do
    run_id=$baseline
    if [[ "$variant" == "with-skill" ]]; then
        run_id=$with_skill
    fi
    printf '\n%s %s\n' "$variant" "$run_id"
    "${nemo_cli[@]}" evals runs status "$run_id" --workspace "$WORKSPACE" --watch
    "${nemo_cli[@]}" evals runs results "$run_id" --workspace "$WORKSPACE"
done
