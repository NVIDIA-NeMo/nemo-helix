<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gym rollout rows (the dataset FileSet)

GRPO has **no labelled completions**. A row is a prompt plus enough routing information for the environment to run a rollout against it and score the result. The reward comes from the environment, not the file.

The dataset FileSet holds a training file (required) and optionally a validation file, named with `train` and `val` such as `training.jsonl` and `validation.jsonl`, referenced by the single `dataset` string. The environment is a **separate** FileSet.

## Row schema

**Where rows are actually checked.** Submit verifies only that the dataset FileSet has a training file at its root (`check_gym_dataset_layout`). A sandboxed run reads one training file and at most one validation file in place. Row *shape* is checked later by `DatasetValidator` inside the training container, against `GRPO_SCHEMA` — which is built from the **agent-agnostic** `GymDatasetRow`, so it requires exactly `responses_create_params` and `agent_ref` and passes everything else through. A wrong `vf_env_id` is therefore not a validation error; it surfaces as the environment failing to load, one job start later. Validate rows locally.

```json
{
  "task_idx": 0,
  "vf_env_id": "ascii-tree",
  "responses_create_params": {"input": [{"role": "user", "content": "Draw a binary tree of depth 3."}]},
  "agent_ref": {"type": "responses_api_agents", "name": "verifiers_agent"},
  "question": "Draw a binary tree of depth 3.",
  "answer": "",
  "task": "ascii-tree",
  "example_id": "ex-0",
  "info": {}
}
```

| Key | Required | Meaning |
|---|---|---|
| `responses_create_params` | **yes** | The prompt, in OpenAI **Responses API** shape. The messages go under `input`, not at the row's top level. NeMo-RL also reads this to apply per-row sampling settings. |
| `agent_ref` | **yes** | `{"type": "responses_api_agents", "name": "<agent>"}`. Routes the row to an agent. `type` is the only allowed value; `name` is the Gym **instance** name in the environment YAML (the converter writes `verifiers_agent`; a hand-built agent instance might be `weather_simple_agent`, not the implementation folder `simple_agent`). |
| `vf_env_id` | verifiers envs | Passed to `verifiers.load_environment()`. Must equal the environment manifest's `metadata.vf_env_id`. |
| `task_idx` | verifiers envs | Row index. Required on the verifiers row type. |
| `answer` | no | Reference answer the environment scores against. Default `""`. Whether it is used at all is the environment's business. |
| `task` | no | Task label, conventionally the env id. Default `""`. |
| `example_id` | no | Stable id for the source example. Int or string, default `0`. |
| `info` | no | Free-form dict passed through to the environment. Default `{}`. |
| `question` | no | The last user message, denormalized for readability. |

**Extra keys are allowed and passed through** (`extra="allow"`), which is how environment-specific fields ride along. Any numeric key an environment returns in its result becomes a per-environment metric on the job.

## Common mistakes

| Wrong | Right |
|---|---|
| `{"messages": [...]}` at the top level | Nest under `responses_create_params.input` |
| `{"prompt": "...", "completion": "..."}` | GRPO has no completions; the environment produces and scores them |
| `{"prompt": ..., "chosen": ..., "rejected": ...}` | That is DPO, which the **nemo-customizer** skill covers |
| `"agent_ref": "verifiers_agent"` | An **object**: `{"type": "responses_api_agents", "name": "verifiers_agent"}` |
| `agent_ref.name` set to the implementation folder (`simple_agent`) | The **instance** name from the environment YAML (`weather_simple_agent`) — see `gym-config.md` |
| `vf_env_id` differing from the manifest's `metadata.vf_env_id` | Make them match. Nothing rejects the mismatch — the environment just fails to load at spin-up |
| Prompt JSONL uploaded into the environment FileSet | A separate dataset FileSet (SKILL.md rule 5) |
| Reusing Gym's own in-tree JSONL unchanged | Gym infers the agent from the owning config; the platform path requires `agent_ref` on every row. Add it |

## Converting a dataset to Gym rows

**Converted hub environments need no work** — `pi-to-gym-conversion` writes `training.jsonl` (and `validation.jsonl` with `--validation-fraction`) alongside the package. Use those.

For your own prompts, one row per prompt, in the shape the converter writes:

```python
import json

def gym_row(idx, prompt_text, *, vf_env_id, answer="", agent="verifiers_agent"):
    return {
        "task_idx": idx,
        "vf_env_id": vf_env_id,
        "responses_create_params": {"input": [{"role": "user", "content": prompt_text}]},
        "agent_ref": {"type": "responses_api_agents", "name": agent},
        "question": prompt_text,
        "answer": answer,
        "task": vf_env_id,
        "example_id": f"ex-{idx}",
        "info": {},
    }

with open("training.jsonl", "w", encoding="utf-8") as f:
    for i, ex in enumerate(examples):                    # e.g. a HF dataset split
        row = gym_row(i, ex["question"], vf_env_id="ascii-tree", answer=ex.get("answer", ""))
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Multi-turn context goes in `input` as additional messages, in order — the same `role`/`content` dicts, ending with the turn the model must respond to.

An image prompt uses that same `content` field. Put a Responses API part list there: an `input_image` part with a string `image_url`, followed by an `input_text` part. A string `content` stays one text part.

## Find out which fields YOUR environment scores — do this first

`GymDatasetRow` requires only `responses_create_params` and `agent_ref`. **Everything the
reward actually depends on rides along under `extra="allow"` and is therefore never
validated.** A row missing the field the environment grades against is accepted at submit,
starts a job, allocates GPUs, and only then produces a meaningless reward.

So before converting a single row, read the environment's verify-request model:

```bash
# The resources server that produces the reward, in the Gym source tree
grep -n "class .*VerifyRequest" -A 15 resources_servers/<name>/app.py
```

Every annotated field on that model is a **row** field. Worked example — `math_with_judge`:

```python
class LibraryJudgeMathVerifyRequest(BaseVerifyRequest):
    question: str
    expected_answer: str
```

So its rows need `question` and **`expected_answer`** — *not* the `answer` field documented
below. `answer` is a `GymDatasetRow` default that this environment never reads; a row that
sets `answer` and omits `expected_answer` scores as if every response were wrong.

| Environment | Fields its `verify()` reads |
|---|---|
| `math_with_judge` | `question`, `expected_answer` |
| `verifiers_agent` (any hub env) | `vf_env_id`, `task_idx` |
| Anything else | Read its `*VerifyRequest` — assume nothing |

**Validate before uploading**, since nothing downstream will:

```bash
python3 -c '
import json,sys
required = {"question","expected_answer"}          # from the VerifyRequest above
for i, line in enumerate(open(sys.argv[1])):
    row = json.loads(line)
    missing = required - row.keys()
    assert not missing, f"row {i} missing {missing}"
    assert row["agent_ref"]["name"] == sys.argv[2], f"row {i} wrong agent_ref"
print("rows OK")' training.jsonl <agent-instance-name>
```

## Rows for a non-verifiers environment

`vf_env_id` and `task_idx` are `verifiers_agent`-specific. A `resources_servers` + `simple_agent` environment needs neither — but if its tools are function-calling tools, **each row must declare them**, because `simple_agent` reads the tool list from `responses_create_params`, not from the resources server:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "You are a helpful assistant."},
      {"role": "user", "content": "what's it like in sf?"}
    ],
    "tools": [{"type": "function", "name": "get_weather", "description": "",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                              "required": ["city"], "additionalProperties": false},
               "strict": true}]
  },
  "agent_ref": {"type": "responses_api_agents", "name": "weather_simple_agent"},
  "answer": "",
  "info": {}
}
```

One `tools` entry per route the server exposes, named after the route. Fields the environment's own `verify()` scores against (`answer`, or anything custom) ride along under `extra="allow"`. The full environment side of this example is in `formats.md` § **Worked example**.

## Validation sampling

`validate()` runs **exactly one rollout per row** of `validation.jsonl` — there is no validation counterpart to `num_generations_per_prompt`. To score a prompt *k* times (mean@k), repeat its **row** *k* times. Validation cost is therefore just the row count, and `val_at_start` / `val_at_end` score the same rows, which makes the before/after comparison paired.
