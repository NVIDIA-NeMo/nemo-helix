<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Deploy and try the Email Security Triage agent

Deploy a Fabric (`nemo-agents-spec-v1`) agent end to end and watch it route an
analyst's question to the right capability. The agent is a single-turn router: it
picks one of three capabilities — general review, phishing/benign triage, and
drafting a staff warning — and answers as that capability, in one generation, with
no tools and no sub-agents.

**What you'll do:** deploy the example, ask it a question, watch the answer change
shape when the question changes, find the run in the trace, and score it against
labeled data.

**Prerequisites:**

- NeMo Helix running locally (see [SETUP.md](../../../../../SETUP.md)); `export NHX_BASE_URL=http://localhost:8080`.
- `export NVIDIA_API_KEY=<your key>`.
- Dependencies synced from the repo root: `uv sync --all-packages`.

## Step 1: Deploy the agent

```bash
nemo agents create --name email-security-triage \
  --agent-config plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/agent.yaml
nemo agents deploy --agent email-security-triage \
  --name email-security-triage-deployment --mode subprocess
```

The deploy command waits until the deployment reports `running` on a loopback port.

## Step 2: Ask whether a message is legitimate

```bash
nemo agents invoke --agent-deployment email-security-triage-deployment \
  --input '{"user_message": "is this legit?", "emails": ["Subject: Verify your account\nFrom: it-support@paypa1-secure.example\n\nYour account is locked. Confirm your password at http://paypa1-secure.example/login"]}'
```

The first line is exactly `phishing` — one lowercase word, alone. The reasoning
follows on the lines after it, naming the lookalike sender domain
(`paypa1-secure.example`) and the credential request.

That answer-first shape is the point: every capability puts a bare answer on line
one and explains after, so a metric can read the result without an LLM. It is also
what makes routing measurable — see Step 3.

Input is a JSON object with two keys. `user_message` is what the analyst typed
(`""` when they typed nothing); `emails` is what they selected. Plain email text
works too and is treated as the selected message.

## Step 3: Watch the question change the answer's shape

Same agent, same email, different question — a different capability answers, with a
different contract on line one.

```bash
# no question -> general review: line 1 is ANALYSIS, then a block per message
nemo agents invoke --agent-deployment email-security-triage-deployment \
  --input '{"user_message": "", "emails": ["Subject: Verify your account\nFrom: it-support@paypa1-secure.example\n\nConfirm your password at http://paypa1-secure.example/login"]}'

# "write a warning" -> drafting: line 1 is prose, no verdict word
nemo agents invoke --agent-deployment email-security-triage-deployment \
  --input '{"user_message": "write a warning for the team", "emails": ["Subject: DocuSign: contract awaiting signature\nFrom: no-reply@docusign-secure-files.com\n\nSign at http://docusign-secure-files.com/sign before it expires."]}'
```

Three capabilities, three distinct first lines: `phishing`/`benign`, `ANALYSIS`,
and prose. Because they cannot be confused for each other, a deterministic check on
line one proves _which capability ran_ — that is how the eval configs assert routing
without inspecting tool calls.

The general review also reports an `IOCS:` field per message, listing the URLs and
domains it found. The prompt produces those; this agent calls no tools. To wire a
deterministic extractor in instead, see [CUSTOMIZE.md](CUSTOMIZE.md).

## Step 4: Find the run in the trace

```bash
nemo agents logs --agent email-security-triage
```

The deployment's `artifacts/.../events.atof.jsonl` records the invocation. With
NeMo Studio Intake enabled (`VITE_FF_INTAKE_ENABLED=true`), the same run appears
under **Traces**.

Expect **one** LLM call per invocation. There is no per-capability span, because
there is no per-capability call — the capability is a section of one prompt, not a
separate step. That is the trade this design makes:
it gives up step-level traces and buys back cost, plus a first line that means what
it says.

## Step 5: Evaluate against labeled emails

This example ships two eval configs, both in the evaluator SDK's format:

| File                             | Shape                                                | Scores                           |
| -------------------------------- | ---------------------------------------------------- | -------------------------------- |
| `eval-config.dataset-driven.yml` | one metric set over every row of `dataset.jsonl`     | verdict accuracy + routing       |
| `eval-config.task-driven.json`   | 8 tasks in 4 families, each carrying its own metrics | all three capabilities + routing |

One is YAML and one is JSON on purpose: **both formats are accepted everywhere a
config is read.** Studio's upload and the CLI's `--spec-file` sniff the content
rather than the extension, so use whichever reads better — YAML suits configs with
multi-line prompts, JSON suits generated ones.

Both ship **without** `target`, `params`, or a resolved `dataset` — those are
per-run, and Studio's Run Evaluation flow injects them. From the CLI you supply
them yourself.

> **The task-driven config is CLI-only today.** Studio's Run Evaluation modal
> requires a dataset file, and bakes a `dataset:` reference into whatever config
> you upload — a task-driven config has neither, since its inputs live in
> `tasks[]`. The submission layer and the runner both handle the task shape
> (`nemo evaluator agent-evaluate run` scores it end to end); only the upload form
> rejects it. Run it from the CLI, as below.

Upload the dataset once, so the config can reference it:

```bash
nemo files filesets create esec-eval-data
nemo files upload plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/dataset.jsonl esec-eval-data
```

Then build a run spec from the shipped config, adding the three per-run fields:

```bash
uv run python - <<'EOF' > /tmp/esec-eval-spec.json
import json, yaml
spec = yaml.safe_load(open("plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/eval-config.dataset-driven.yml"))
agent = "email-security-triage"
spec["dataset"] = "default/esec-eval-data#dataset.jsonl"
spec["target"] = {
    "format": "generic",
    "url": f"http://localhost:8080/apis/agents/v2/workspaces/default/agents/{agent}/-/v1/chat/completions",
    "name": agent,
    "body": {"model": agent, "messages": [{"role": "user", "content": "{{ prompt }}"}], "stream": False},
    "response_path": "$.choices[0].message.content",
    "stream": False,
}
spec["params"] = {"parallelism": 4, "request_timeout": 300, "max_retries": 5, "ignore_request_failure": False}
print(json.dumps(spec, indent=2))
EOF

nemo evaluator evaluate run --spec-file /tmp/esec-eval-spec.json
```

Note this is `nemo evaluator evaluate run`, not `nemo agents evaluate` — the
latter takes a NAT-format eval YAML, and these configs are evaluator SDK specs.
The task-driven config runs the same way through
`nemo evaluator agent-evaluate run`, whose spec wraps its tasks in an `agent`
target instead.

Each row carries the agent's real input — an `emails` array plus a `user_message` —
and `prompt_template` assembles it into the text the agent reads. The `user_message`
is what routes: 20 rows ask _is this legit?_ over one message and should reach
`triage_message`; 8 rows ask nothing over two or three messages and should reach
`review_messages`. Two metrics score that:

- `string-check` — does the first line's shape match the row's `expected_tool`? A
  verdict word means triage, `ANALYSIS` means review, anything else means draft.
  This is the **routing** assertion, and it holds whether or not the verdict is right.
- `llm-judge.accuracy` — do the verdicts, read in message order, match the row's
  `expected_answer`? Scored as a **fraction**, so a 3-message row that gets two right
  scores 0.67.

Read the two together. Accuracy with headroom while routing holds means the misses
are judgement calls; both collapsing together means the agent answered as the wrong
capability. That is the distinction the routing check exists to make.

`parallelism` above matches what Studio submits, so the two paths stay comparable.
Rows are independent, so changing it moves wall-clock only, never a score. Lower it
if the agent endpoint returns 502s under concurrent load.

`ignore_request_failure: False` is deliberate. Set it to `True` and a failing agent
endpoint stops aborting the run — every failed row becomes an empty response, scores
0.0 for routing and `NaN` for the judge, and the job still reports **completed**. A
run that tested nothing then looks like a run that found a broken agent. Keep it
`False` while iterating, and read the traceback rather than the score table.

## Next Steps

- **Make it your own:** [CUSTOMIZE.md](CUSTOMIZE.md) — swap the capabilities, model, and data for your own agent.
- **Container deploys (docker/k8s):** [docs/agents/deploy-agents.mdx](../../../../../docs/agents/deploy-agents.mdx).
- **Compare with a tool-calling agent:** the sibling [calculator-agent](../calculator-agent) example.
