<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dataset-driven evaluation — Email Security Triage

`eval-config.dataset-driven.yml` and `dataset.jsonl`, beside this note, score the
[email-security-triage](README.md) agent. Studio's Run Evaluation flow seeds them into a
fileset; the CLI reads them from here (see README Step 5). Either way they are yours to
edit — nothing regenerates them.

Dataset-driven means **one metric set scores every row** of a fixed dataset. The sibling
task-driven config scores the same agent a different way: a set of distinct tasks, each
carrying its own metrics.

## What is being evaluated

Can the agent pick the right capability for what the analyst asked, and get the verdicts
right once it has? Every row carries the agent's real input — an `emails` array plus a
`user_message` — so the row itself decides which capability should answer.

This measures two capabilities over a fixed body of mail, where the task-driven suite
measures breadth across varied one-off tasks. Same agent, same subject matter, different
lens — which is the point of shipping both.

## The dataset

`dataset.jsonl` — 28 rows built from 40 labelled emails. Each row mirrors what the agent
actually receives:

```json
{
  "user_message": "is this legit?",
  "emails": [{ "from": "...", "subject": "...", "body": "..." }],
  "expected_tool": "triage_message",
  "expected_answer": "phishing"
}
```

| Rows | `user_message`   | `emails` | Routes to         |
| ---- | ---------------- | -------- | ----------------- |
| 20   | `is this legit?` | 1        | `triage_message`  |
| 8    | `""` (none)      | 2 or 3   | `review_messages` |

`expected_tool` names the capability that should answer; `expected_answer` holds the expected
verdicts in message order (`"phishing"`, or `"phishing, benign"` for a multi-message row).
Both are ground truth and neither is ever shown to the agent — `prompt_template` renders only
`user_message` and `emails`.

Emails are objects rather than pre-assembled text so the row stays readable and the template
owns the formatting. The 8 multi-message rows are each mixed-verdict on purpose: a row that
was all-benign would let a lazy reviewer score well without discriminating.

The rows are deliberately contestable, not textbook. Phishing rows are calmly written and
correctly branded, with a lookalike domain or an out-of-band request as the only tell.
Benign rows look alarming on the surface — invoices, password resets, enrollment deadlines —
and most of them _do_ ask the reader to act: review a document, log in to a portal, reset a
password. What makes them benign is that the sender and the link both belong to the
organisation they claim to be from. That is the discrimination under test: an agent keying on
"was I asked to do something urgent?" will call them phishing. Rows that leaked their own
label, ran too short, or duplicated a subject were filtered out.

Across the 28 rows the 40 emails are 22 phishing / 18 benign.

## How a row reaches the agent

`prompt_template` assembles the row into the plain text the agent reads:

```text
{{ item.user_message }}
{% for e in item.emails %}
--- Message {{ loop.index }} ---
Subject: {{ e.subject }}
From: {{ e.from }}

{{ e.body }}
{% endfor %}
```

The `user_message` is what routes: a question reaches `triage_message`, which answers with a
bare `phishing` or `benign` on line one; an empty one reaches `review_messages`, which heads
its report with `ANALYSIS` and then one block per message.

**Do not use the `tojson` filter here.** The SDK JSON-parses the rendered output of any
template containing that substring (`nhx_evals_sdk/templates.py`), so the request stops
being a string and `{{ prompt }}` in the target body goes undefined. The same trap applies to
judge prompts. Plain text avoids it and exercises the agent's open-ended input path, which is
what a pasted mail thread looks like anyway.

## The metric

| Metric         | Output         | Range | Checks                                         |
| -------------- | -------------- | ----- | ---------------------------------------------- |
| `llm-judge`    | accuracy       | 0–1   | Verdicts read in order match `expected_answer` |
| `string-check` | `string-check` | 0/1   | The first line's shape matches `expected_tool` |

The `string-check` is a **routing** assertion, not an accuracy one. It maps line one to a
capability — a verdict word means `triage_message`, `ANALYSIS` means `review_messages`, anything
else means `draft_warning` — and compares that to `expected_tool`. It scores 1 when the right
capability answered, whether or not the verdict was right.

`accuracy` is a **fraction**, not a pass/fail: a 3-message review row that gets two verdicts
right scores 0.67. That is why the metric's `scores` bounds are floats — the SDK emits an
integer-only schema when they are ints (`nhx_evals_sdk/metrics/llm_judge.py`), and a judge
forced to return an integer cannot express partial credit.

Read the two together — accuracy with headroom while routing holds is a judgement problem; both
falling together is a routing problem. Routing is asserted from the output contract rather than
from tool calls, which a Fabric agent's responses do not carry (see the task-driven README).

The judge reads the **answer**, not the explanation: line one for a triage row, and the
`VERDICT:` lines in order for a review row. The agent answers first and explains after, so
its reasoning routinely names the opposite verdict — a judge told to take the "final"
verdict would score correct answers as wrong.

A review row's report is plain text, not YAML: `IOCS: a.com, b.com` and free-text
`REASONING:` do not parse. That is why per-message verdicts are read by a judge rather than
by a deterministic template.

## Reading the results

Expect a mid-range score, not a perfect one; the dataset was built to leave headroom.

Look at the shape of the misses, not just the headline number. The agent is a security
assistant told to hunt social-engineering signals, so it tends to lean cautious — check
whether your errors skew toward benign mail called phishing, and read that tendency
against the rows the dataset deliberately made contestable.

Read the routing check alongside accuracy. While routing holds, the misses are judgement
calls; if both fall together, the agent answered as the wrong capability.

A near-zero score means something mechanical, not a collapsed agent: most likely the judge
reading the wrong line, or the agent answering as a different capability. The `string-check`
tells the two apart — it stays at 1.0 in the first case and collapses with accuracy in the
second.

## Editing this config

`dataset` carries a placeholder, `<workspace>/<fileset>#dataset.jsonl`, because no such
fileset exists in a fresh workspace. Studio's Run Evaluation flow rewrites it at seed time
so the config and its data travel together. A CLI run must set a real reference first —
upload `dataset.jsonl` and point at it (README Step 5).

If you point it elsewhere, keep the field names — `user_message` and `emails[].from`/
`.subject`/`.body` are referenced by `prompt_template`, and `expected_tool`/`expected_answer`
by the two metrics. Ground truth must stay out of `prompt_template`: rendering the whole row
would hand the agent its own answer key.

Metric templates get string operations only; there is no JSON parsing, which is why answers
are read positionally rather than structurally.

This config is authored in YAML and its task-driven sibling in JSON, to show that both
are accepted: Studio's upload and the CLI's `--spec-file` sniff the content, not the
extension. Converting between them is lossless — ship one format per file rather than
the same config twice.

## Related

- **The agent these score:** [README.md](README.md) — deploy it, try each capability, run this suite.
- **Adapting it:** [CUSTOMIZE.md](CUSTOMIZE.md) — what to change, and the couplings that break silently.
- **The sibling suite:** [eval-config.task-driven.README.md](eval-config.task-driven.README.md) — the same agent scored the other way.
