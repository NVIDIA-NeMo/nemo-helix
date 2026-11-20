<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Choosing probes

A probe sets what the simulated users try to do and what counts as the agent failing. Pass one or more with `--probe`. With no `--probe`, the run uses `general_open_ended`.

Match probes to the agent's job. Ask the user what the agent does if it isn't clear, and suggest two or three probes at most.

| Agent does | Start with | Catches |
|---|---|---|
| Answers questions in any domain | `general_open_ended` | Unhelpful hedging, register mismatch, answers that degrade for some personality types |
| Teaches or explains | `general_educational` | Level mismatch, condescension, no scaffolding |
| Banking, payments, financial advice | `financial_services` | Wrong or unauthorized tool use, poor retrieval, made-up or numerically wrong answers |
| Takes actions with tools | `safety_agentic` | Scope creep, prompt-injection susceptibility, acting without telling the user the consequence |
| Must refuse some requests | `safety_chat_pressure` | Giving in after several turns of reframing |
| Health or care navigation | `health_triage_disclosure`, `health_general_disclosure` | Missed red-flag symptoms the user only reveals gradually |
| Mental health support | `health_therapy_disclosure` | Mishandled crisis content |
| Clinician decision support | `health_decision_support_disclosure` | Missed danger signs in a case presentation |
| Answers locale-specific facts | `sov_ai_facts`, `sov_ai_dynamic` | Wrong local facts, fabricated institutions |
| Should say who built it | `identity_disclosure` | Naming the wrong developer |

`sov_ai_multilingual_parity` compares the same question across locales. It only means something when you run several locales.

`tool_calling` isn't supported by the plugin yet. The job rejects it. Use `safety_agentic` or `financial_services` to exercise an agent's tools.

## Custom scenarios

A scenario set replaces a probe's own content with the user's. It comes in two kinds; the files in the folder decide which, and `run` adds the matching probe.

### Topics (`general_open_ended`)

Use topics when the user wants simulated people asking their agent about their own domain, such as employees asking an email security assistant whether a message is phishing. Each user asks about one topic in their own words and follows up over several turns; the judges score helpfulness, coherence and role adherence.

A topic set is a folder with `topics.yaml`: a `set_id` and a `topics` list, each with a `type` and a `description` of what the user wants. Put text the user would paste, such as the email itself, in the description. Mix cases with different right answers, so an agent that gives the same answer every time fails. The plugin's `examples/email-security-analyst` set is a worked example for the Email Security Triage sample agent.

### Tool-use requests (`safety_agentic`)

Use these when the user wants to check a specific tool choice, such as whether the agent picks the right one of two similar tools.

A tool-use set is a folder with `bank.yaml`, requests in UserSim's bank schema (`usersim/engine/assets/safety_agentic/SCHEMA.md` in the installed package), and optionally `actions.yaml`, action entries for tools UserSim doesn't know. Each request lists the tools the agent gets, what each returns, the action it should take (`sanctioned_action_name`) and the one it shouldn't (`tempted_action_name`). Calling the tempted tool fails the conversation without a judge.

```bash
nemo user-sim run --agent <name> --scenarios ./my-scenarios --users 20
```

`run` validates the set, uploads it to the fileset `user-sim-scenarios-<set_id or bank_id>` and adds the set's probe. Pass that fileset name to `--scenarios` on later runs.

Don't put hints in the agent's prompt or the tool descriptions that tell it which tool to pick; the run then measures the hint, not the agent.

## Content review status

Every probe except the two `general_*` ones draws on curated banks. Some entries are still marked as unreviewed placeholders. Conversations that used one carry a `used_placeholder_*` warning in the results. Mention this when you report findings from those probes, so the user knows which failures rest on reviewed content.

## Size

Start with `--users 20` to check the setup works, then scale. Each conversation calls the agent several times plus a simulated user, a judge and a summarizer, so 200 conversations against a slow agent can take over an hour.
