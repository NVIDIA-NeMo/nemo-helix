<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Authoring context

Read the requirements section during source inspection, and the explanation
section before asking the user to make evaluation-design or setup decisions.
These procedures supplement the shared evidence standard and boundaries in
`SKILL.md`; they do not authorize additional actions.

## Gather requirements and setup from evidence

During discovery and source inspection, read relevant repository and supplied
documentation before asking the user to reconstruct it. Follow references to
requirements, rubrics, setup guides, dependency manifests and lockfiles, agent
configs, runner scripts, and license or access instructions. Gather what applies:

- **Purpose and success:** scenarios, expected behavior, grading rules, and relevant constraints.
- **Inputs and documentation:** source cases, fixtures, example outputs, and instructions for using them.
- **Execution:** the agent and how to invoke it, required software/services and versions, OS, hardware, and where each dependency runs.
- **Access and licenses:** documented installation and execution requirements, license provisioning, required accounts and credential variable names. Do not request secret values in chat or copy them into reports.
- **Repeatability:** starting data/state, session handling, reset procedure, and how results reach the grader.

Distinguish documented requirements, user-confirmed information, verified
availability, and unknowns. Cite the source and record what an unresolved item
blocks: task preparation, a particular grading check, or live execution. A repo
license does not establish the license or availability of its dependencies.
Ask focused questions only about missing facts that affect the next work, and
reuse earlier answers. Missing access or license provisioning can leave execution
pending while task files and independent checks proceed.

Keep the gathered requirements, progress, and next action in the selected
sub-flow's existing human-readable findings or task README under `.eval-author/`.
Do not overwrite generated evidence reports or require a new intake document
before authoring. For read-only requests, explain findings in the reply within
that sub-flow's reporting boundaries.

### Explain the eval pieces as they become relevant

Do not assume the user knows Harbor terminology or has a particular repository
layout, scorer, agent runner, or access to the system being tested. Ground the
explanation in inspected material and the user's answers. Introduce the relevant
pieces in plain language before asking the user to make decisions about them:

| Piece | What it does |
|---|---|
| Task | The test scenario: what the agent is asked to do, with any inputs and conversation steps |
| Environment | The files, data, tools, and software the task needs, including its starting state and how to reset it |
| Agent connection | How Harbor gives the task to the actual agent and collects its responses and actions |
| Grading criteria | The rules for deciding whether the agent did the task well |
| Grader, also called a verifier | The checks that apply those rules to evidence from the attempt and produce a result or score |
| Run and results | One attempt at the task, its recorded actions or outputs, and the grading results |

Use a short explanation of the pieces relevant now, not this entire table in
every reply. When useful, explain a reference solution as a known correct way to
complete the task, used to test the grader; a past response is not automatically
such a solution. Keep environment setup separate from the agent connection.

Map each discussed piece to what was found or created, what was actually tested,
and any specific gap. Existing executable evals may already supply grading;
written criteria may need implementation; recordings may lack intended outcomes.
Unknown access is not proof that access is unavailable. Missing evidence should
lead to a focused question or an explicit limitation, not an assumed setup.
Explain unfamiliar terms such as rubric, judge, weights, or partial credit before
asking about them. A checklist status or file link cannot replace that explanation.

Grading asks how an attempt performed against the task's criteria. A coverage
audit asks which intended agent behaviors the evaluation evidence covers. Explain
that distinction when offering the existing audit flow; the audit does not finish
a task's grader or supply its runtime access.
