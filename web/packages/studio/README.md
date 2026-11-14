<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Studio

Studio's product charter. Read it before proposing or building a Studio feature. For setup, see [web/README.md](../../README.md).

## Mission

**Give every team that ships agents one place to see how their agents behave, decide what to change, and ship improvements they can trust.**

NeMo Helix runs a loop on production agents: observe → evaluate → improve → govern → ship. Automation (the CLI, SDK, skills, and coding agents) does more of that work every release. It can't decide what "better" means, whether a change should ship, or whether an agent is safe. People make those calls, and they need evidence they can read, compare, and share. Studio provides it.

Studio is valuable because it **closes the loop** in one product: what an agent did (traces), why it fell short (insights), what was tried (experiments and optimizations), and what shipped (deployments). Observability tools and model playgrounds each cover only one of those steps. Studio also lets people who don't use the CLI, such as leads, domain experts, and safety reviewers, shape agent quality. And it is the single front door to every NeMo capability.

## Responsibilities

Studio owns:

- **Legibility:** traces, insights, and eval results, from a summary down to the span that explains it.
- **Decisions:** proposed changes shown against a baseline, with approve, reject, and rollback, and with goals and rejection reasons recorded for the loop to use.
- **Orchestration:** starting, monitoring, and cancelling long-running jobs, with results shown where the user is working.
- **The shell:** workspaces, resources, auth, navigation, and the design system that core pages and plugin pages share.

Studio does not own business logic (it lives in services and plugins), anything you can't also do from the CLI or API, durable state of its own, or agent authoring (that happens in the IDE).

## Principles

When two principles conflict, the one listed first wins.

1. **Agent-first.** Models, data, evals, and jobs are shown in service of an agent.
2. **Evidence before action.** Every action sits next to the data that justifies it.
3. **People decide, automation does the work.** Ask for input only on goals, approvals, and genuinely ambiguous questions. Don't build wizards for work a job or skill can do.
4. **API parity.** Every Studio action maps to a public API. The in-Studio assistant uses the same APIs and asks for approval before it changes anything.
5. **Thin core, rich plugins.** Cross-cutting views live in core. A standalone capability ships as a [plugin UI](../../../plugins/example-plugin/web/AGENTS.md), and a preview ships behind a feature flag.

## Direction

The goal is a mostly autonomous improvement loop, with people approving the outcomes. Studio's job shifts from running workflows step by step to reviewing what the platform proposes. The next areas to build:

1. A **proposed-improvements inbox**, with before/after evidence and Approve, Reject, and Undo.
2. **Goals** per agent: what to improve and what must not regress.
3. **Change history** with one-click rollback.
4. **Notifications and questions** from the platform to a person.
5. **Collaboration:** real identities, shared preferences, and attribution.

Studio is succeeding when new users reach their first insight soon after setup, the time from insight to shipped change goes down, and people who don't use the CLI rely on Studio on their own.
