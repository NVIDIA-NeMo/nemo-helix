---
name: nemo-help
description: Exploratory entry point for users who don't know what NeMo Platform does yet. Catches open-ended questions like "what can you do", "where do I start", "tour", "show me the menu". Presents the menu of capabilities and hands off to nemo-skill-selection once the user picks a direction.
triggers:
  - what can you do
  - how can you help
  - what can I do here
  - where do I start
  - show me what's possible
  - what is this
  - what's in here
  - tour
  - menu
  - help me explore
  - I don't know what I want
not-for:
  - nemo-skill-selection (use once the user has picked a direction — build, optimize, evaluate, secure, status, teardown)
  - setup (use to verify install or to be told how to run the CLI install)
  - nemo-explore (use to reason about a specific agent's design)
  - superpowers:brainstorming (use for design work unrelated to NeMo Platform)
compatibility: nemo-platform >= 0.1.0; pure presentation (no commands run from this skill); safe under macOS or Linux sandbox; works without an installed CLI.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: [Read]
---

# NeMo Platform: where to start

You are helping a user who landed in a NeMo Platform repo and asked open-endedly what's possible. They have not yet picked a direction. Your job is to give them a short tour of what the platform does and route them to the right downstream skill once they commit.

This skill never runs commands. It presents options, asks a clarifying question, and hands off.

## What NeMo Platform does

NeMo Platform brings NVIDIA NeMo libraries together under one CLI, Python SDK, and web UI. The shipping value props:

- **Build agents** — scaffold a LangGraph agent wrapped in NVIDIA NeMo Agent Toolkit (NAT), deploy locally, iterate.
- **Harden agents** — add guardrails (content safety, jailbreak detection, PII redaction), red-team via the auditor, anonymize training data.
- **Evaluate agents** — LLM-as-judge, deterministic checks, agentic and RAG benchmarks via Harbor-backed eval suites.
- **Tune agents** — skill optimization, prompt and hyperparameter tuning, Switchyard model routing. Fine-tuning coming soon.

The platform optimizes LangGraph agents wrapped in NAT today. If the user has an agent in another framework (CrewAI, AutoGen, plain LangChain, Pydantic AI), the build and optimize paths need a NAT wrapper they write themselves. Be honest about this if it comes up.

## Tour script

Present the four journeys in one short message, then ask the user which fits. Do not dump every detail at once.

> NeMo Platform has four main journeys. Which one fits where you are?
>
> 1. **Build a new agent** — design it, scaffold a NAT workflow, deploy locally.
> 2. **Improve an existing agent** — harden with guardrails, evaluate accuracy, optimize cost or routing.
> 3. **Check what's running** — read-only health dashboard for the platform and any deployed agents.
> 4. **Set up or tear down** — first-time install, or shut down a local platform.
>
> If you're not sure yet, tell me what you're working on and I'll suggest a starting point.

## Routing once they pick

After the user answers, hand off to `nemo-skill-selection`. Do not try to handle the downstream flow yourself. `nemo-skill-selection` owns the full decision table and will pick the right next skill (`setup`, `nemo-explore`, `nemo-spec`, `nemo-build-agent`, `nemo-try-agent`, `nemo-status`, `nemo-teardown`, `nemo-fine-tune`, or a plugin skill like `agents-secure` or `agents-optimize`).

If the user says something off-topic (general LLM question, unrelated code task, weather, etc.), do not route to a NeMo skill. Defer to your default behavior.

## What not to do

- Do not run `nemo` CLI commands from this skill. Hand off first.
- Do not explain every capability in detail upfront. Tour, then ask, then route.
- Do not invent journeys that don't exist. If the user wants something NeMo Platform doesn't ship (e.g., training a base model from scratch), say so plainly.
- Do not skip the handoff to `nemo-skill-selection` once the user has picked a direction. The decision table lives there, not here.
