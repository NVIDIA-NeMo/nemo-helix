#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
refresh-benchmark-cache.py

Pulls current scores from BFCL (tool-calling) and LMSYS Chatbot Arena
Elo (head-to-head human preference), enriches each model entry with
plain-English capability descriptions, and writes a JSON cache the
nemo-model-selection skill reads.

The skill uses this cache to recommend models in plain English first
and benchmark numbers second.

Usage:
    python scripts/refresh-benchmark-cache.py
    python scripts/refresh-benchmark-cache.py --output path/to/cache.json
    python scripts/refresh-benchmark-cache.py --dry-run

Stdlib only. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NotRequired, TypedDict


class BenchmarkEntry(TypedDict):
    """One row in BENCHMARK_REGISTRY: what a leaderboard measures, in plain English."""

    full_name: str
    url: str
    what_it_measures: str
    what_it_predicts: str
    does_not_predict: str
    scale: str
    primary_signal_for: list[str]


class ModelProfile(TypedDict):
    """One row in NIM_MODEL_PROFILES: a model's architectural strengths and caveats.

    `derived_from` and `intent_hints` are editorial. `derived_from` points to
    another NIM ID whose benchmark scores should be used as a rough proxy when
    the upstream leaderboard hasn't measured this exact variant. `intent_hints`
    is plain-English unpacking of what the NIM ID itself implies — the fallback
    when no benchmark data is available at all.
    """

    aliases: list[str]
    architecture_note: str
    strong_at: list[str]
    watch_out_for: list[str]
    best_deployment: str
    primary_benchmarks: list[str]
    derived_from: NotRequired[str]
    intent_hints: NotRequired[list[str]]


class BfclScore(TypedDict):
    raw: float
    percent: str  # raw formatted as e.g. "48%" for direct display
    tier: str
    plain: str
    source: Literal["direct", "inferred_from_ancestor"]
    inferred_from: NotRequired[str]


class ArenaEloScore(TypedDict):
    """One per-category Elo rating: raw points, tier band, plain-English gloss."""

    raw: int
    tier: str
    plain: str
    source: Literal["direct", "inferred_from_ancestor"]
    inferred_from: NotRequired[str]


class ModelScores(TypedDict, total=False):
    """Scores attached to a cached model — each key only present if the upstream matched.

    `arena_elo` is keyed by Arena category (e.g. "overall", "coding",
    "hard_prompts", "instruction_following"). The skill picks the category
    that matches the user's profile question 2 answer.
    """

    bfcl_v4: BfclScore
    arena_elo: dict[str, ArenaEloScore]


class CacheModelEntry(TypedDict):
    """One model entry as it appears in the written cache."""

    nim_model: str
    architecture_note: str
    strong_at: list[str]
    watch_out_for: list[str]
    best_deployment: str
    primary_benchmarks: list[str]
    intent_hints: list[str]  # plain-English fallback when no benchmark data exists
    scores: ModelScores


class CacheSources(TypedDict):
    bfcl: str
    arena_elo: str


class UpstreamIndex(TypedDict):
    """Full upstream tables, retained in the cache so the skill can token-match
    arbitrary model names from /v1/models without re-fetching at recommendation time.

    `bfcl_v4` keys are model names with variant tags already stripped; values are
    raw 0–1 scores (max across (FC)/(Prompt) variants).
    `arena_elo` keys are model names from the lmarena dataset; values are
    {category: Elo} dicts. Both tables are pre-tokenized at lookup time by the
    consumer — we keep them in their natural casing here.
    """

    bfcl_v4: dict[str, float]
    arena_elo: dict[str, dict[str, float]]


class NamespaceRule(TypedDict):
    """One row in the namespace_to_type lookup. The skill matches the user's
    model id by namespace prefix to pick the NAT `_type` value for the YAML emitter."""

    prefix: str  # e.g. "openai/", "anthropic/", "bedrock/"
    nat_type: str  # e.g. "openai", "anthropic", "aws_bedrock"
    note: str  # short justification or caveat surfaced to the user


class NameDecompositionRule(TypedDict):
    """One row in the name_decomposition_rules reference. The skill uses these
    when an unknown model name lands and needs synthesized intent_hints —
    each rule's `pattern` is a token to look for; `hint` is the plain-English
    bullet to add to the candidate's intent_hints list when the token is present.
    """

    pattern: str  # token to look for in the decomposed name
    hint: str  # plain-English bullet to emit when this token is found
    category: str  # 'family' | 'size' | 'tuning' | 'specialization' | 'version'


class Cache(TypedDict):
    """Top-level shape written to benchmark_cache.json."""

    generated_at: str
    schema_version: str
    sources: CacheSources
    benchmarks: dict[str, BenchmarkEntry]
    models: list[CacheModelEntry]
    upstream_index: UpstreamIndex
    namespace_to_type: list[NamespaceRule]
    name_decomposition_rules: list[NameDecompositionRule]


# ---------------------------------------------------------------------------
# Upstream data sources
# ---------------------------------------------------------------------------

BFCL_CSV_URL = "https://gorilla.cs.berkeley.edu/data_overall.csv"

# LMSYS Chatbot Arena Elo is published daily as a Hugging Face dataset.
# The "text" config / "latest" split holds the most recent leaderboard snapshot,
# broken down across ~22 categories (overall, coding, hard_prompts,
# instruction_following, creative_writing, industry verticals, language-specific,
# etc.). We page through the whole split so the cache can carry per-category Elo
# for each NIM model — that's what lets the skill differentiate models by the
# capability the user actually cares about, not just the headline number.
ARENA_ELO_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=lmarena-ai%2Fleaderboard-dataset&config=text&split=latest"
)
ARENA_ELO_PAGE_SIZE = 100  # HF datasets-server caps page size at 100
ARENA_ELO_MAX_PAGES = 100  # ~360 models × 22 categories ≈ 79 pages today; ceiling for safety
ARENA_ELO_SLEEP_S = 0.5  # baseline polite delay; _fetch_text retries with backoff on 429
HTTP_429_BACKOFF_S = (30.0, 60.0, 120.0)  # backoff schedule per 429 retry

DEFAULT_CACHE_PATH = Path(
    "packages/nemo_helix_ext/src/nemo_helix_ext/skills/nemo-model-selection/references/benchmark_cache.json"
)

# ---------------------------------------------------------------------------
# Benchmark registry
#
# Each benchmark entry answers three questions a non-expert can reason about:
#   what_it_measures      one sentence, no acronyms, describes the test
#   what_it_predicts      what good or bad scores mean for the user's agent
#   does_not_predict      common misreadings to guard against
#   scale                 how to interpret the raw number
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, BenchmarkEntry] = {
    "bfcl_v4": {
        "full_name": "Berkeley Function Calling Leaderboard v4",
        "url": "https://gorilla.cs.berkeley.edu/leaderboard.html",
        "what_it_measures": (
            "Whether a model can correctly decide which tool to call, with what "
            "arguments, and when not to call any tool at all — tested across "
            "single calls, parallel calls, multi-step sequences, and situations "
            "where calling a tool would be the wrong answer."
        ),
        "what_it_predicts": (
            "How reliably your agent will invoke tools without hallucinating "
            "function names, fabricating arguments, or calling the wrong tool "
            "when the user's intent is ambiguous. A high score here means fewer "
            "silent failures in production where the agent confidently calls "
            "something that doesn't exist."
        ),
        "does_not_predict": (
            "General reasoning quality or how well the model writes prose. "
            "A model can score poorly here and still be excellent at open-ended "
            "tasks that don't involve structured tool calls."
        ),
        "scale": (
            "Reported as a percentage 0–100%, higher is better. Current "
            "leaderboard top is ~77% (Claude Opus 4.5). This script buckets "
            "scores as top ≥70%, strong ≥55%, mid ≥35%, weak below that — "
            "editorial bands calibrated against the live distribution, not "
            "published by Berkeley."
        ),
        "primary_signal_for": ["tool_calling", "mcp_tools", "api_agents"],
    },
    "swe_bench_verified": {
        "full_name": "SWE-bench Verified",
        "url": "https://www.swebench.com",
        "what_it_measures": (
            "Whether a model can read a real GitHub issue, understand an "
            "existing codebase it has never seen before, write a patch that "
            "fixes the issue, and pass the repo's existing test suite — all "
            "without human guidance."
        ),
        "what_it_predicts": (
            "How well your agent will handle multi-step software tasks: reading "
            "unfamiliar code, making targeted edits, and not breaking things "
            "that were already working. Good signal for agents that interact "
            "with codebases rather than just generating isolated snippets."
        ),
        "does_not_predict": (
            "Performance on short code-generation prompts, or on tasks outside "
            "software engineering. Also does not predict tool-calling accuracy."
        ),
        "scale": "% of issues resolved, higher is better. Top OSS models reach 40–55%.",
        "primary_signal_for": ["code_agents", "software_tasks", "repo_interaction"],
    },
    "arena_elo": {
        "full_name": "LMSYS Chatbot Arena Elo (overall)",
        "url": "https://lmarena.ai",
        "what_it_measures": (
            "How often real users, shown two anonymous model responses side by "
            "side, prefer one model over the other — aggregated across millions "
            "of head-to-head votes covering every topic imaginable."
        ),
        "what_it_predicts": (
            "Whether the model will feel good to interact with in practice: "
            "clear writing, appropriate length, not over-hedging, following "
            "instructions without being obtuse. Good proxy for general-purpose "
            "agents whose output is prose the user will read directly."
        ),
        "does_not_predict": (
            "Structured output quality, tool-calling accuracy, or correctness "
            "on any specific domain. Human preference can favor confident-sounding "
            "wrong answers over correct but uncertain ones."
        ),
        "scale": (
            "Elo points, similar to chess ratings. Current leaderboard top is "
            "~1500. This script buckets ratings as top ≥1450, strong ≥1350, "
            "mid ≥1250, weak below that — editorial bands calibrated against "
            "the live distribution, not published by LMSYS."
        ),
        "primary_signal_for": ["general_purpose", "conversational", "instruction_following"],
    },
    "gpqa_diamond": {
        "full_name": "Graduate-Level Google-Proof Q&A (Diamond set)",
        "url": "https://arxiv.org/abs/2311.12022",
        "what_it_measures": (
            "Whether a model can answer questions that were written by PhD "
            "researchers to be hard enough that other experts in adjacent fields "
            "get them wrong — covering biology, chemistry, and physics. The "
            "'Diamond' subset is the hardest tier."
        ),
        "what_it_predicts": (
            "Deep multi-step reasoning and the ability to synthesize complex "
            "information without shortcuts. Relevant for agents that need to "
            "draw real conclusions from dense technical or scientific content, "
            "not just retrieve and reformat it."
        ),
        "does_not_predict": (
            "Performance on everyday tasks, instruction following, or anything "
            "involving tool use. Overkill as a signal for most business agents."
        ),
        "scale": "% correct, higher is better. Human expert baseline is ~70%; top models reach 75–80%.",
        "primary_signal_for": ["reasoning", "scientific_analysis", "technical_depth"],
    },
    "ruler": {
        "full_name": "RULER (Realistic and Unbiased Long-context Evaluation)",
        "url": "https://arxiv.org/abs/2404.06654",
        "what_it_measures": (
            "Whether a model can actually find and use information buried deep "
            "in a long document — not just claim it supports a large context "
            "window. Tests retrieval, tracking multiple entities, and "
            "aggregating information across a long input."
        ),
        "what_it_predicts": (
            "How much of the model's advertised context window you can actually "
            "rely on. A model claiming 128K context might only reliably use "
            "50–70K. Critical for agents that process long documents, codebases, "
            "or conversation histories."
        ),
        "does_not_predict": (
            "Quality on short inputs. A model can be excellent at short tasks and lose the thread badly at 60K tokens."
        ),
        "scale": "Score 0–100; also reported as effective context length vs advertised length.",
        "primary_signal_for": ["long_context", "document_analysis", "rag"],
    },
}

# ---------------------------------------------------------------------------
# Model registry
#
# Static knowledge that doesn't come from upstream leaderboards:
# what each model is architecturally good at and what to watch out for.
# Scores are fetched at runtime; these descriptions persist across refreshes.
# ---------------------------------------------------------------------------

NIM_MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen/qwen3-235b-a22b": {
        "aliases": ["Qwen3-235B-A22B", "qwen3-235b-a22b"],
        "architecture_note": "235B sparse MoE, ~22B active parameters per forward pass",
        "strong_at": [
            "Calling multiple tools in a single turn without mixing up their arguments",
            "Parallel tool invocation where the order of calls matters",
            "Recovering gracefully when a tool returns an error instead of hallucinating a result",
        ],
        "watch_out_for": [
            "Requires significant VRAM for self-hosted deployment despite low active params",
            "Overkill for agents that only call one or two simple tools",
        ],
        "best_deployment": "cloud_api",
        "primary_benchmarks": ["bfcl_v4"],
        "intent_hints": [
            "Qwen3 family — Alibaba's instruction-tuned series with a strong public tool-calling track record",
            "235B total parameters, 22B active per forward pass — sparse mixture-of-experts",
            "A22B suffix marks the 'active-22B' MoE configuration variant",
        ],
    },
    "qwen/qwen3-30b-a3b": {
        "aliases": ["Qwen3-30B-A3B", "qwen3-30b-a3b"],
        "architecture_note": "30B sparse MoE, ~3B active parameters — very fast inference",
        "strong_at": [
            "Tool-calling tasks where speed and cost matter more than perfection",
            "High-throughput agents that run many short tool-calling loops per minute",
            "Good baseline for tool-heavy agents before you know if you need the larger model",
        ],
        "watch_out_for": [
            "Lower ceiling than the 235B for complex nested or parallel tool chains",
            "May struggle with ambiguous tool selection when multiple tools fit the query",
        ],
        "best_deployment": "cloud_api_cost_sensitive",
        "primary_benchmarks": ["bfcl_v4"],
        "intent_hints": [
            "Qwen3 family",
            "30B total parameters, 3B active per forward pass — fast-inference sparse MoE",
            "Designed for high throughput at moderate quality",
        ],
    },
    "qwen/qwen3-coder-30b-a3b-instruct": {
        "aliases": ["Qwen3-Coder-30B-A3B-Instruct", "qwen3-coder-30b-a3b-instruct"],
        "architecture_note": "30B MoE fine-tuned specifically on software engineering tasks",
        "strong_at": [
            "Navigating unfamiliar codebases and making targeted edits",
            "Writing patches that don't break existing tests",
            "Agents that interact with git, CI, or code review workflows",
        ],
        "watch_out_for": [
            "Not the right choice for non-code tasks — general reasoning suffers from the specialization",
            "Tool-calling accuracy is good but not best-in-class; use qwen3-235b if tools matter more than code",
        ],
        "best_deployment": "cloud_api",
        "primary_benchmarks": ["swe_bench_verified"],
        "derived_from": "qwen/qwen3-30b-a3b",
        "intent_hints": [
            "Qwen3 family, code-specialized fine-tune",
            "Same 30B-A3B backbone as qwen3-30b-a3b (30B total, 3B active)",
            "Instruction-tuned variant, not base-model completion",
        ],
    },
    "nvidia/llama-3.1-nemotron-ultra-253b": {
        "aliases": ["Llama-3.1-Nemotron-Ultra-253B", "llama-3.1-nemotron-ultra-253b"],
        "architecture_note": "253B dense model, NVIDIA post-trained on Llama 3.1 for reasoning",
        "strong_at": [
            "Multi-step reasoning over long documents without losing the thread",
            "Technical and scientific analysis where intermediate reasoning steps matter",
            "Agents that need to synthesize information from many retrieved chunks",
        ],
        "watch_out_for": [
            "Very large model — cloud API is the practical deployment path for most teams",
            "Slower inference than MoE alternatives; not ideal for latency-sensitive loops",
        ],
        "best_deployment": "cloud_api",
        "primary_benchmarks": ["gpqa_diamond", "ruler"],
        "intent_hints": [
            "Llama 3.1 base, post-trained by NVIDIA",
            "253B dense parameters — large, slower inference",
            "'Ultra' tier in Nemotron lineup, optimized for multi-step reasoning over function calling",
        ],
    },
    "nvidia/llama-3.3-nemotron-super-49b-v1": {
        "aliases": ["Llama-3.3-Nemotron-Super-49B-v1", "llama-3.3-nemotron-super-49b-v1"],
        "architecture_note": "49B dense model, NVIDIA post-trained on Llama 3.3; platform default for cloud agents",
        "strong_at": [
            "General-purpose agent work where you don't yet know the bottleneck",
            "Reasonable tool-calling accuracy plus solid prose — a balanced starting point",
            "The platform's curated default; well-tested across the agent build path",
        ],
        "watch_out_for": [
            "Not specialized — a tool-calling specialist will beat it on heavy tool chains",
            "Not the fastest — a smaller MoE wins on cost and latency for simple loops",
        ],
        "best_deployment": "cloud_api",
        "primary_benchmarks": ["arena_elo"],
        "derived_from": "meta/llama-3.3-70b-instruct",
        "intent_hints": [
            "Llama 3.3 70B base, distilled and post-trained by NVIDIA",
            "49B dense — distilled from a larger model",
            "'Super' tier, v1 — NVIDIA's current cloud agent default",
        ],
    },
    "meta/llama-3.3-70b-instruct": {
        "aliases": ["Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct"],
        "architecture_note": "70B dense model, widely deployed and well-understood",
        "strong_at": [
            "General-purpose instruction following across a wide range of tasks",
            "Stable, predictable outputs — well-characterized by the community",
            "Good starting point for any agent before you know its bottlenecks",
        ],
        "watch_out_for": [
            "Not a specialist in any one area — if tool-calling or code quality is critical, use a specialist",
            "Human preference scores well but that doesn't translate directly to agentic reliability",
        ],
        "best_deployment": "cloud_api_or_self_hosted",
        "primary_benchmarks": ["arena_elo"],
        "intent_hints": [
            "Meta's Llama 3.3 70B, instruction-tuned",
            "Dense 70B — well-characterized, widely deployed",
            "Most reference implementations target this size class",
        ],
    },
    "microsoft/phi-4-mini-instruct": {
        "aliases": ["Phi-4-Mini-Instruct", "phi-4-mini-instruct"],
        "architecture_note": "Small dense model (~4B), optimized for quality-per-parameter",
        "strong_at": [
            "Fast, cheap inference for agents where latency is the bottleneck",
            "Single-tool or low-complexity tool-calling loops",
            "Edge or resource-constrained deployments",
        ],
        "watch_out_for": [
            "Ceiling is lower than larger models for complex multi-tool orchestration",
            "May miss subtle nuance in tool argument generation under ambiguous prompts",
        ],
        "best_deployment": "self_hosted_low_vram",
        "primary_benchmarks": ["arena_elo"],
        "intent_hints": [
            "Microsoft's Phi-4 small variant, instruction-tuned",
            "'Mini' — roughly 3.8B dense parameters, distinct from the 14B Phi-4",
            "Designed for high quality-per-parameter at edge / low-VRAM deployments",
        ],
    },
    "qwen/qwen3-8b": {
        "aliases": ["Qwen3-8B", "qwen3-8b"],
        "architecture_note": "8B dense model with strong BFCL performance for its size class",
        "strong_at": [
            "Tool-calling on a single consumer GPU (fits in 12–16 GB VRAM)",
            "Local development and prototyping before scaling to a larger model",
            "Agents where self-hosting is non-negotiable and tools are the primary task",
        ],
        "watch_out_for": [
            "Not competitive with larger models on complex reasoning chains",
            "Context window reliability drops faster than in larger models",
        ],
        "best_deployment": "self_hosted_local_gpu",
        "primary_benchmarks": ["bfcl_v4"],
        "intent_hints": [
            "Qwen3 family, dense",
            "8B parameters — fits on a single consumer GPU (12–16 GB VRAM)",
            "Strong BFCL performance for its size class",
        ],
    },
}

# ---------------------------------------------------------------------------
# Provider-type inference table
#
# Maps the namespace prefix of a model id (the part before '/') to the NAT
# workflow YAML `_type` value the LLM provider block should use. Editorial.
# When a prefix isn't listed, the skill instructs the user to run
# `nat info components -t llm_provider` to confirm the right type.
# ---------------------------------------------------------------------------

NAMESPACE_TO_TYPE: list[NamespaceRule] = [
    {"prefix": "nim/", "nat_type": "nim", "note": "NVIDIA NIM — the platform's native provider type."},
    {"prefix": "openai/", "nat_type": "openai", "note": "OpenAI direct API."},
    {
        "prefix": "anthropic/",
        "nat_type": "anthropic",
        "note": "Anthropic direct API. Verify with `nat info components -t llm_provider`.",
    },
    {
        "prefix": "bedrock/",
        "nat_type": "aws_bedrock",
        "note": "AWS Bedrock. Verify with `nat info components -t llm_provider`; exact type may be `aws_bedrock` or `bedrock`.",
    },
    {
        "prefix": "ollama/",
        "nat_type": "openai",
        "note": "Local Ollama exposes an OpenAI-compatible endpoint; use `_type: openai` with the Ollama base_url.",
    },
    # Vendor-published NIM names without explicit nim/ prefix — these are still served
    # by the platform's NIM provider when listed in /v1/models, so they map to `nim`.
    {"prefix": "qwen/", "nat_type": "nim", "note": "Qwen models served through NIM."},
    {"prefix": "meta/", "nat_type": "nim", "note": "Meta Llama models served through NIM."},
    {"prefix": "nvidia/", "nat_type": "nim", "note": "NVIDIA-published models (Nemotron family) served through NIM."},
    {"prefix": "microsoft/", "nat_type": "nim", "note": "Microsoft Phi models served through NIM."},
    {"prefix": "mistralai/", "nat_type": "nim", "note": "Mistral models served through NIM when published there."},
]

# ---------------------------------------------------------------------------
# Name decomposition rules
#
# When an unknown model name lands (custom fine-tune, vendor model we haven't
# registered, etc.) the skill synthesizes intent_hints by checking which of
# these tokens appear in the decomposed name. Editorial — patterns can be
# extended without code changes, the skill reads from the cache.
# ---------------------------------------------------------------------------

NAME_DECOMPOSITION_RULES: list[NameDecompositionRule] = [
    # Family
    {
        "pattern": "qwen3",
        "category": "family",
        "hint": "Qwen3 family — Alibaba's instruction-tuned series with a public tool-calling track record",
    },
    {
        "pattern": "qwen2",
        "category": "family",
        "hint": "Qwen2 family — older Alibaba generation, generally superseded by Qwen3 where available",
    },
    {
        "pattern": "llama",
        "category": "family",
        "hint": "Meta Llama family — widely deployed, well-characterized baseline",
    },
    {
        "pattern": "nemotron",
        "category": "family",
        "hint": "NVIDIA Nemotron — Llama-derived post-trained variant emphasizing reasoning over function calling",
    },
    {
        "pattern": "phi",
        "category": "family",
        "hint": "Microsoft Phi family — small-but-capable models tuned for quality-per-parameter",
    },
    {
        "pattern": "mistral",
        "category": "family",
        "hint": "Mistral family — French open-weight models, generally strong on European languages",
    },
    {"pattern": "mixtral", "category": "family", "hint": "Mistral Mixtral — sparse MoE variant of the Mistral line"},
    {
        "pattern": "claude",
        "category": "family",
        "hint": "Anthropic Claude family — frontier closed-weight model, strong on instruction-following and code",
    },
    {"pattern": "gpt", "category": "family", "hint": "OpenAI GPT family — frontier closed-weight model"},
    {
        "pattern": "gemini",
        "category": "family",
        "hint": "Google Gemini family — frontier closed-weight model with strong multimodal support",
    },
    {
        "pattern": "deepseek",
        "category": "family",
        "hint": "DeepSeek family — Chinese open-weight models with strong code and reasoning track record",
    },
    # Size markers (parameter counts)
    {"pattern": "0.6b", "category": "size", "hint": "0.6B parameters — very small, edge/embedded class"},
    {"pattern": "1.7b", "category": "size", "hint": "1.7B parameters — small, low-VRAM"},
    {"pattern": "3b", "category": "size", "hint": "~3B parameters — small dense, fits single consumer GPU"},
    {"pattern": "4b", "category": "size", "hint": "~4B parameters — small dense, fits single consumer GPU"},
    {"pattern": "7b", "category": "size", "hint": "~7B parameters — small-mid dense, fits 12–16 GB VRAM"},
    {"pattern": "8b", "category": "size", "hint": "~8B parameters — small-mid dense, fits single consumer GPU"},
    {"pattern": "14b", "category": "size", "hint": "~14B parameters — mid dense, needs 24+ GB for self-hosting"},
    {
        "pattern": "30b",
        "category": "size",
        "hint": "~30B parameters — mid-large; pair with active-param suffix for MoE variants",
    },
    {"pattern": "32b", "category": "size", "hint": "~32B parameters — mid-large dense"},
    {
        "pattern": "49b",
        "category": "size",
        "hint": "~49B parameters — large dense, distilled-from-larger variants are common",
    },
    {"pattern": "70b", "category": "size", "hint": "~70B parameters — large dense, cloud or multi-GPU self-host"},
    {
        "pattern": "235b",
        "category": "size",
        "hint": "~235B parameters total — very large; check for MoE active-param suffix",
    },
    {"pattern": "253b", "category": "size", "hint": "~253B parameters — very large dense"},
    {"pattern": "405b", "category": "size", "hint": "~405B parameters — frontier dense scale"},
    # MoE active-params marker (suffix like a3b, a22b)
    {
        "pattern": "a3b",
        "category": "size",
        "hint": "Sparse MoE with ~3B active parameters per forward pass — fast inference at moderate quality",
    },
    {"pattern": "a22b", "category": "size", "hint": "Sparse MoE with ~22B active parameters per forward pass"},
    # Tuning
    {
        "pattern": "instruct",
        "category": "tuning",
        "hint": "Instruction-tuned — designed to follow user directions, not base-model completion",
    },
    {"pattern": "chat", "category": "tuning", "hint": "Chat-tuned — optimized for conversational back-and-forth"},
    {
        "pattern": "base",
        "category": "tuning",
        "hint": "Base model — pretraining only, no instruction tuning. Usually wrong for agents.",
    },
    {
        "pattern": "rlhf",
        "category": "tuning",
        "hint": "RLHF-tuned — aligned via reinforcement learning from human feedback",
    },
    {"pattern": "dpo", "category": "tuning", "hint": "DPO-tuned — aligned via direct preference optimization"},
    {
        "pattern": "thinking",
        "category": "tuning",
        "hint": "Reasoning/thinking variant — exposes intermediate chain-of-thought; usually slower but better on hard problems",
    },
    # Specialization
    {
        "pattern": "coder",
        "category": "specialization",
        "hint": "Code-specialized fine-tune — strong on software engineering tasks, weaker on general reasoning",
    },
    {
        "pattern": "code",
        "category": "specialization",
        "hint": "Code-specialized variant — strong on software engineering tasks",
    },
    {"pattern": "vl", "category": "specialization", "hint": "Vision-language variant — accepts image input"},
    {"pattern": "vision", "category": "specialization", "hint": "Vision-language variant — accepts image input"},
    {
        "pattern": "math",
        "category": "specialization",
        "hint": "Math-specialized variant — strong on numerical/symbolic reasoning",
    },
    # Versions / variants — kept loose since vendors aren't consistent
    {
        "pattern": "mini",
        "category": "size",
        "hint": "'Mini' variant — small parameter count within the family; distinct from the same family's full-size model",
    },
    {"pattern": "super", "category": "size", "hint": "'Super' tier — middle ground within NVIDIA's Nemotron lineup"},
    {"pattern": "ultra", "category": "size", "hint": "'Ultra' tier — largest size within NVIDIA's Nemotron lineup"},
]


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def _fetch_text(url: str, timeout: int = 15) -> str:
    """Fetch a URL with retry-on-429 backoff.

    HF's free-tier datasets-server enforces a per-IP rate limit that's stricter
    than its docs claim. When it returns 429, sleep for the next backoff
    interval and retry — up to len(HTTP_429_BACKOFF_S) attempts. Other HTTP
    errors propagate immediately.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "nemo-helix-benchmark-refresh/1.0"})
    last_exc: Exception | None = None
    for attempt in range(len(HTTP_429_BACKOFF_S) + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt >= len(HTTP_429_BACKOFF_S):
                raise
            backoff = HTTP_429_BACKOFF_S[attempt]
            print(f"\n    429 from upstream — backing off {backoff:.0f}s...", flush=True)
            time.sleep(backoff)
            last_exc = exc
    # Unreachable in practice (the loop either returns or raises) but satisfies type checkers
    raise last_exc if last_exc else RuntimeError("fetch retry loop exited unexpectedly")


def fetch_bfcl_scores() -> dict[str, float]:
    """Pull the live BFCL `data_overall.csv` from gorilla.cs.berkeley.edu.

    Schema (as of 2026): columns include `Model` and `Overall Acc`; values are
    percent-formatted strings like `77.47%`. Normalize to 0–1 floats. Model
    names carry suffixes like ` (FC)` for the function-calling variant; strip
    those so substring matching against NIM aliases works.
    """
    print("  Fetching BFCL overall scores...", end=" ", flush=True)
    try:
        raw = _fetch_text(BFCL_CSV_URL)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return {}

    if not raw.strip():
        print("FAILED (empty response)")
        return {}

    scores: dict[str, float] = {}
    # csv.reader handles quoted/escaped fields correctly if Berkeley ever starts
    # quoting model names (today they don't, but the parser shouldn't depend on it).
    reader = csv.reader(raw.splitlines())
    try:
        header = [h.strip().lower() for h in next(reader)]
    except StopIteration:
        print("FAILED (empty CSV)")
        return {}
    try:
        model_idx = header.index("model")
        score_idx = header.index("overall acc")
    except ValueError:
        print("FAILED (unexpected CSV header)")
        return {}

    for parts in reader:
        if len(parts) <= max(model_idx, score_idx):
            continue
        model = parts[model_idx].strip()
        # Strip " (FC)", " (Prompt)" and similar variant tags
        for tag in (" (FC)", " (Prompt)", " (Function Calling)"):
            if model.endswith(tag):
                model = model[: -len(tag)].rstrip()
        raw_score = parts[score_idx].strip().rstrip("%")
        try:
            score = float(raw_score) / 100.0
        except ValueError:
            continue
        # When both (FC) and (Prompt) variants of the same base model exist, both
        # collapse to the same key after tag-stripping. Keep the higher score —
        # last-write-wins would arbitrarily depend on CSV row order.
        scores[model] = max(scores.get(model, 0.0), score)

    print(f"OK ({len(scores)} entries)")
    return scores


def fetch_arena_elo_scores() -> dict[str, dict[str, float]]:
    """Pull current Arena Elo ratings, broken down by category, from the LMSYS dataset.

    The dataset is updated daily and served as JSON via the HF datasets-server
    `/rows` endpoint — stdlib-friendly, no parquet decoding. We page through
    the full `latest` split (~360 models × ~22 categories), with a short sleep
    between requests to stay under HF's rate limit. Returns a nested dict:
    `{model_name: {category: rating}}`.
    """
    print("  Fetching Arena Elo scores (per-category)...", end=" ", flush=True)

    by_model: dict[str, dict[str, float]] = {}
    publish_date: str = ""
    categories_seen: set[str] = set()

    for page in range(ARENA_ELO_MAX_PAGES):
        if page > 0:
            time.sleep(ARENA_ELO_SLEEP_S)

        offset = page * ARENA_ELO_PAGE_SIZE
        url = f"{ARENA_ELO_ROWS_URL}&offset={offset}&length={ARENA_ELO_PAGE_SIZE}"
        try:
            raw = _fetch_text(url, timeout=20)
            payload = json.loads(raw)
        except Exception as exc:
            print(f"PARTIAL — failed page {page} ({exc})")
            break  # keep what we already collected

        rows = payload.get("rows", [])
        if not rows:
            break

        for entry in rows:
            row = entry.get("row", {})
            model = row.get("model_name")
            category = row.get("category")
            rating = row.get("rating")
            if not (isinstance(model, str) and isinstance(category, str) and isinstance(rating, (int, float))):
                continue
            by_model.setdefault(model, {})[category] = float(rating)
            categories_seen.add(category)
            if not publish_date and isinstance(row.get("leaderboard_publish_date"), str):
                publish_date = row["leaderboard_publish_date"]

        if len(rows) < ARENA_ELO_PAGE_SIZE:
            break
    else:
        # Loop completed without break — we may have hit the page ceiling.
        print(f"NOTE — hit ARENA_ELO_MAX_PAGES={ARENA_ELO_MAX_PAGES} ceiling, dataset may be larger.")

    suffix = f" from {publish_date}" if publish_date else ""
    print(f"OK ({len(by_model)} models × {len(categories_seen)} categories{suffix})")
    return by_model


# ---------------------------------------------------------------------------
# Matching + cache assembly
# ---------------------------------------------------------------------------

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokenize(name: str) -> list[str]:
    """Lowercase and split on any non-alphanumeric run.

    `Qwen3-235B-A22B-Instruct-2507 (FC)` → ['qwen3', '235b', 'a22b', 'instruct', '2507', 'fc']
    `qwen/qwen3-coder-30b-a3b-instruct`  → ['qwen', 'qwen3', 'coder', '30b', 'a3b', 'instruct']
    """
    return [tok for tok in _TOKEN_SPLIT.split(name.lower()) if tok]


def _tokens_match(alias_tokens: list[str], upstream_tokens: list[str]) -> bool:
    """True when alias tokens appear as a contiguous subsequence of upstream tokens.

    `phi-4-mini` → ['phi', '4', 'mini'] is NOT a subsequence of ['phi', '4', 'prompt'],
    so the prior false-positive against the 14B `Phi-4` row is correctly rejected.
    `qwen3-235b` → ['qwen3', '235b'] IS a subsequence of
    ['qwen3', '235b', 'a22b', 'instruct', '2507', 'fc'], so the legitimate match holds.
    """
    if not alias_tokens or len(alias_tokens) > len(upstream_tokens):
        return False
    n = len(alias_tokens)
    for i in range(len(upstream_tokens) - n + 1):
        if upstream_tokens[i : i + n] == alias_tokens:
            return True
    return False


def _match_score(nim_model: str, aliases: list[str], upstream: dict[str, float]) -> float | None:
    """Token-aware contiguous-subsequence match. Returns the highest score across all matches.

    Picking the max is intentional: when both `(FC)` and `(Prompt)` BFCL variants of the
    same model exist, we want the user-friendlier number, not whichever row came second
    in the CSV.
    """
    candidates_tokenized = [_tokenize(c) for c in [nim_model] + aliases]
    best: float | None = None
    for key, val in upstream.items():
        key_tokens = _tokenize(key)
        for cand_tokens in candidates_tokenized:
            if _tokens_match(cand_tokens, key_tokens):
                if best is None or val > best:
                    best = val
                break  # one match per upstream key is enough; move on
    return best


def _match_arena_categories(
    nim_model: str,
    aliases: list[str],
    upstream: dict[str, dict[str, float]],
) -> dict[str, float] | None:
    """Token-aware match for the per-category Arena dict. Returns the matched model's full
    category map, or None. When multiple upstream entries match (rare), picks the one with
    the highest 'overall' rating; falls back to the first match if no 'overall' exists.
    """
    candidates_tokenized = [_tokenize(c) for c in [nim_model] + aliases]
    best_key: str | None = None
    best_overall: float = float("-inf")
    for key, cat_map in upstream.items():
        key_tokens = _tokenize(key)
        for cand_tokens in candidates_tokenized:
            if _tokens_match(cand_tokens, key_tokens):
                overall = cat_map.get("overall", float("-inf"))
                if best_key is None or overall > best_overall:
                    best_key = key
                    best_overall = overall
                break
    return upstream[best_key] if best_key else None


# Tier thresholds are editorial buckets calibrated against the current BFCL
# distribution (top score ~77%). They are NOT published by Berkeley — they're
# this script's bucketing on top of the raw percentage. When the distribution
# shifts (a new frontier model raises the ceiling, or the benchmark is revised),
# revisit these.
def _bfcl_tier(score: float) -> str:
    if score >= 0.70:
        return "top"
    if score >= 0.55:
        return "strong"
    if score >= 0.35:
        return "mid"
    return "weak"


def _bfcl_plain(score: float) -> str:
    """Translate a BFCL score into a sentence a layperson can act on."""
    tier = _bfcl_tier(score)
    return {
        "top": (
            f"Correctly selects and calls tools in {score:.0%} of test cases, "
            "including tricky parallel and multi-step scenarios. At the frontier "
            "of what currently-public models can do on this benchmark."
        ),
        "strong": (
            f"Correctly handles tools in roughly {score:.0%} of cases. Competitive "
            "for production tool-calling agents; may occasionally mis-select a "
            "tool under ambiguous prompts."
        ),
        "mid": (
            f"Gets tool calls right about {score:.0%} of the time. Workable for "
            "single-tool or low-complexity agents; expect reliability issues "
            "with parallel calls or chained sequences."
        ),
        "weak": (
            f"Tool-calling accuracy around {score:.0%} — well behind current "
            "competitive models. Consider a stronger model if precise tool "
            "invocation is the main job."
        ),
    }[tier]


def _bfcl_plain_inferred(score: float, ancestor_id: str) -> str:
    """Editorial blurb for a BFCL score inherited from an ancestor model.

    Surfaces both the inference and the reasoning behind it. The skill is
    expected to read this verbatim — do not hide that the number is inferred.
    """
    ancestor_short = ancestor_id.split("/")[-1]
    reasoning = {
        "qwen/qwen3-30b-a3b": (
            "The coder variant is fine-tuned on software engineering data, "
            "which usually tightens structured-output discipline (helpful for "
            "tool calls) but narrows the variety of tool patterns the model has "
            "seen during training (potentially hurts on unfamiliar APIs)."
        ),
        "meta/llama-3.3-70b-instruct": (
            "NVIDIA's Nemotron post-training emphasizes reasoning and "
            "instruction following, not function-call discipline, so a "
            "meaningful uplift over the base model isn't likely."
        ),
    }.get(
        ancestor_id,
        (
            "No public information indicates the post-training changes tool-calling "
            "discipline materially, so the base score is the best available anchor."
        ),
    )
    return (
        f"Berkeley hasn't published a BFCL score for this exact variant. "
        f"Best available signal: its ancestor {ancestor_short} scored {score:.0%}. "
        f"{reasoning} Treat as rough signal, not measurement."
    )


def _arena_elo_plain_inferred(score: float, category: str, ancestor_id: str) -> str:
    """Editorial blurb for an Arena Elo entry inherited from an ancestor model."""
    ancestor_short = ancestor_id.split("/")[-1]
    label = _category_label(category)
    rounded = int(round(score))
    return (
        f"LMSYS hasn't published an Arena rating for this exact variant on {label}. "
        f"Best available signal: its ancestor {ancestor_short} sits at {rounded} Elo. "
        f"Post-training can shift human-preference scores in either direction — "
        f"treat as rough signal, not measurement."
    )


# Tier thresholds are editorial buckets calibrated against the current Arena
# distribution (top ~1500). They are NOT published by LMSYS — they're this
# script's bucketing on top of the raw Elo. Revisit when the ceiling shifts.
def _arena_elo_tier(score: float) -> str:
    if score >= 1450:
        return "top"
    if score >= 1350:
        return "strong"
    if score >= 1250:
        return "mid"
    return "weak"


def _category_label(category: str) -> str:
    """Human-readable name for an Arena category in plain-English sentences."""
    return {
        "overall": "overall head-to-head preference",
        "coding": "coding tasks",
        "hard_prompts": "hard prompts",
        "hard_prompts_english": "hard prompts (English)",
        "instruction_following": "instruction following",
        "creative_writing": "creative writing",
        "expert": "expert-level prompts",
        "english": "English prompts",
        "chinese": "Chinese prompts",
        "german": "German prompts",
        "french": "French prompts",
        "japanese": "Japanese prompts",
        "korean": "Korean prompts",
        "exclude_ties": "head-to-head preference (excluding ties)",
    }.get(category, category.replace("_", " "))


def _arena_elo_plain(score: float, category: str = "overall") -> str:
    """Translate an Arena Elo rating into a sentence a layperson can act on."""
    tier = _arena_elo_tier(score)
    rounded = int(round(score))
    label = _category_label(category)
    return {
        "top": (
            f"Rated {rounded} Elo on {label} in head-to-head matches with real "
            "users — at the frontier for this capability."
        ),
        "strong": (
            f"Rated {rounded} Elo on {label} — competitive, reliable across the "
            "kinds of prompts real users send in this category."
        ),
        "mid": (
            f"Rated {rounded} Elo on {label} — workable, but expect to lose "
            "preference comparisons against top models in this category."
        ),
        "weak": (
            f"Rated {rounded} Elo on {label} — well below current competitive "
            "models. Consider a stronger model if this capability is the main job."
        ),
    }[tier]


def _resolve_bfcl(
    nim_id: str,
    profile: ModelProfile,
    bfcl: dict[str, float],
) -> tuple[float, Literal["direct", "inferred_from_ancestor"], str | None] | None:
    """Try direct match first; if that fails and the profile declares a `derived_from`,
    try the ancestor's aliases instead. Returns (score, source, inferred_from)."""
    direct = _match_score(nim_id, profile["aliases"], bfcl)
    if direct is not None:
        return (direct, "direct", None)
    ancestor_id = profile.get("derived_from")
    if ancestor_id and ancestor_id in NIM_MODEL_PROFILES:
        ancestor = NIM_MODEL_PROFILES[ancestor_id]
        inherited = _match_score(ancestor_id, ancestor["aliases"], bfcl)
        if inherited is not None:
            return (inherited, "inferred_from_ancestor", ancestor_id)
    return None


def _resolve_arena(
    nim_id: str,
    profile: ModelProfile,
    arena_elo: dict[str, dict[str, float]],
) -> tuple[dict[str, float], Literal["direct", "inferred_from_ancestor"], str | None] | None:
    """Same lineage fallback for Arena's per-category dict."""
    direct = _match_arena_categories(nim_id, profile["aliases"], arena_elo)
    if direct is not None:
        return (direct, "direct", None)
    ancestor_id = profile.get("derived_from")
    if ancestor_id and ancestor_id in NIM_MODEL_PROFILES:
        ancestor = NIM_MODEL_PROFILES[ancestor_id]
        inherited = _match_arena_categories(ancestor_id, ancestor["aliases"], arena_elo)
        if inherited is not None:
            return (inherited, "inferred_from_ancestor", ancestor_id)
    return None


def build_cache(bfcl: dict[str, float], arena_elo: dict[str, dict[str, float]]) -> Cache:
    models: list[CacheModelEntry] = []

    for nim_id, profile in NIM_MODEL_PROFILES.items():
        scores: ModelScores = {}

        bfcl_resolved = _resolve_bfcl(nim_id, profile, bfcl)
        if bfcl_resolved is not None:
            score, source, inferred_from = bfcl_resolved
            entry: BfclScore = {
                "raw": round(score, 4),
                "percent": f"{score:.0%}",
                "tier": _bfcl_tier(score),
                "plain": _bfcl_plain(score) if source == "direct" else _bfcl_plain_inferred(score, inferred_from or ""),
                "source": source,
            }
            if inferred_from:
                entry["inferred_from"] = inferred_from
            scores["bfcl_v4"] = entry

        arena_resolved = _resolve_arena(nim_id, profile, arena_elo)
        if arena_resolved is not None:
            arena_by_category, arena_source, arena_inferred_from = arena_resolved
            per_cat: dict[str, ArenaEloScore] = {}
            for category, rating in sorted(arena_by_category.items()):
                cat_entry: ArenaEloScore = {
                    "raw": int(round(rating)),
                    "tier": _arena_elo_tier(rating),
                    "plain": (
                        _arena_elo_plain(rating, category)
                        if arena_source == "direct"
                        else _arena_elo_plain_inferred(rating, category, arena_inferred_from or "")
                    ),
                    "source": arena_source,
                }
                if arena_inferred_from:
                    cat_entry["inferred_from"] = arena_inferred_from
                per_cat[category] = cat_entry
            scores["arena_elo"] = per_cat

        models.append(
            {
                "nim_model": nim_id,
                "architecture_note": profile["architecture_note"],
                "strong_at": profile["strong_at"],
                "watch_out_for": profile["watch_out_for"],
                "best_deployment": profile["best_deployment"],
                "primary_benchmarks": profile["primary_benchmarks"],
                "intent_hints": profile.get("intent_hints", []),
                "scores": scores,
            }
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "schema_version": "6",
        "sources": {
            "bfcl": BFCL_CSV_URL,
            "arena_elo": ARENA_ELO_ROWS_URL,
        },
        "benchmarks": BENCHMARK_REGISTRY,
        "models": models,
        "upstream_index": {
            "bfcl_v4": bfcl,
            "arena_elo": arena_elo,
        },
        "namespace_to_type": NAMESPACE_TO_TYPE,
        "name_decomposition_rules": NAME_DECOMPOSITION_RULES,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def load_existing_cache(path: Path) -> Cache | None:
    """Read the cache that's already on disk, or return None if there isn't one."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  Existing cache at {path} unreadable ({exc}); starting fresh.")
        return None
    return data


def merge_cache(existing: Cache | None, fresh: Cache) -> Cache:
    """Overlay `fresh` onto `existing` — fresh wins per category, existing fills gaps.

    Each refresh may be cut short by rate limits and only capture a subset of the
    22 Arena categories. Without merge, every partial run wipes the richer cache
    from previous runs. With merge, multiple runs over time converge on full
    coverage; a single bad run can't make the cache worse.
    """
    if existing is None or existing.get("schema_version") != fresh["schema_version"]:
        return fresh

    existing_by_id = {m["nim_model"]: m for m in existing.get("models", [])}

    merged_models: list[CacheModelEntry] = []
    for fresh_m in fresh["models"]:
        nim_id = fresh_m["nim_model"]
        old_m = existing_by_id.get(nim_id)
        if old_m is None:
            merged_models.append(fresh_m)
            continue

        old_scores = old_m.get("scores", {})
        new_scores = fresh_m.get("scores", {})
        merged_scores: ModelScores = {}

        # BFCL is a single number per model — fresh wins if present, else keep old
        if "bfcl_v4" in new_scores:
            merged_scores["bfcl_v4"] = new_scores["bfcl_v4"]
        elif "bfcl_v4" in old_scores:
            merged_scores["bfcl_v4"] = old_scores["bfcl_v4"]

        # Arena Elo is per-category — overlay fresh categories onto cached ones
        merged_arena: dict[str, ArenaEloScore] = {}
        if "arena_elo" in old_scores:
            merged_arena.update(old_scores["arena_elo"])
        if "arena_elo" in new_scores:
            merged_arena.update(new_scores["arena_elo"])
        if merged_arena:
            merged_scores["arena_elo"] = merged_arena

        # Static descriptive fields always come from the current registry, not the cache
        merged_models.append(
            {
                "nim_model": nim_id,
                "architecture_note": fresh_m["architecture_note"],
                "strong_at": fresh_m["strong_at"],
                "watch_out_for": fresh_m["watch_out_for"],
                "best_deployment": fresh_m["best_deployment"],
                "primary_benchmarks": fresh_m["primary_benchmarks"],
                "intent_hints": fresh_m.get("intent_hints", []),
                "scores": merged_scores,
            }
        )

    # Upstream tables also merge: fresh wins per key, old fills gaps when the
    # current fetch was partial (rate-limited).
    merged_upstream: UpstreamIndex = {
        "bfcl_v4": {**existing.get("upstream_index", {}).get("bfcl_v4", {}), **fresh["upstream_index"]["bfcl_v4"]},
        "arena_elo": {**existing.get("upstream_index", {}).get("arena_elo", {})},
    }
    # Per-model arena dicts need their own merge so a partial fetch doesn't drop
    # categories the cache previously had for the same model.
    for model_name, fresh_cats in fresh["upstream_index"]["arena_elo"].items():
        merged_upstream["arena_elo"][model_name] = {
            **merged_upstream["arena_elo"].get(model_name, {}),
            **fresh_cats,
        }

    return {
        "generated_at": fresh["generated_at"],
        "schema_version": fresh["schema_version"],
        "sources": fresh["sources"],
        "benchmarks": fresh["benchmarks"],
        "models": merged_models,
        "upstream_index": merged_upstream,
        # Reference tables come fresh from the registry, not the cache — these are editorial
        "namespace_to_type": fresh["namespace_to_type"],
        "name_decomposition_rules": fresh["name_decomposition_rules"],
    }


def write_cache(cache: Cache, path: Path, dry_run: bool) -> None:
    payload = json.dumps(cache, indent=2)
    if dry_run:
        print("\n--- DRY RUN OUTPUT ---")
        print(payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as f:
            f.write(payload)
            tmp = Path(f.name)
        tmp.replace(path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print(f"\n  Cache written to {path}")


def print_summary(cache: Cache) -> None:
    print("\nModel capability summary (raw value, tier, source) — '*' marks inferred-from-ancestor:")
    print(f"  {'NIM model':<45} {'BFCL':<16} {'overall':<16} {'coding':<16} {'hard_prompts'}")
    print("  " + "-" * 110)
    for m in cache["models"]:

        def _bfcl_cell(score: BfclScore | None) -> str:
            if not score:
                return "—"
            marker = "*" if score["source"] != "direct" else ""
            return f"{score['percent']} ({score['tier']}){marker}"

        def _arena_cell(cat: str) -> str:
            s = m["scores"].get("arena_elo", {}).get(cat)
            if not s:
                return "—"
            marker = "*" if s["source"] != "direct" else ""
            return f"{s['raw']} ({s['tier']}){marker}"

        bfcl_cell = _bfcl_cell(m["scores"].get("bfcl_v4"))
        print(
            f"  {m['nim_model']:<45} {bfcl_cell:<16} "
            f"{_arena_cell('overall'):<16} {_arena_cell('coding'):<16} {_arena_cell('hard_prompts')}"
        )
        if m["strong_at"]:
            hint = m["strong_at"][0]
            print(f"    → {hint[:72]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Cache output path (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cache JSON to stdout without writing",
    )
    args = parser.parse_args()

    print("Refreshing benchmark cache...")

    bfcl = fetch_bfcl_scores()
    arena_elo = fetch_arena_elo_scores()

    if not bfcl and not arena_elo:
        print("\nERROR: All upstream fetches failed. Check network access.", file=sys.stderr)
        print("The existing cache (if any) is unchanged.", file=sys.stderr)
        return 1

    fresh = build_cache(bfcl, arena_elo)
    existing = load_existing_cache(args.output)
    cache = merge_cache(existing, fresh)
    if existing is not None:
        # Quick diagnostic: how many arena categories landed across all models
        old_cats = sum(len(m.get("scores", {}).get("arena_elo", {})) for m in existing.get("models", []))
        new_cats = sum(len(m.get("scores", {}).get("arena_elo", {})) for m in cache["models"])
        print(f"  Cache merge: {old_cats} → {new_cats} category entries across all models.")
    write_cache(cache, args.output, dry_run=args.dry_run)
    print_summary(cache)

    missing = [
        m["nim_model"] for m in cache["models"] if "bfcl_v4" not in m["scores"] and "arena_elo" not in m["scores"]
    ]
    if missing:
        print(f"\n  NOTE: No upstream scores found for {len(missing)} model(s):")
        for m in missing:
            print(f"    - {m}")
        print("  Static capability descriptions are still available in the cache.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
