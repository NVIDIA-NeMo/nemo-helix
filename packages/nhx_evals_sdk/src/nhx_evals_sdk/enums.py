# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enums for evals SDK runtime."""

from enum import Enum


class MetricType(str, Enum):
    """The predefined metric types."""

    BLEU = "bleu"
    ROUGE = "rouge"
    F1 = "f1"
    SEQUENCE_SIMILARITY = "sequence-similarity"
    EXACT_MATCH = "exact-match"
    STRING_CHECK = "string-check"
    NUMBER_CHECK = "number-check"
    LLM_JUDGE = "llm-judge"
    TOOL_CALLING = "tool-calling"
    REMOTE = "remote"
    NEMO_AGENT_TOOLKIT_REMOTE = "nemo-agent-toolkit-remote"
    RETRIEVAL_NDCG = "retrieval-ndcg"
    RETRIEVAL_RECALL = "retrieval-recall"
    RETRIEVAL_PRECISION = "retrieval-precision"
    RETRIEVAL_MAP = "retrieval-map"

    TOPIC_ADHERENCE = "topic_adherence"
    TOOL_CALL_ACCURACY = "tool_call_accuracy"
    AGENT_GOAL_ACCURACY = "agent_goal_accuracy"

    ANSWER_ACCURACY = "answer_accuracy"
    CONTEXT_RELEVANCE = "context_relevance"
    RESPONSE_GROUNDEDNESS = "response_groundedness"

    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_ENTITY_RECALL = "context_entity_recall"
    RESPONSE_RELEVANCY = "response_relevancy"
    FAITHFULNESS = "faithfulness"
    NOISE_SENSITIVITY = "noise_sensitivity"
    TUNABLE_RAG_EVALUATOR = "tunable-rag-evaluator"

    # Runner-owned rewards: the runner scores, these surface it.
    GYM_REWARD = "gym_reward"
    HARBOR_REWARD = "harbor_reward"

    # Agent-eval scoring over trial metadata and evidence.
    AGENT_PHASE_SUCCESS = "agent_phase_success"
    EVIDENCE_PRESENCE = "evidence_presence"
    SKILL_USED = "skill_used"
    TOOL_CALL_COUNT = "tool_call_count"
    TOOL_ARGUMENT_MATCHES_INPUT = "tool_argument_matches_input"

    # Skill artifact readiness and trial scoring.
    SKILL_STRUCTURE = "skill_structure"
    SKILL_REFERENCE_INTEGRITY = "skill_reference_integrity"
    SKILL_CONTEXT_EFFICIENCY = "skill_context_efficiency"
    SKILL_ACTIVATION = "skill_activation"
    TOKEN_EFFICIENCY = "token_efficiency"
    SKILL_STATIC_CORRECTNESS = "skill_static_correctness"
    SKILL_DESCRIPTION_QUALITY = "skill_description_quality"
    SKILL_RELIABILITY_READINESS = "skill_reliability_readiness"
    SKILL_SCRIPT_QUALITY = "skill_script_quality"
    SKILL_UNICODE_SAFETY = "skill_unicode_safety"
    SKILL_SECURITY_SCAN = "skill_security_scan"
    SKILL_SENSITIVE_DATA = "skill_sensitive_data"
    SKILL_CODE_RISK = "skill_code_risk"
    SKILL_LICENSE = "skill_license"
    SKILL_DEPENDENCY_COMPLIANCE = "skill_dependency_compliance"
    SKILL_PACKAGE_CONFIGURATION_RISK = "skill_package_configuration_risk"
    AGENT_SAFETY = "agent_safety"
    SKILL_BUNDLED_TEST = "skill_bundled_test"
    SKILL_OVERACTIVATION = "skill_overactivation"
    SKILL_ROUTING = "skill_routing"
    EXPECTED_SCRIPT_INVOCATION = "expected_script_invocation"
    WORKFLOW_COMPLIANCE = "workflow_compliance"
    ERROR_RECOVERY = "error_recovery"
    TOOL_PRODUCTIVITY = "tool_productivity"
    TOOL_SELECTION = "tool_selection"
    TOOL_ARGUMENT_COMPLIANCE = "tool_argument_compliance"
    TOOL_EXECUTION_OUTCOME = "tool_execution_outcome"
    COMPONENT_COORDINATION = "component_coordination"
    COMPONENT_USAGE = "component_usage"

    SYSTEM = "system"


class TaskStatus(str, Enum):
    """Status of an evaluation task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModelFormat(str, Enum):
    """Inference format for a model."""

    NVIDIA_NIM = "nim"
    OPEN_AI = "openai"
    LLAMA_STACK = "llama_stack"


class AgentFormat(str, Enum):
    """Inference format for an agent."""

    GENERIC = "generic"
    NEMO_AGENT_TOOLKIT = "nemo_agent_toolkit"
