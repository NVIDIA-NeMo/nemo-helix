// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SpanKind } from '@nemo/sdk/generated/platform/schema/SpanKind';
import { agentSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/AgentSpanTemplate';
import { chainSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/ChainSpanTemplate';
import { defaultSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/DefaultSpanTemplate';
import { embeddingSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/EmbeddingSpanTemplate';
import { evaluatorSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/EvaluatorSpanTemplate';
import { guardrailSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/GuardrailSpanTemplate';
import { llmSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/LlmSpanTemplate';
import { rerankerSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/RerankerSpanTemplate';
import { retrieverSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/RetrieverSpanTemplate';
import { toolSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/ToolSpanTemplate';
import type { SpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/types';

/**
 * Per-kind template registry — pure wiring. Each kind's behavior (label, icon,
 * sections, row-header title/badge, claimed attribute namespaces) lives in its
 * own `*SpanTemplate.ts` descriptor beside its `*SpanContent.tsx`. Add a kind by
 * implementing those two files and registering the descriptor here.
 *
 * UNKNOWN spans intentionally have no template — they degrade to the generic
 * Input/Output/Metadata sections.
 */
const SPAN_TEMPLATES: Partial<Record<SpanKind, SpanTemplate>> = {
  [SpanKind.LLM]: llmSpanTemplate,
  [SpanKind.TOOL]: toolSpanTemplate,
  [SpanKind.RETRIEVER]: retrieverSpanTemplate,
  [SpanKind.EMBEDDING]: embeddingSpanTemplate,
  [SpanKind.AGENT]: agentSpanTemplate,
  [SpanKind.RERANKER]: rerankerSpanTemplate,
  [SpanKind.EVALUATOR]: evaluatorSpanTemplate,
  [SpanKind.GUARDRAIL]: guardrailSpanTemplate,
  [SpanKind.CHAIN]: chainSpanTemplate,
  [SpanKind.UNKNOWN]: defaultSpanTemplate,
};

/** Always returns a template — kinds without a dedicated one fall back to the generic default. */
export const getSpanTemplate = (kind: SpanKind): SpanTemplate =>
  SPAN_TEMPLATES[kind] ?? defaultSpanTemplate;
