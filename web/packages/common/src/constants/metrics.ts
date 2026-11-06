// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING = '{{sample.output_text | trim}}';

export const METRIC_NAMES_API = ['llm-judge', 'f1', 'bleu', 'rouge', 'string-check', 'em'] as const;
export type MetricNameApi = (typeof METRIC_NAMES_API)[number];

/**
 * Valid score types for LLM Judge metrics
 * Based on the Evaluator API validation
 */
export const LLM_JUDGE_SCORE_TYPES = ['number', 'integer', 'boolean'] as const;

export const DEFAULT_LLM_JUDGE_DEFAULTS = {
  systemMessage: 'Your task is to evaluate the semantic similarity between two responses.',
  userMessage: `Respond in the following format SIMILARITY: 4.
The similarity should be a score between 0 and 10.
RESPONSE 1: {{response1}}
RESPONSE 2: {{response2}}`,
  similarityScoreType: 'integer' as const,
  similarityScoreParserPattern: 'SIMILARITY: (\\d*)',
};

// Human-readable labels for metrics
export const METRIC_LABELS: Record<MetricNameApi, string> = {
  bleu: 'BLEU',
  'string-check': 'STRING CHECK',
  rouge: 'ROUGE',
  em: 'EXACT MATCH',
  f1: 'F1',
  'llm-judge': 'LLM-AS-A-JUDGE',
} as const;
