// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const AVG_CHARS_PER_TOKEN = 4;
const MIN_GENERATION_WINDOW_MS = 100;

interface CompletionMetricsInput {
  readonly totalMs: number;
  readonly firstTokenMs: number;
  readonly usageTokens?: number;
  readonly outputChars: number;
  readonly chunkCount: number;
}

interface CompletionMetrics {
  readonly completionTokens: number;
  readonly tokensPerSec: number;
}

export const getReasoningDelta = (chunk: unknown): string => {
  if (typeof chunk !== 'object' || chunk === null || !('choices' in chunk)) return '';
  const firstChoice = Array.isArray(chunk.choices) ? chunk.choices[0] : undefined;
  if (typeof firstChoice !== 'object' || firstChoice === null || !('delta' in firstChoice)) {
    return '';
  }
  const { delta } = firstChoice;
  if (typeof delta !== 'object' || delta === null) return '';
  if ('reasoning_content' in delta && typeof delta.reasoning_content === 'string') {
    return delta.reasoning_content;
  }
  if ('reasoning' in delta && typeof delta.reasoning === 'string') return delta.reasoning;
  return '';
};

export const getUsageCompletionTokens = (completion: unknown): number | undefined => {
  if (typeof completion !== 'object' || completion === null || !('usage' in completion)) {
    return undefined;
  }
  const { usage } = completion;
  if (typeof usage !== 'object' || usage === null || !('completion_tokens' in usage)) {
    return undefined;
  }
  const tokens = usage.completion_tokens;
  return typeof tokens === 'number' && Number.isFinite(tokens) && tokens > 0 ? tokens : undefined;
};

export const computeCompletionMetrics = ({
  totalMs,
  firstTokenMs,
  usageTokens,
  outputChars,
  chunkCount,
}: CompletionMetricsInput): CompletionMetrics => {
  const completionTokens =
    usageTokens ?? Math.max(chunkCount, Math.round(outputChars / AVG_CHARS_PER_TOKEN));
  const generationMs = totalMs - firstTokenMs;
  const windowMs =
    firstTokenMs > 0 && generationMs >= MIN_GENERATION_WINDOW_MS ? generationMs : totalMs;
  return {
    completionTokens,
    tokensPerSec: (completionTokens * 1000) / Math.max(1, windowMs),
  };
};
