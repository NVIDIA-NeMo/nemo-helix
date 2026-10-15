// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  computeCompletionMetrics,
  getReasoningDelta,
  getUsageCompletionTokens,
} from '@nemo/common/src/components/AssistantChat/completionMetrics';

describe('getReasoningDelta', () => {
  it('reads reasoning_content', () => {
    expect(getReasoningDelta({ choices: [{ delta: { reasoning_content: 'hmm' } }] })).toBe('hmm');
  });

  it('reads reasoning', () => {
    expect(getReasoningDelta({ choices: [{ delta: { reasoning: 'hmm' } }] })).toBe('hmm');
  });

  it('returns empty string for content-only and usage-only chunks', () => {
    expect(getReasoningDelta({ choices: [{ delta: { content: 'hi' } }] })).toBe('');
    expect(getReasoningDelta({ choices: [] })).toBe('');
    expect(getReasoningDelta(null)).toBe('');
  });
});

describe('getUsageCompletionTokens', () => {
  it('reads usage.completion_tokens', () => {
    expect(getUsageCompletionTokens({ usage: { completion_tokens: 42 } })).toBe(42);
  });

  it('ignores missing, null, and zero usage', () => {
    expect(getUsageCompletionTokens({})).toBeUndefined();
    expect(getUsageCompletionTokens({ usage: null })).toBeUndefined();
    expect(getUsageCompletionTokens({ usage: { completion_tokens: 0 } })).toBeUndefined();
  });
});

describe('computeCompletionMetrics', () => {
  it('prefers usage tokens over the character estimate', () => {
    const metrics = computeCompletionMetrics({
      totalMs: 10_000,
      firstTokenMs: 1_000,
      usageTokens: 900,
      outputChars: 40,
      chunkCount: 10,
    });
    expect(metrics.completionTokens).toBe(900);
  });

  it('measures throughput from first token to end', () => {
    const metrics = computeCompletionMetrics({
      totalMs: 10_000,
      firstTokenMs: 1_000,
      usageTokens: 900,
      outputChars: 0,
      chunkCount: 0,
    });
    expect(metrics.tokensPerSec).toBeCloseTo(100);
  });

  it('estimates tokens from characters when usage is missing', () => {
    const metrics = computeCompletionMetrics({
      totalMs: 2_000,
      firstTokenMs: 0,
      outputChars: 400,
      chunkCount: 5,
    });
    expect(metrics.completionTokens).toBe(100);
    expect(metrics.tokensPerSec).toBeCloseTo(50);
  });

  it('falls back to total time when the generation window is tiny', () => {
    const metrics = computeCompletionMetrics({
      totalMs: 2_000,
      firstTokenMs: 1_990,
      usageTokens: 200,
      outputChars: 0,
      chunkCount: 1,
    });
    expect(metrics.tokensPerSec).toBeCloseTo(100);
  });
});
