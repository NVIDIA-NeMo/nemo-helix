// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TOOL_CALLING_RESULT_FIXTURE } from '@studio/mocks/handlers/toolCallingRun';
import {
  deltaDirection,
  formatDelta,
  formatMetric,
  parseToolCallingResult,
  rejectionReasonRows,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/toolCalling/toolCallingResults';

describe('parseToolCallingResult', () => {
  it('reads a result that matches the contract', () => {
    const result = parseToolCallingResult(JSON.stringify(TOOL_CALLING_RESULT_FIXTURE));

    expect(result.variants.map((variant) => variant.id)).toEqual([
      'baseline',
      'config-fix',
      'schema-fix',
    ]);
    expect(result.findings[1]?.tool).toBeNull();
  });

  it('fills defaults for optional lists and accepts a baseline-only run', () => {
    const [baseline] = TOOL_CALLING_RESULT_FIXTURE.variants;
    const result = parseToolCallingResult(
      JSON.stringify({
        version: 1,
        agent: 'default/a',
        variants: [{ id: 'baseline', label: 'Current agent', metrics: baseline?.metrics }],
      })
    );

    expect(result.variants[0]?.rejection_reasons).toEqual({});
    expect(result.variants[0]?.examples).toEqual([]);
    expect(result.findings).toEqual([]);
    expect(result.suggestions).toEqual([]);
    expect(result.recommended_variant).toBeUndefined();
  });

  it('reads the recommended variant, including an explicit null', () => {
    expect(
      parseToolCallingResult(JSON.stringify(TOOL_CALLING_RESULT_FIXTURE)).recommended_variant
    ).toBe('config-fix');
    expect(
      parseToolCallingResult(
        JSON.stringify({ ...TOOL_CALLING_RESULT_FIXTURE, recommended_variant: null })
      ).recommended_variant
    ).toBeNull();
  });

  it('rejects text that is not JSON', () => {
    expect(() => parseToolCallingResult('{')).toThrow(
      'tool-calling-result.json is not valid JSON.'
    );
  });

  it('names where the shape is wrong', () => {
    const [baseline, ...rest] = TOOL_CALLING_RESULT_FIXTURE.variants;
    const bad = {
      ...TOOL_CALLING_RESULT_FIXTURE,
      variants: [{ ...baseline, metrics: { ...baseline?.metrics, tool_error_rate: 80 } }, ...rest],
    };

    expect(() => parseToolCallingResult(JSON.stringify(bad))).toThrow(
      /at variants\.0\.metrics\.tool_error_rate/
    );
  });
});

describe('metric formatting', () => {
  it.each([
    [0.8, 'rate', '80%'],
    [4.04, 'decimal', '4.0'],
    [1, 'count', '1'],
  ] as const)('formats %s as a %s', (value, format, expected) => {
    expect(formatMetric(value, format)).toBe(expected);
  });

  it('signs deltas and reads rate deltas in points', () => {
    expect(formatDelta(0.059 - 0.8, 'rate')).toBe('−74 pts');
    expect(formatDelta(0.6, 'rate')).toBe('+60 pts');
    expect(formatDelta(0, 'count')).toBe('0');
  });

  it('judges a delta by whether lower is better', () => {
    expect(deltaDirection(-0.7, true)).toBe('better');
    expect(deltaDirection(-0.7, false)).toBe('worse');
    expect(deltaDirection(0, true)).toBe('same');
  });
});

describe('rejectionReasonRows', () => {
  it('merges reasons across variants, most frequent in the baseline first', () => {
    expect(rejectionReasonRows(TOOL_CALLING_RESULT_FIXTURE.variants)).toEqual([
      {
        reason: 'corrupted or unknown key',
        counts: { baseline: 22, 'config-fix': 14, 'schema-fix': 1 },
      },
      { reason: 'model/provider not on platform', counts: { baseline: 12 } },
    ]);
  });
});
