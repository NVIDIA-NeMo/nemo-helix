// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getTestCaseVerdict } from '@studio/components/evaluation/testCaseVerdict';

describe('getTestCaseVerdict', () => {
  it('passes when every evaluator scored 1', () => {
    expect(
      getTestCaseVerdict({ status: 'success', evaluator_scores: { exact_match: 1, judge: 1 } })
    ).toBe('passed');
  });

  it('fails a successful run that scored below 1', () => {
    expect(
      getTestCaseVerdict({ status: 'success', evaluator_scores: { exact_match: 1, judge: 0.5 } })
    ).toBe('failed');
  });

  it('reads scores above 1 as percentages', () => {
    expect(
      getTestCaseVerdict({ status: 'success', evaluator_scores: { check: 1, judge: 100 } })
    ).toBe('passed');
    expect(
      getTestCaseVerdict({ status: 'success', evaluator_scores: { check: 1, judge: 50 } })
    ).toBe('failed');
  });

  it('reports an error when the run errored', () => {
    expect(getTestCaseVerdict({ status: 'error', evaluator_scores: { judge: 1 } })).toBe('error');
  });

  it('reports an error when an evaluator failed to score', () => {
    expect(
      getTestCaseVerdict({
        status: 'success',
        evaluator_scores: { exact_match: 1 },
        failed_evaluators: ['judge'],
      })
    ).toBe('error');
  });

  it('is not scored without any finite score', () => {
    expect(getTestCaseVerdict({ status: 'success', evaluator_scores: {} })).toBe('not-scored');
    expect(getTestCaseVerdict({ status: 'success' })).toBe('not-scored');
  });
});
