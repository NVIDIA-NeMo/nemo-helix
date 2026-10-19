// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  evalRunOutcomeLabel,
  evalRunOutcomeOf,
  evalRunOutcomeSeverity,
} from '@studio/api/evaluation/runOutcome';

const failedBlock = {
  unit: 'trials',
  total: 28,
  errored: 28,
  scored: 0,
  failed: true,
  message: 'No usable scores across 28 trials (28 reported errors).',
};

describe('evalRunOutcomeOf', () => {
  it('reads the evaluator rollup from status_details.evaluation', () => {
    expect(
      evalRunOutcomeOf({ message: 'Job exited with code 1', evaluation: failedBlock })
    ).toEqual(failedBlock);
  });

  it('returns null for jobs without the block, including non-evaluator jobs', () => {
    expect(evalRunOutcomeOf(undefined)).toBeNull();
    expect(evalRunOutcomeOf({ message: 'Job completed successfully' })).toBeNull();
    expect(evalRunOutcomeOf({ evaluation: 'oops' })).toBeNull();
  });

  it('rejects a malformed block rather than rendering nonsense', () => {
    expect(evalRunOutcomeOf({ evaluation: { ...failedBlock, unit: 'rounds' } })).toBeNull();
    expect(evalRunOutcomeOf({ evaluation: { ...failedBlock, scored: -1 } })).toBeNull();
    expect(evalRunOutcomeOf({ evaluation: { ...failedBlock, failed: 'true' } })).toBeNull();
    expect(evalRunOutcomeOf({ evaluation: { ...failedBlock, message: undefined } })).toBeNull();
  });
});

describe('evalRunOutcomeSeverity', () => {
  it('is an error for a failed run', () => {
    expect(evalRunOutcomeSeverity(evalRunOutcomeOf({ evaluation: failedBlock })!)).toBe('error');
  });

  it('is a warning when some units errored or went unscored on a completed run', () => {
    const partial = { ...failedBlock, errored: 3, scored: 25, failed: false };
    expect(evalRunOutcomeSeverity(evalRunOutcomeOf({ evaluation: partial })!)).toBe('warning');
    const unscored = { ...failedBlock, errored: 0, scored: 27, failed: false };
    expect(evalRunOutcomeSeverity(evalRunOutcomeOf({ evaluation: unscored })!)).toBe('warning');
  });

  it('is nothing for a clean run, so healthy jobs stay quiet', () => {
    const clean = { ...failedBlock, errored: 0, scored: 28, failed: false };
    expect(evalRunOutcomeSeverity(evalRunOutcomeOf({ evaluation: clean })!)).toBeNull();
  });
});

describe('evalRunOutcomeLabel', () => {
  it('reads "scored of total"', () => {
    expect(evalRunOutcomeLabel(evalRunOutcomeOf({ evaluation: failedBlock })!)).toBe(
      '0 of 28 scored'
    );
  });
});
