// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  countAttacks,
  countBenign,
  formatCounts,
} from '@agent-hardener/components/SanityCheckReport';
import type {
  ValidationAttackRow,
  ValidationBenignRow,
  ValidationSummary,
} from '@agent-hardener/components/useSanityCheck';

const summary = (over: Partial<ValidationSummary> = {}): ValidationSummary => ({
  attacks_total: 0,
  attacks_blocked: 0,
  benign_total: 0,
  benign_false_positives: 0,
  ...over,
});

const benignRow = (status?: string, outcome?: string): ValidationBenignRow =>
  ({ status, outcome } as ValidationBenignRow);
const attackRow = (status?: string, outcome?: string): ValidationAttackRow =>
  ({ status, outcome } as ValidationAttackRow);

describe('countBenign', () => {
  // The shape that exposed the bug: job agent-hardener-fgrr8adq reported 8/9 preserved while
  // agent-hardener's own log said 7/9 complied (1 refused, 1 errors).
  it('does not count an errored row as preserved', () => {
    const rows = [
      ...Array.from({ length: 7 }, () => benignRow('passed')),
      benignRow('refused'),
      benignRow('error'),
    ];

    expect(countBenign(rows, summary({ benign_total: 9, benign_false_positives: 1 }))).toEqual({
      ok: 7,
      bad: 1,
      errored: 1,
      notCounted: 0,
      total: 9,
    });
  });

  it('reports a clean run as all preserved', () => {
    const rows = Array.from({ length: 3 }, () => benignRow('passed'));

    expect(countBenign(rows, summary({ benign_total: 3 }))).toEqual({
      ok: 3,
      bad: 0,
      errored: 0,
      notCounted: 0,
      total: 3,
    });
  });

  it('treats a missing status as inconclusive rather than passed', () => {
    expect(countBenign([benignRow('passed'), benignRow(undefined)], summary())).toEqual({
      ok: 1,
      bad: 0,
      errored: 1,
      notCounted: 0,
      total: 2,
    });
  });

  it('falls back to the summary when the artifact carries no rows', () => {
    expect(countBenign([], summary({ benign_total: 5, benign_false_positives: 2 }))).toEqual({
      ok: 3,
      bad: 2,
      errored: 0,
      notCounted: 0,
      total: 5,
    });
  });
});

describe('countAttacks', () => {
  it('separates an errored attack from one that was cleanly not blocked', () => {
    expect(countAttacks([attackRow('error')], summary({ attacks_total: 1 }))).toEqual({
      ok: 0,
      bad: 0,
      errored: 1,
      notCounted: 0,
      total: 1,
    });

    expect(countAttacks([attackRow('not_blocked')], summary({ attacks_total: 1 }))).toEqual({
      ok: 0,
      bad: 1,
      errored: 0,
      notCounted: 0,
      total: 1,
    });
  });

  it('falls back to the summary when the artifact carries no rows', () => {
    expect(countAttacks([], summary({ attacks_total: 4, attacks_blocked: 3 }))).toEqual({
      ok: 3,
      bad: 1,
      errored: 0,
      notCounted: 0,
      total: 4,
    });
  });
});

describe('counting by outcome (traces labels)', () => {
  it('counts each attack label by its outcome, not as an error', () => {
    const rows = [
      attackRow('blocked_guardrail', 'blocked'),
      attackRow('blocked_network', 'blocked'),
      attackRow('agent_refused', 'blocked'),
      attackRow('tool_failed', 'blocked'), // the args judge found no attack in the call
      attackRow('tool_not_called', 'landed'),
      attackRow('tool_failed', 'landed'), // the call carried the attack
      attackRow('error', 'error'),
    ];

    expect(countAttacks(rows, summary({ attacks_total: 7 }))).toEqual({
      ok: 4,
      bad: 2,
      errored: 1,
      notCounted: 0,
      total: 7,
    });
  });

  it('counts a guardrail or network block of a benign request as a false positive', () => {
    const rows = [
      benignRow('passed', 'served'),
      benignRow('tool_failed', 'served'), // the call did what was asked
      benignRow('blocked_guardrail', 'refused'),
      benignRow('blocked_network', 'refused'),
      benignRow('tool_failed', 'refused'),
    ];

    expect(countBenign(rows, summary({ benign_total: 5 }))).toEqual({
      ok: 2,
      bad: 3,
      errored: 0,
      notCounted: 0,
      total: 5,
    });
  });

  it('leaves a request the agent declined on its own out of the total', () => {
    const rows = [benignRow('passed', 'served'), benignRow('agent_refused', 'not_counted')];

    expect(countBenign(rows, summary({ benign_total: 2 }))).toEqual({
      ok: 1,
      bad: 0,
      errored: 0,
      notCounted: 1,
      total: 1,
    });
  });

  it('falls back to the summary counted by outcome when the artifact carries no rows', () => {
    const counted = summary({
      attacks_total: 6,
      attacks_blocked: 4,
      attacks_landed: 1,
      benign_total: 10,
      benign_false_positives: 1, // understates: a refused failed call is not in it
      benign_served: 7,
      benign_refused: 2,
      benign_not_counted: 1,
    });

    expect(countAttacks([], counted)).toEqual({ ok: 4, bad: 1, errored: 1, notCounted: 0, total: 6 });
    expect(countBenign([], counted)).toEqual({ ok: 7, bad: 2, errored: 0, notCounted: 1, total: 9 });
  });
});

describe('formatCounts', () => {
  it('stays terse when everything is good', () => {
    expect(formatCounts({ ok: 9, bad: 0, errored: 0, notCounted: 0, total: 9 }, 'false positive')).toBe('9 / 9');
  });

  it('appends only the non-zero segments', () => {
    expect(formatCounts({ ok: 7, bad: 1, errored: 1, notCounted: 0, total: 9 }, 'false positive')).toBe(
      '7 / 9 · 1 false positive · 1 error'
    );
    expect(formatCounts({ ok: 0, bad: 0, errored: 1, notCounted: 0, total: 1 }, 'not blocked')).toBe(
      '0 / 1 · 1 error'
    );
  });

  it('shows the requests set aside', () => {
    expect(formatCounts({ ok: 7, bad: 1, errored: 0, notCounted: 2, total: 8 }, 'false positive')).toBe(
      '7 / 8 · 1 false positive · 2 not counted'
    );
  });

  it('pluralises', () => {
    expect(formatCounts({ ok: 4, bad: 2, errored: 3, notCounted: 0, total: 9 }, 'false positive')).toBe(
      '4 / 9 · 2 false positives · 3 errors'
    );
  });
});
