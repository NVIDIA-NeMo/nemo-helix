// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  sinceFor,
  toDateTimeLocalValue,
} from '@studio/routes/agents/AgentDetailRoute/analysis/analysisSince';

const now = new Date('2026-10-09T12:00:00Z');

describe('sinceFor', () => {
  it('reads a last-run timestamp without a zone as UTC', () => {
    expect(sinceFor('last-run', { now, lastRunAt: '2026-10-09T02:10:13.237316' })).toBe(
      '2026-10-09T02:10:13.237Z'
    );
  });

  it('counts the day and week presets back from now', () => {
    expect(sinceFor('day', { now })).toBe('2026-10-08T12:00:00.000Z');
    expect(sinceFor('week', { now })).toBe('2026-10-02T12:00:00.000Z');
  });

  it('has no bound for all history, a missing last run, or an unusable custom value', () => {
    expect(sinceFor('all', { now })).toBeUndefined();
    expect(sinceFor('last-run', { now })).toBeUndefined();
    expect(sinceFor('custom', { now, custom: '' })).toBeUndefined();
    expect(sinceFor('custom', { now, custom: 'not a date' })).toBeUndefined();
  });

  it('round-trips a custom value through the datetime-local format', () => {
    const local = toDateTimeLocalValue(new Date(2026, 9, 1, 8, 5));
    expect(local).toBe('2026-10-01T08:05');
    expect(sinceFor('custom', { now, custom: local })).toBe(
      new Date(2026, 9, 1, 8, 5).toISOString()
    );
  });
});
