// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TraceStatisticsBucket } from '@studio/components/AgentTraceStatistics/types';
import { bucketParamForRange, fillBucketGaps } from '@studio/components/AgentTraceStatistics/utils';

const at = (
  iso: string,
  overrides: Partial<TraceStatisticsBucket> = {}
): TraceStatisticsBucket => ({
  timestamp: new Date(iso).getTime(),
  costUsd: 0.01,
  tokens: 100,
  latencyMs: 1000,
  ...overrides,
});

describe('bucketParamForRange', () => {
  it('asks Intake for hourly buckets on the day range, daily on week/month, weekly on max', () => {
    expect(bucketParamForRange('day')).toBe('hour');
    expect(bucketParamForRange('week')).toBe('day');
    expect(bucketParamForRange('month')).toBe('day');
    expect(bucketParamForRange('max')).toBe('week');
  });
});

describe('fillBucketGaps', () => {
  it('emits null for gaps so the line breaks rather than dipping to zero', () => {
    const buckets = fillBucketGaps(
      [at('2026-07-01T00:00:00Z'), at('2026-07-04T00:00:00Z')],
      'week'
    );

    expect(buckets).toHaveLength(4);
    expect(buckets[1].tokens).toBeNull();
    expect(buckets[2].tokens).toBeNull();
    expect(buckets[3].tokens).toBe(100);
  });

  it('fills hourly gaps on the day range', () => {
    const buckets = fillBucketGaps([at('2026-07-01T02:00:00Z'), at('2026-07-01T04:00:00Z')], 'day');

    expect(buckets).toHaveLength(3);
    expect(buckets[1].latencyMs).toBeNull();
  });

  it('orders points Intake returned out of sequence', () => {
    const buckets = fillBucketGaps(
      [at('2026-07-02T00:00:00Z'), at('2026-07-01T00:00:00Z')],
      'week'
    );

    expect(buckets.map((bucket) => bucket.timestamp)).toEqual([
      new Date('2026-07-01T00:00:00Z').getTime(),
      new Date('2026-07-02T00:00:00Z').getTime(),
    ]);
  });

  it('leaves a single bucket alone', () => {
    expect(fillBucketGaps([at('2026-07-01T00:00:00Z')], 'month')).toHaveLength(1);
  });

  it('returns no points for no buckets', () => {
    expect(fillBucketGaps([], 'month')).toEqual([]);
  });

  describe('across a DST transition', () => {
    const originalTz = process.env.TZ;

    beforeAll(() => {
      process.env.TZ = 'America/New_York';
    });

    afterAll(() => {
      if (originalTz === undefined) {
        delete process.env.TZ;
      } else {
        process.env.TZ = originalTz;
      }
    });

    it('keeps daily buckets on local midnight when the clocks spring forward', () => {
      // 2026-03-08 is the US spring-forward date: those local midnights are 23 hours apart.
      const march7 = at('2026-03-07T05:00:00Z');
      const march9 = at('2026-03-09T04:00:00Z');

      const buckets = fillBucketGaps([march7, march9], 'week');

      expect(buckets.map((bucket) => bucket.timestamp)).toEqual([
        march7.timestamp,
        new Date('2026-03-08T05:00:00Z').getTime(),
        march9.timestamp,
      ]);
      expect(buckets[1].tokens).toBeNull();
      expect(buckets[2].tokens).toBe(100);
    });

    it('keeps daily buckets on local midnight when the clocks fall back', () => {
      // 2026-11-01 is the US fall-back date: those local midnights are 25 hours apart.
      const oct31 = at('2026-10-31T04:00:00Z');
      const nov2 = at('2026-11-02T05:00:00Z');

      const buckets = fillBucketGaps([oct31, nov2], 'week');

      expect(buckets.map((bucket) => bucket.timestamp)).toEqual([
        oct31.timestamp,
        new Date('2026-11-01T04:00:00Z').getTime(),
        nov2.timestamp,
      ]);
      expect(buckets[2].tokens).toBe(100);
    });
  });
});
