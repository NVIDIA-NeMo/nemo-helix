// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatDurationMs, formatTimeInSeconds, getElapsedSeconds, utcToLocalDate } from './date';

describe('formatTimeInSeconds', () => {
  it('Returns empty string when seconds is undefined', () => {
    expect(formatTimeInSeconds(undefined)).toBe('');
  });

  it('Returns the correctly-formatted time string', () => {
    expect(formatTimeInSeconds(30)).toBe('30s');
    expect(formatTimeInSeconds(60)).toBe('1m');
    expect(formatTimeInSeconds(3600)).toBe('1h');
    expect(formatTimeInSeconds(3600 + 60)).toBe('1h 1m');
    expect(formatTimeInSeconds(3600 + 60 + 30)).toBe('1h 1m 30s');
    expect(formatTimeInSeconds(3600 * 25)).toBe('25h');
  });

  it('Never renders a sub-second component', () => {
    expect(formatTimeInSeconds(30.9)).toBe('30s');
  });

  it('Returns empty string for sub-second and negative values', () => {
    expect(formatTimeInSeconds(0.9)).toBe('');
    expect(formatTimeInSeconds(-5)).toBe('');
  });
});

describe('formatDurationMs', () => {
  it('Returns an em dash for null/undefined', () => {
    expect(formatDurationMs(null)).toBe('—');
    expect(formatDurationMs(undefined)).toBe('—');
  });

  it('Renders a single millisecond segment under one second', () => {
    expect(formatDurationMs(34)).toBe('34ms');
    expect(formatDurationMs(999)).toBe('999ms');
  });

  it('Renders seconds and milliseconds', () => {
    expect(formatDurationMs(12_034)).toBe('12s 34ms');
  });

  it('Renders minutes, seconds, and milliseconds', () => {
    expect(formatDurationMs(612_013)).toBe('10m 12s 13ms');
  });

  it('Includes hours for long durations', () => {
    expect(formatDurationMs(3_661_000)).toBe('1h 1m 1s');
  });

  it('Drops zero-valued units, including interior ones', () => {
    expect(formatDurationMs(12_000)).toBe('12s');
    expect(formatDurationMs(600_000)).toBe('10m');
    expect(formatDurationMs(3_600_000 + 5_000)).toBe('1h 5s');
  });

  it('Rounds fractional milliseconds', () => {
    expect(formatDurationMs(34.6)).toBe('35ms');
  });

  it('Keeps precision for sub-millisecond durations', () => {
    expect(formatDurationMs(0.34)).toBe('0.34ms');
    expect(formatDurationMs(0.5)).toBe('0.5ms');
  });

  it('Renders zero and negative values as 0ms', () => {
    expect(formatDurationMs(0)).toBe('0ms');
    expect(formatDurationMs(-5)).toBe('0ms');
  });

  describe('hideMsAboveMinute', () => {
    it('drops the ms component once the total reaches a minute', () => {
      expect(formatDurationMs(612_013, { hideMsAboveMinute: true })).toBe('10m 12s');
      expect(formatDurationMs(3_661_013, { hideMsAboveMinute: true })).toBe('1h 1m 1s');
      expect(formatDurationMs(60_034, { hideMsAboveMinute: true })).toBe('1m');
    });

    it('keeps ms for sub-minute durations', () => {
      expect(formatDurationMs(12_034, { hideMsAboveMinute: true })).toBe('12s 34ms');
      expect(formatDurationMs(999, { hideMsAboveMinute: true })).toBe('999ms');
    });
  });
});

describe('utcToLocalDate', () => {
  it('Returns undefined when input is undefined', () => {
    expect(utcToLocalDate(undefined)).toBeUndefined();
  });

  it('Returns undefined when input is an empty string', () => {
    expect(utcToLocalDate('')).toBeUndefined();
  });

  it('Returns undefined for invalid date strings', () => {
    expect(utcToLocalDate('not-a-date')).toBeUndefined();
    expect(utcToLocalDate('2025-13-45T99:99:99')).toBeUndefined();
  });

  it('Converts UTC ISO string without timezone indicator to Date', () => {
    const result = utcToLocalDate('2025-11-17T21:53:35.903780');
    expect(result).toBeInstanceOf(Date);
    expect(result?.toISOString()).toBe('2025-11-17T21:53:35.903Z');
  });

  it('Converts UTC ISO string with Z timezone indicator to Date', () => {
    const result = utcToLocalDate('2025-11-17T21:53:35.903780Z');
    expect(result).toBeInstanceOf(Date);
    expect(result?.toISOString()).toBe('2025-11-17T21:53:35.903Z');
  });

  it('Handles UTC ISO string with timezone offset', () => {
    const result = utcToLocalDate('2025-11-17T21:53:35.903780+00:00');
    expect(result).toBeInstanceOf(Date);
    expect(result?.toISOString()).toBe('2025-11-17T21:53:35.903Z');
  });

  it('Preserves timezone offset in the conversion', () => {
    // A date with +05:00 offset should be 5 hours earlier in UTC
    const result = utcToLocalDate('2025-11-17T21:53:35+05:00');
    expect(result).toBeInstanceOf(Date);
    expect(result?.toISOString()).toBe('2025-11-17T16:53:35.000Z');
  });

  it('Handles various ISO string formats', () => {
    // Without milliseconds
    const result1 = utcToLocalDate('2025-11-17T21:53:35');
    expect(result1?.toISOString()).toBe('2025-11-17T21:53:35.000Z');

    // With milliseconds
    const result2 = utcToLocalDate('2025-11-17T21:53:35.123');
    expect(result2?.toISOString()).toBe('2025-11-17T21:53:35.123Z');

    // With microseconds (JavaScript will truncate to milliseconds)
    const result3 = utcToLocalDate('2025-11-17T21:53:35.123456');
    expect(result3?.toISOString()).toBe('2025-11-17T21:53:35.123Z');
  });
});

describe('getElapsedSeconds', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('Returns whole seconds between start and end', () => {
    expect(getElapsedSeconds(0, 0)).toBe(0);
    expect(getElapsedSeconds(1_000, 4_000)).toBe(3);
    expect(getElapsedSeconds(0, 61_999)).toBe(61);
  });

  it('Defaults the end to now', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-11-17T21:53:35.900Z'));
    const start = new Date('2025-11-17T21:52:30.000Z').getTime();
    expect(getElapsedSeconds(start)).toBe(65);
  });
});
