// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatDateRange, parseUTCDateForPicker } from './formatDateRange';

describe('formatDateRange', () => {
  it('should format both start and end dates', () => {
    const result = formatDateRange('2024-01-01', '2024-12-31');
    expect(result).toContain('2024');
  });

  it('should format only start date', () => {
    const result = formatDateRange('2024-06-15');
    expect(result).toContain('2024');
  });

  it('should return empty string when no dates provided', () => {
    const result = formatDateRange();
    expect(result).toBe('');
  });
});

describe('parseUTCDateForPicker', () => {
  beforeEach(() => {
    vi.stubEnv('TZ', 'Asia/Shanghai');
  });
  afterEach(() => {
    vi.unstubAllEnvs();
  });
  it('should parse date at midnight UTC correctly', () => {
    const dateString = '2024-10-16T00:00:00.000Z';
    const result = parseUTCDateForPicker(dateString);

    // Verify the UTC components match the input date
    expect(result.getUTCFullYear()).toBe(2024);
    expect(result.getUTCMonth()).toBe(9); // October is month 9 (0-indexed)
    expect(result.getUTCDate()).toBe(16);
    expect(result.getUTCHours()).toBe(0);
    expect(result.getUTCMinutes()).toBe(0);
    expect(result.getUTCSeconds()).toBe(0);
    expect(result.getUTCMilliseconds()).toBe(0);
  });

  it('should produce consistent results regardless of local timezone', () => {
    // This test verifies that the UTC date components are always the same
    // regardless of what timezone the test is run in
    const dateString = '2024-10-16T00:00:00.000Z';
    const result = parseUTCDateForPicker(dateString);

    // The UTC date should always be October 16, 2024
    const expectedTimestamp = Date.UTC(2024, 9, 16, 0, 0, 0, 0);
    expect(result.getTime()).toBe(expectedTimestamp);
  });

  describe('comparison with naive Date constructor', () => {
    it('ensures parseUTCDateForPicker correctly parses UTC dates', () => {
      const dateString = '2024-10-16T23:59:59.999Z';

      // Naive approach (actually works fine for ISO strings)
      const naiveDate = new Date(dateString);

      // Our approach that explicitly handles date components
      const fixedDate = parseUTCDateForPicker(dateString);

      // Different dates because of timezone
      expect(naiveDate.getDate()).toBe(17);
      expect(fixedDate.getDate()).toBe(16);

      // Verify the fixed date has correct components
      expect(fixedDate.getUTCFullYear()).toBe(2024);
      expect(fixedDate.getUTCMonth()).toBe(9); // October
      expect(fixedDate.getUTCDate()).toBe(16);
    });
  });
});

describe('formatDateRange timezone stability', () => {
  beforeEach(() => {
    vi.stubEnv('TZ', 'America/Los_Angeles');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('names the UTC day for a boundary instant, not the local one', () => {
    const result = formatDateRange('2024-01-01T00:00:00.000Z');

    expect(result).toContain('2024');
    expect(result).not.toContain('2023');
  });

  it('names the UTC day at both ends of a range', () => {
    const result = formatDateRange('2024-01-01T00:00:00.000Z', '2024-01-31T23:59:59.999Z');

    expect(result).not.toContain('2023');
    expect(result.split('—')).toHaveLength(2);
    for (const half of result.split('—')) {
      expect(half).toContain('2024');
    }
  });
});
