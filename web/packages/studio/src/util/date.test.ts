// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatDateRange } from '@nemo/common/src/utils/formatDateRange';
import { formatDateTime, formatElapsedTime } from '@studio/util/date';

describe('formatDateTime', () => {
  describe('Given the date as a string', () => {
    it('Returns the correctly-formatted date string', () => {
      expect(formatDateTime('2023-10-06T20:14:00.690Z')).toBe('10/06/23 08:14:00 PM');
      expect(formatDateTime('2023-10-06T20:14:00.690Z', false)).toBe('08:14:00 PM');
      expect(formatDateTime('2021-01-31T02:30:00.690Z')).toBe('01/31/21 02:30:00 AM');
      expect(formatDateTime('2021-01-31T02:30:00.690Z', false)).toBe('02:30:00 AM');
    });
  });

  describe('Given the date as a number', () => {
    it('Returns the correctly-formatted date string', () => {
      expect(formatDateTime(1696623240690)).toBe('10/06/23 08:14:00 PM');
      expect(formatDateTime(1696623240690, false)).toBe('08:14:00 PM');
      expect(formatDateTime(1612060200690)).toBe('01/31/21 02:30:00 AM');
      expect(formatDateTime(1612060200690, false)).toBe('02:30:00 AM');
    });
  });
});

describe('formatElapsedTime', () => {
  it('Returns formatted elapsed time', () => {
    expect(
      formatElapsedTime(new Date('2024-01-10T20:00:00.000Z'), new Date('2024-01-10T20:00:00.000Z'))
    ).toEqual('00:00:00');

    expect(
      formatElapsedTime(new Date('2024-01-10T20:00:00.000Z'), new Date('2024-01-10T20:00:30.000Z'))
    ).toEqual('00:00:30');

    expect(
      formatElapsedTime(new Date('2024-01-10T20:10:00.000Z'), new Date('2024-01-10T20:00:00.000Z'))
    ).toEqual('00:10:00');

    expect(
      formatElapsedTime(new Date('2024-01-10T20:00:00.000Z'), new Date('2024-01-11T22:10:20.000Z'))
    ).toEqual('26:10:20');
  });
});

const day = (year: number, month: number, date: number) =>
  new Date(Date.UTC(year, month - 1, date)).toLocaleDateString(undefined, { timeZone: 'UTC' });

describe('formatDateRange', () => {
  describe('Given both start and end dates', () => {
    it('Returns the date range with em dash separator', () => {
      const expected = `${day(2023, 10, 6)} — ${day(2023, 10, 10)}`;
      expect(formatDateRange('2023-10-06T20:14:00.690Z', '2023-10-10T15:30:00.000Z')).toBe(
        expected
      );
      expect(formatDateRange(1696623240690, 1696958400000)).toBe(expected);
    });
  });

  describe('Given only start date', () => {
    it('Returns only the start date formatted', () => {
      expect(formatDateRange('2023-10-06T20:14:00.690Z')).toBe(day(2023, 10, 6));
      expect(formatDateRange(1696623240690)).toBe(day(2023, 10, 6));
    });
  });

  describe('Given only end date', () => {
    it('Returns only the end date formatted', () => {
      expect(formatDateRange(undefined, '2023-10-10T15:30:00.000Z')).toBe(day(2023, 10, 10));
      expect(formatDateRange(undefined, 1696958400000)).toBe(day(2023, 10, 10));
    });
  });

  describe('Given no dates', () => {
    it('Returns empty string', () => {
      expect(formatDateRange()).toBe('');
      expect(formatDateRange(undefined, undefined)).toBe('');
    });
  });
});
