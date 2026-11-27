// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  gradeForScore,
  meetsGrade,
  PRIVACY_RECOMMENDED_USES,
  QUALITY_RECOMMENDED_USES,
} from '@studio/routes/SafeSynthesizerJobReportRoute/util';

describe('gradeForScore', () => {
  it.each([
    [10, 'Excellent'],
    [8, 'Excellent'],
    [7.9, 'Very Good'],
    [6, 'Very Good'],
    [5.9, 'Good'],
    [4, 'Good'],
    [3.9, 'Moderate'],
    [2, 'Moderate'],
    [1.9, 'Poor'],
  ])('uses the library scale for quality and privacy: %s is %s', (score, grade) => {
    expect(gradeForScore(score)).toBe(grade);
  });

  it.each([undefined, null, -1, Number.NaN])('treats %s as Unavailable', (score) => {
    expect(gradeForScore(score)).toBe('Unavailable');
  });

  it('grades 0 as Poor', () => {
    expect(gradeForScore(0)).toBe('Poor');
  });
});

describe('meetsGrade', () => {
  it('passes at or above the minimum grade', () => {
    expect(meetsGrade('Good', 'Good')).toBe(true);
    expect(meetsGrade('Excellent', 'Very Good')).toBe(true);
  });

  it('fails below the minimum grade or when unavailable', () => {
    expect(meetsGrade('Moderate', 'Good')).toBe(false);
    expect(meetsGrade('Unavailable', 'Moderate')).toBe(false);
  });
});

describe('recommended uses', () => {
  const suitableAt = (uses: typeof QUALITY_RECOMMENDED_USES, score: number) =>
    uses
      .filter((use) => meetsGrade(gradeForScore(score), use.minimumGrade))
      .map((use) => use.label);

  it('matches the library quality guidance', () => {
    expect(suitableAt(QUALITY_RECOMMENDED_USES, 3)).toEqual([]);
    expect(suitableAt(QUALITY_RECOMMENDED_USES, 5)).toEqual([
      'Demo environments or mock data',
      'Pre-production testing environments',
    ]);
    expect(suitableAt(QUALITY_RECOMMENDED_USES, 6)).toHaveLength(4);
  });

  it('matches the library privacy guidance', () => {
    expect(suitableAt(PRIVACY_RECOMMENDED_USES, 1)).toEqual([]);
    expect(suitableAt(PRIVACY_RECOMMENDED_USES, 2)).toEqual(['Internally, within the same team']);
    expect(suitableAt(PRIVACY_RECOMMENDED_USES, 7.9)).toHaveLength(3);
    expect(suitableAt(PRIVACY_RECOMMENDED_USES, 8)).toHaveLength(4);
  });
});
