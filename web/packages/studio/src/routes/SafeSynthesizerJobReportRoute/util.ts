// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SCORE_TIER_COLORS, type ScoreTier } from '@nemo/common/src/components/ScoreGauge/consts';
import { scoreTier } from '@nemo/common/src/components/ScoreGauge/utils';

const TIERS_HIGHEST_FIRST = Object.keys(SCORE_TIER_COLORS) as ScoreTier[];

export const GRADE_ORDER: readonly ScoreTier[] = [
  'Unavailable',
  ...[...TIERS_HIGHEST_FIRST].reverse(),
];

export function gradeForScore(score: number | null | undefined): ScoreTier {
  return scoreTier(score ?? Number.NaN);
}

export function meetsGrade(grade: ScoreTier, minimumGrade: ScoreTier): boolean {
  return GRADE_ORDER.indexOf(grade) >= GRADE_ORDER.indexOf(minimumGrade);
}

export interface RecommendedUse {
  readonly label: string;
  readonly minimumGrade: ScoreTier;
}

export const QUALITY_RECOMMENDED_USES: readonly RecommendedUse[] = [
  { label: 'Demo environments or mock data', minimumGrade: 'Good' },
  { label: 'Pre-production testing environments', minimumGrade: 'Good' },
  { label: 'Balance or augment machine learning data', minimumGrade: 'Very Good' },
  { label: 'Machine learning or statistical analysis', minimumGrade: 'Very Good' },
];

export const PRIVACY_RECOMMENDED_USES: readonly RecommendedUse[] = [
  { label: 'Internally, within the same team', minimumGrade: 'Moderate' },
  { label: 'Internally, across different teams', minimumGrade: 'Good' },
  { label: 'Externally, with trusted partners', minimumGrade: 'Very Good' },
  { label: 'Externally, with public availability', minimumGrade: 'Excellent' },
];
