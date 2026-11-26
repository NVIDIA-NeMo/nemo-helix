// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ARC_START,
  ARC_SWEEP,
  CENTER,
  GRADIENT_PALETTE,
  GRADIENT_SEGMENTS,
  RADIUS,
  SCORE_TIER_COLORS,
  type ScoreTier,
  UNAVAILABLE_COLOR,
} from '@nemo/common/src/components/ScoreGauge/consts';

export function scoreTier(score: number): ScoreTier {
  if (!Number.isFinite(score) || score < 0) return 'Unavailable';
  if (score >= 8) return 'Excellent';
  if (score >= 6) return 'Very Good';
  if (score >= 4) return 'Good';
  if (score >= 2) return 'Moderate';
  return 'Poor';
}

export function scoreColor(score: number): string {
  const tier = scoreTier(score);
  return tier === 'Unavailable' ? UNAVAILABLE_COLOR : SCORE_TIER_COLORS[tier];
}

export function point(angleDeg: number): { x: number; y: number } {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: CENTER + RADIUS * Math.cos(rad), y: CENTER + RADIUS * Math.sin(rad) };
}

export function arcPath(fromPct: number, toPct: number): string {
  const a0 = ARC_START + (ARC_SWEEP * fromPct) / 100;
  const a1 = ARC_START + (ARC_SWEEP * toPct) / 100;
  const p0 = point(a0);
  const p1 = point(a1);
  const largeArc = a1 - a0 > 180 ? 1 : 0;
  return `M ${p0.x} ${p0.y} A ${RADIUS} ${RADIUS} 0 ${largeArc} 1 ${p1.x} ${p1.y}`;
}

export function interpolate(t: number): string {
  const position = t * (GRADIENT_PALETTE.length - 1);
  const low = Math.floor(position);
  const high = Math.min(low + 1, GRADIENT_PALETTE.length - 1);
  const ratio = position - low;
  const [r, g, b] = GRADIENT_PALETTE[low].map((value, index) =>
    Math.round(value + (GRADIENT_PALETTE[high][index] - value) * ratio)
  );
  return `rgb(${r}, ${g}, ${b})`;
}

export const gradientSegments = Array.from({ length: GRADIENT_SEGMENTS }, (_, index) => ({
  d: arcPath((index / GRADIENT_SEGMENTS) * 100, ((index + 1) / GRADIENT_SEGMENTS) * 100),
  stroke: interpolate(index / (GRADIENT_SEGMENTS - 1)),
}));
