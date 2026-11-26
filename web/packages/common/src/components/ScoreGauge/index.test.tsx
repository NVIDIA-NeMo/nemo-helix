// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  SCORE_TIER_COLORS,
  ScoreGauge,
  scoreColor,
  scoreTier,
} from '@nemo/common/src/components/ScoreGauge/index';
import { render, screen } from '@testing-library/react';

describe('scoreTier', () => {
  it.each([
    [9, 'Excellent'],
    [8, 'Excellent'],
    [7.9, 'Very Good'],
    [6, 'Very Good'],
    [5, 'Good'],
    [4, 'Good'],
    [3, 'Moderate'],
    [2, 'Moderate'],
    [1.5, 'Poor'],
  ])('maps %s to %s', (score, tier) => {
    expect(scoreTier(score)).toBe(tier);
  });

  it('treats 0, negative, and non-finite as Unavailable', () => {
    expect(scoreTier(0)).toBe('Unavailable');
    expect(scoreTier(-1)).toBe('Unavailable');
    expect(scoreTier(Number.NaN)).toBe('Unavailable');
  });
});

describe('scoreColor', () => {
  it('returns the tier color for a scored value', () => {
    expect(scoreColor(8.2)).toBe(SCORE_TIER_COLORS.Excellent);
    expect(scoreColor(2.5)).toBe(SCORE_TIER_COLORS.Moderate);
  });

  it('returns a neutral color when unavailable', () => {
    expect(scoreColor(0)).toBe('#888888');
  });
});

describe('ScoreGauge', () => {
  it('renders a gradient ring and marker for an available large gauge', () => {
    render(<ScoreGauge score={7.7} size="lg" />);
    expect(screen.getAllByTestId('gauge-gradient-segment').length).toBeGreaterThan(0);
    expect(screen.getByTestId('gauge-marker')).toBeInTheDocument();
    expect(screen.getByTestId('gauge-display')).toHaveTextContent('7.7');
  });

  it('renders a single-tier progress arc for a small gauge', () => {
    render(<ScoreGauge score={5} size="sm" />);
    expect(screen.getByTestId('gauge-progress')).toHaveAttribute('stroke', SCORE_TIER_COLORS.Good);
    expect(screen.queryByTestId('gauge-marker')).not.toBeInTheDocument();
  });

  it('renders an unavailable state without a marker or progress', () => {
    render(<ScoreGauge score={0} size="lg" />);
    expect(screen.getByTestId('gauge-display')).toHaveTextContent('—');
    expect(screen.queryByTestId('gauge-gradient-segment')).not.toBeInTheDocument();
    expect(screen.queryByTestId('gauge-marker')).not.toBeInTheDocument();
  });

  it.each([null, undefined])('renders a %s score as unavailable', (score) => {
    render(<ScoreGauge score={score} size="lg" />);
    expect(screen.getByRole('img')).toHaveAccessibleName('Score unavailable');
    expect(screen.getByTestId('gauge-display')).toHaveTextContent('—');
    expect(screen.queryByTestId('gauge-gradient-segment')).not.toBeInTheDocument();
    expect(screen.queryByTestId('gauge-marker')).not.toBeInTheDocument();
    expect(screen.queryByTestId('gauge-progress')).not.toBeInTheDocument();
  });

  it('clamps an out-of-range score to 10', () => {
    render(<ScoreGauge score={15} size="lg" />);
    expect(screen.getByTestId('gauge-display')).toHaveTextContent('10.0');
  });
});
