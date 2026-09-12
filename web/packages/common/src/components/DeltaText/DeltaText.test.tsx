// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeltaText, deltaTone, formatSignedDelta } from '@nemo/common/src/components/DeltaText';
import { render, screen } from '@testing-library/react';

describe('formatSignedDelta', () => {
  it('signs the value and uses a true minus', () => {
    expect(formatSignedDelta(0.07)).toBe('+0.07');
    expect(formatSignedDelta(-0.07)).toBe('−0.07');
    expect(formatSignedDelta(0)).toBe('0.00');
    expect(formatSignedDelta(1.25, 1)).toBe('+1.3');
  });
});

describe('deltaTone', () => {
  it('flips with the metric direction', () => {
    expect(deltaTone(1)).toBe('improved');
    expect(deltaTone(-1)).toBe('regressed');
    expect(deltaTone(1, false)).toBe('regressed');
    expect(deltaTone(-1, false)).toBe('improved');
    expect(deltaTone(0, false)).toBe('unchanged');
  });
});

describe('DeltaText', () => {
  it('renders a rise as an improvement', () => {
    render(<DeltaText value={0.07} />);

    const delta = screen.getByTestId('delta-text');
    expect(delta).toHaveAttribute('data-delta', 'improved');
    expect(delta).toHaveTextContent('+0.07');
    expect(delta).toHaveAccessibleName('Improved by 0.07');
  });

  it('tints a fall as a regression and points the triangle down', () => {
    render(<DeltaText value={-0.07} />);

    expect(screen.getByTestId('delta-text')).toHaveAttribute('data-delta', 'regressed');
    expect(screen.getByTestId('delta-text-icon')).toHaveClass('rotate-180');
  });

  it('reads a fall as an improvement when lower is better', () => {
    render(<DeltaText value={-0.07} higherIsBetter={false} />);

    expect(screen.getByTestId('delta-text')).toHaveAttribute('data-delta', 'improved');
  });

  it('keeps an equals glyph in the gutter when nothing moved', () => {
    render(<DeltaText value={0} />);

    expect(screen.getByTestId('delta-text')).toHaveAttribute('data-delta', 'unchanged');
    expect(screen.getByTestId('delta-text')).toHaveAccessibleName('No change');

    // Same gutter as a moved row, so a column of deltas stays aligned.
    const icon = screen.getByTestId('delta-text-icon');
    expect(icon).toHaveAttribute('width', '0.72em');
    expect(icon).not.toHaveClass('fill-current');
  });

  it('sizes the glyph in em so it follows the text at every step', () => {
    const { rerender } = render(<DeltaText value={0.07} size="xs" />);
    expect(screen.getByTestId('delta-text-icon')).toHaveAttribute('width', '0.72em');

    rerender(<DeltaText value={0.07} size="xl" />);
    const icon = screen.getByTestId('delta-text-icon');
    expect(icon).toHaveAttribute('width', '0.72em');
    expect(icon).toHaveAttribute('height', '0.72em');
  });

  it('steps the type scale with `size`', () => {
    const { rerender } = render(<DeltaText value={0.07} />);
    expect(screen.getByTestId('delta-text')).toHaveClass('nv-text--body-semibold-xs');

    // `lg` reaches past the adjacent step onto 24px, so the top of the scale reads as large.
    rerender(<DeltaText value={0.07} size="lg" />);
    expect(screen.getByTestId('delta-text')).toHaveClass('nv-text--body-semibold-2xl');

    rerender(<DeltaText value={0.07} size="xl" />);
    expect(screen.getByTestId('delta-text')).toHaveClass('nv-text--body-semibold-3xl');
  });

  it('lets `kind` override the weight `size` picks', () => {
    render(<DeltaText value={0.07} size="lg" kind="label/regular/sm" />);

    expect(screen.getByTestId('delta-text')).toHaveClass('nv-text--label-regular-sm');
    expect(screen.getByTestId('delta-text')).not.toHaveClass('nv-text--body-semibold-2xl');
  });

  it('carries a trailing qualifier through `format`', () => {
    render(
      <DeltaText
        value={-18}
        higherIsBetter={false}
        format={(value) => `${formatSignedDelta(value, 0)}% vs baseline`}
      />
    );

    const delta = screen.getByTestId('delta-text');
    expect(delta).toHaveTextContent('−18% vs baseline');
    // Lower is better here, so a fall is the good news.
    expect(delta).toHaveAttribute('data-delta', 'improved');
    expect(delta).toHaveAccessibleName('Improved by 18% vs baseline');
  });

  it('honors a custom format', () => {
    render(<DeltaText value={-120} higherIsBetter={false} format={(v) => `${v} ms`} />);

    expect(screen.getByTestId('delta-text')).toHaveTextContent('-120 ms');
  });
});
