// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  MetricTrendPanel,
  type MetricTrendSeries,
} from '@studio/components/charts/MetricTrendPanel';
import { fireEvent, render, screen } from '@studio/tests/util/render';

const series: MetricTrendSeries[] = [
  {
    id: 'solved',
    label: 'Solved',
    value: 78.4,
    delta: 3.3,
    points: [
      { label: 'Day 1', value: 70 },
      { label: 'Day 2', value: 78.4 },
    ],
  },
  {
    id: 'tool-use',
    label: 'Tool Use',
    value: 69.2,
    delta: -2.4,
    points: [
      { label: 'Day 1', value: 71 },
      { label: 'Day 2', value: 69.2 },
    ],
  },
];

describe('MetricTrendPanel', () => {
  it('renders the first series value and delta by default', () => {
    render(
      <MetricTrendPanel
        title="Primary use cases"
        series={series}
        comparisonLabel="vs. 7 days ago"
      />
    );

    expect(screen.getByText('78.4%')).toBeInTheDocument();
    expect(screen.getByText('+3.3')).toBeInTheDocument();
    expect(screen.getByText('vs. 7 days ago')).toBeInTheDocument();
  });

  it('switches series when a pill is clicked', () => {
    render(<MetricTrendPanel title="Primary use cases" series={series} />);

    fireEvent.click(screen.getByRole('button', { name: 'Tool Use' }));

    expect(screen.getByText('69.2%')).toBeInTheDocument();
    expect(screen.getByText('−2.4')).toBeInTheDocument();
  });

  it('stays on the controlled series and reports the change', () => {
    const onSeriesChange = vi.fn();
    render(
      <MetricTrendPanel
        title="Primary use cases"
        series={series}
        selectedSeriesId="solved"
        onSeriesChange={onSeriesChange}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: 'Tool Use' }));

    expect(onSeriesChange).toHaveBeenCalledWith('tool-use');
    expect(screen.getByText('78.4%')).toBeInTheDocument();
  });

  it('calls onViewClick from the header action', () => {
    const onViewClick = vi.fn();
    render(
      <MetricTrendPanel title="Primary use cases" series={series} onViewClick={onViewClick} />
    );

    fireEvent.click(screen.getByRole('button', { name: 'View' }));

    expect(onViewClick).toHaveBeenCalledTimes(1);
  });

  it('renders a neutral delta with no directional icon when delta is zero', () => {
    render(<MetricTrendPanel title="Primary use cases" series={[{ ...series[0], delta: 0 }]} />);

    const deltaTag = screen.getByTestId('nv-tag-root');
    expect(deltaTag).toHaveTextContent('0.0');
    expect(deltaTag.className).toContain('nv-tag--color-gray');
    expect(deltaTag.querySelector('svg')).not.toBeInTheDocument();
  });

  it('renders a neutral delta when a nonzero delta rounds to zero at the display precision', () => {
    render(<MetricTrendPanel title="Primary use cases" series={[{ ...series[0], delta: 0.04 }]} />);

    const deltaTag = screen.getByTestId('nv-tag-root');
    expect(deltaTag).toHaveTextContent('+0.0');
    expect(deltaTag.className).toContain('nv-tag--color-gray');
    expect(deltaTag.querySelector('svg')).not.toBeInTheDocument();
  });

  it('renders a neutral delta when a negative delta rounds to zero at the display precision', () => {
    render(
      <MetricTrendPanel title="Primary use cases" series={[{ ...series[0], delta: -0.04 }]} />
    );

    const deltaTag = screen.getByTestId('nv-tag-root');
    expect(deltaTag).toHaveTextContent('−0.0');
    expect(deltaTag.className).toContain('nv-tag--color-gray');
    expect(deltaTag.querySelector('svg')).not.toBeInTheDocument();
  });

  it('does not show the loading skeleton for an empty series once pending finishes', () => {
    render(<MetricTrendPanel title="Primary use cases" series={[]} isPending={false} />);

    expect(screen.queryByTestId('nv-skeleton')).not.toBeInTheDocument();
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });
});
