// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile } from '@nemo/common/src/components/StatTile';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Bot } from 'lucide-react';
import { MemoryRouter } from 'react-router';

describe('StatTile', () => {
  it('renders the label and value', () => {
    render(<StatTile label="Final training loss" value="0.1234" />);

    expect(screen.getByText('Final training loss')).toBeInTheDocument();
    expect(screen.getByText('0.1234')).toBeInTheDocument();
  });

  it('renders the trailing label only when provided', () => {
    const { rerender } = render(
      <StatTile label="gen_kl_error" value="5.4e-4" trailingLabel="ok" />
    );
    expect(screen.getByText('ok')).toBeInTheDocument();

    rerender(<StatTile label="gen_kl_error" value="5.4e-4" />);
    expect(screen.queryByText('ok')).not.toBeInTheDocument();
  });

  it('statuses the trailing label independently of the hint', () => {
    render(
      <StatTile
        label="approx_entropy"
        value="0.31"
        trailingLabel="falling"
        trailingLabelStatus="warning"
        hint="entropy collapse risk"
      />
    );

    expect(screen.getByText('falling')).toHaveClass(
      'text-[color:var(--text-color-feedback-warning)]'
    );
    expect(screen.getByText('entropy collapse risk')).toHaveClass('text-placeholder');
  });

  it('uses the metric variant styling for the label and trailing unit', () => {
    render(<StatTile label="Sessions" value="0" trailingLabel="avg / week" variant="metric" />);

    expect(screen.getByText('Sessions')).toHaveClass('text-secondary');
    expect(screen.getByText('avg / week')).toHaveClass('text-secondary');
    expect(screen.getByTestId('stat-tile-surface')).toHaveClass('w-full');
  });

  it('still statuses the trailing label in the metric variant', () => {
    render(
      <StatTile
        label="Cost"
        value="$1.20"
        trailingLabel="up 12%"
        trailingLabelStatus="warning"
        variant="metric"
      />
    );

    expect(screen.getByText('up 12%')).toHaveClass(
      'text-[color:var(--text-color-feedback-warning)]'
    );
  });

  it('renders the hint only when provided', () => {
    const { rerender } = render(
      <StatTile label="Final training loss" value="0.1234" hint="-0.05 from start" />
    );
    expect(screen.getByText('-0.05 from start')).toBeInTheDocument();

    rerender(<StatTile label="Final training loss" value="0.1234" />);
    expect(screen.queryByText('-0.05 from start')).not.toBeInTheDocument();
  });

  it('applies the success hint status class when specified', () => {
    render(
      <StatTile label="Final training loss" value="0.1234" hint="-0.05" hintStatus="success" />
    );

    expect(screen.getByText('-0.05')).toHaveClass(
      'text-[color:var(--text-color-feedback-success)]'
    );
  });

  it('applies the warning hint status class when specified', () => {
    render(
      <StatTile label="Truncation Rate" value="4.1%" hint="hit length limit" hintStatus="warning" />
    );

    expect(screen.getByText('hit length limit')).toHaveClass(
      'text-[color:var(--text-color-feedback-warning)]'
    );
  });

  it('applies the error hint status class when specified', () => {
    render(
      <StatTile label="Final validation loss" value="0.9876" hint="+0.05" hintStatus="error" />
    );

    expect(screen.getByText('+0.05')).toHaveClass('text-[color:var(--text-color-feedback-danger)]');
  });

  it('drops its surrounding surface when not bordered', () => {
    const { rerender } = render(<StatTile label="Duration" value="00:01:38" />);
    expect(screen.getByTestId('stat-tile-surface')).toBeInTheDocument();

    rerender(<StatTile label="Duration" value="00:01:38" bordered={false} />);
    expect(screen.queryByTestId('stat-tile-surface')).not.toBeInTheDocument();
    expect(screen.getByText('Duration')).toBeInTheDocument();
    expect(screen.getByText('00:01:38')).toBeInTheDocument();
  });

  it('keeps the default border for a healthy tile', () => {
    render(<StatTile label="gen_kl_error" value="5.4e-4" status="success" />);

    const surface = screen.getByTestId('stat-tile-surface');
    expect(surface).not.toHaveClass('border-(--border-color-feedback-warning)');
    expect(surface).not.toHaveClass('border-(--border-color-feedback-danger)');
  });

  it('tints the border amber on warning status', () => {
    render(<StatTile label="approx_entropy" value="0.31" status="warning" />);

    expect(screen.getByTestId('stat-tile-surface')).toHaveClass(
      'border-(--border-color-feedback-warning)'
    );
  });

  it('tints the border red on error status', () => {
    render(<StatTile label="token_mult_prob_error" value="1.03" status="error" />);

    expect(screen.getByTestId('stat-tile-surface')).toHaveClass(
      'border-(--border-color-feedback-danger)'
    );
  });

  it('statuses the border independently of the trailing label and hint', () => {
    render(
      <StatTile
        label="approx_entropy"
        value="0.31"
        trailingLabel="falling"
        trailingLabelStatus="warning"
        status="warning"
        hint="entropy collapse risk"
      />
    );

    // The threshold hint stays static grey even while the tile itself is flagged.
    expect(screen.getByText('entropy collapse risk')).toHaveClass('text-placeholder');
  });

  it('renders the label in the same muted tone as an unstatused hint', () => {
    render(<StatTile label="Learning Rate" value="1.00e-6" hint="at latest step" />);

    expect(screen.getByText('Learning Rate')).toHaveClass('text-placeholder');
    expect(screen.getByText('at latest step')).toHaveClass('text-placeholder');
  });

  it('renders the actionable variant as a link with its icon, label, and value', () => {
    render(
      <MemoryRouter>
        <StatTile
          label="Agents"
          value="1"
          icon={<Bot data-testid="agents-icon" />}
          to="/agents"
          variant="actionable"
        />
      </MemoryRouter>
    );

    const link = screen.getByRole('link', { name: /agents/i });
    expect(link).toHaveAttribute('href', '/agents');
    expect(screen.getByTestId('agents-icon')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  it('renders the actionable variant as a button and fires onClick when there is no destination', async () => {
    const onClick = vi.fn();
    render(<StatTile label="Insights" value="4" onClick={onClick} variant="actionable" />);

    const button = screen.getByRole('button', { name: /insights/i });
    await userEvent.click(button);

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('still fires onClick for the actionable variant when a destination is also set', async () => {
    const onClick = vi.fn();
    render(
      <MemoryRouter>
        <StatTile label="Agents" value="1" to="/agents" onClick={onClick} variant="actionable" />
      </MemoryRouter>
    );

    await userEvent.click(screen.getByRole('link', { name: /agents/i }));

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('tints the actionable variant border on status', () => {
    render(
      <MemoryRouter>
        <StatTile label="Agents" value="1" to="/agents" status="error" variant="actionable" />
      </MemoryRouter>
    );

    expect(screen.getByTestId('stat-tile-surface')).toHaveClass(
      'border-(--border-color-feedback-danger)'
    );
  });
});
