// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Star } from '@nvidia/react-gui-icons/icons';
import { EvalCard } from '@studio/components/evaluation/EvalCard';
import { render, screen } from '@testing-library/react';

describe('EvalCard', () => {
  it('renders the metric name', () => {
    render(<EvalCard name="my-metric" />);
    expect(screen.getByText('my-metric')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    render(<EvalCard name="my-metric" description="Evaluates response quality." />);
    expect(screen.getByText('Evaluates response quality.')).toBeInTheDocument();
  });

  it('renders a badge with the mapped label for llm-judge type', () => {
    render(<EvalCard name="my-metric" type="llm-judge" />);
    expect(screen.getByText('LLM Judge')).toBeInTheDocument();
  });

  it('renders a badge with title-cased words for unmapped types', () => {
    render(<EvalCard name="my-metric" type="custom-metric" />);
    expect(screen.getByText('Custom Metric')).toBeInTheDocument();
  });

  it('does not render a badge when type is null', () => {
    render(<EvalCard name="my-metric" type={null} />);
    expect(screen.queryByTestId('nv-badge')).not.toBeInTheDocument();
  });

  it('does not render a badge when type is undefined', () => {
    render(<EvalCard name="my-metric" />);
    expect(screen.queryByTestId('nv-badge')).not.toBeInTheDocument();
  });

  it('renders a custom icon when provided', () => {
    render(<EvalCard name="my-metric" icon={<Star data-testid="custom-icon" size="12" />} />);
    expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
  });
});
