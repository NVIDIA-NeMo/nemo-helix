// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@nvidia/foundations-react-core';
import { ScoreItem } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreItem';
import { render, screen } from '@testing-library/react';

const renderScoreItem = (success: boolean, value: string) => {
  return render(
    <ThemeProvider>
      <ScoreItem success={success} value={value} />
    </ThemeProvider>
  );
};

describe('ScoreItem', () => {
  it('renders CheckCircle icon when success is true', () => {
    renderScoreItem(true, 'Test passed');

    // eslint-disable-next-line testing-library/no-node-access
    expect(document.querySelector('svg:not([id*="sprite"]).text-brand')).toBeInTheDocument();
  });

  it('renders Cancel icon when success is false', () => {
    renderScoreItem(false, 'Test failed');

    // eslint-disable-next-line testing-library/no-node-access
    expect(document.querySelector('svg:not([id*="sprite"]).text-brand')).not.toBeInTheDocument();
  });

  it('displays the correct value text', () => {
    const testValue = 'Sample test value';
    renderScoreItem(true, testValue);

    expect(screen.getByText(testValue)).toBeInTheDocument();
  });

  it('applies text-brand className to CheckCircle when success is true', () => {
    renderScoreItem(true, 'Success message');

    // eslint-disable-next-line testing-library/no-node-access
    expect(document.querySelector('svg:not([id*="sprite"])')).toHaveClass('text-brand');
  });
});
