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

    expect(screen.getByTestId('score-check-icon')).toBeInTheDocument();
  });

  it('renders Cancel icon when success is false', () => {
    renderScoreItem(false, 'Test failed');

    expect(screen.getByTestId('score-cancel-icon')).toBeInTheDocument();
  });

  it('displays the correct value text', () => {
    const testValue = 'Sample test value';
    renderScoreItem(true, testValue);

    expect(screen.getByText(testValue)).toBeInTheDocument();
  });

  it('applies text-brand className to CheckCircle when success is true', () => {
    renderScoreItem(true, 'Success message');

    expect(screen.getByTestId('score-check-icon')).toHaveClass('text-brand');
  });
});
