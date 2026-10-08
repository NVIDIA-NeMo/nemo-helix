// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QuickstartSamplePanelSkeleton } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/QuickstartSamplePanelSkeleton';
import { render, screen } from '@testing-library/react';

describe('QuickstartSamplePanelSkeleton', () => {
  it('marks itself busy for assistive tech', () => {
    render(<QuickstartSamplePanelSkeleton intakeEnabled agentOptimizationsEnabled />);

    expect(screen.getByTestId('quickstart-sample-panel-skeleton')).toHaveAttribute(
      'aria-busy',
      'true'
    );
  });

  it('reserves a row for each of the four steps', () => {
    render(<QuickstartSamplePanelSkeleton intakeEnabled agentOptimizationsEnabled />);

    expect(screen.getAllByTestId('quickstart-sample-step-skeleton')).toHaveLength(4);
  });

  it('reserves only the steps the panel will render once features are gated out', () => {
    // Matches the panel, which drops the trace and optimize steps under these flags.
    render(
      <QuickstartSamplePanelSkeleton intakeEnabled={false} agentOptimizationsEnabled={false} />
    );

    expect(screen.getAllByTestId('quickstart-sample-step-skeleton')).toHaveLength(2);
  });
});
