// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MeanValueTooltipCell } from '@studio/components/dataViews/ExperimentDataView/MeanValueTooltipCell';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const renderCell = (failedCount?: number) =>
  render(
    <TestProviders>
      <MeanValueTooltipCell
        label="Accuracy"
        count={4}
        runCount={4}
        countsMissingAsZero
        failedCount={failedCount}
      >
        0.25
      </MeanValueTooltipCell>
    </TestProviders>
  );

describe('MeanValueTooltipCell', () => {
  it('says how many runs failed to score', async () => {
    renderCell(3);

    await userEvent.hover(screen.getByText('0.25'));

    expect(
      (
        await screen.findAllByText(
          'Mean Accuracy over 4 test cases. 3 runs failed to score and count as 0.'
        )
      ).length
    ).toBeGreaterThan(0);
  });

  it('keeps the generic note when nothing failed', async () => {
    renderCell(0);

    await userEvent.hover(screen.getByText('0.25'));

    expect(
      (
        await screen.findAllByText(
          'Mean Accuracy over 4 test cases. Runs with no score count as 0.'
        )
      ).length
    ).toBeGreaterThan(0);
  });
});
