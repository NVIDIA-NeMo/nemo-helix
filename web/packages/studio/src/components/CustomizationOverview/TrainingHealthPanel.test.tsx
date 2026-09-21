// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TrainingHealthPanel } from '@studio/components/CustomizationOverview/TrainingHealthPanel';
import type { CustomizationStatusDetailsWithMetrics } from '@studio/types/customization';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const series = (...values: number[]) => values.map((value, step) => ({ step, value }));

/** The panel is collapsed by default and unmounts its charts until it is opened. */
const open = async () => {
  await userEvent.click(screen.getByText('Training health'));
};

/** What an unsloth run stores: the HF callback reports loss, lr and grad_norm per step. */
const unslothDetails = {
  metrics: {
    train_loss: series(2.1, 1.8, 1.5),
    train_lr: series(2e-4, 1.5e-4, 1e-4),
    train_grad_norm: series(1.2, 0.9, 0.7),
  },
} as unknown as CustomizationStatusDetailsWithMetrics;

describe('TrainingHealthPanel', () => {
  it('charts the series a run recorded besides its loss', async () => {
    render(<TrainingHealthPanel statusDetails={unslothDetails} />);
    await open();

    expect(screen.getByText('Learning rate')).toBeInTheDocument();
    expect(screen.getByText('Gradient norm')).toBeInTheDocument();
  });

  it('leaves the loss to the panel that already charts it', async () => {
    render(<TrainingHealthPanel statusDetails={unslothDetails} />);
    await open();

    expect(screen.queryByText('train_loss')).not.toBeInTheDocument();
  });

  it('drops a metric this backend did not report', async () => {
    // The list is shared with DPO, which is the only one reporting preference accuracy.
    render(<TrainingHealthPanel statusDetails={unslothDetails} />);
    await open();

    expect(screen.queryByText('Preference accuracy')).not.toBeInTheDocument();
  });

  it('renders nothing at all when only the loss was recorded', () => {
    const lossOnly = {
      metrics: { train_loss: series(2.1, 1.8) },
    } as unknown as CustomizationStatusDetailsWithMetrics;

    const { container } = render(<TrainingHealthPanel statusDetails={lossOnly} />);

    expect(container).toBeEmptyDOMElement();
  });
});
