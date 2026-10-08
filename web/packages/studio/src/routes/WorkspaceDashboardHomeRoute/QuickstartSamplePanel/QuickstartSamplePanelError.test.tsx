// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QuickstartSamplePanelError } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/QuickstartSamplePanelError';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

/** A retry the test settles by hand, so the in-flight state can be observed. */
const pendingRetry = () => {
  let settle: () => void = () => {};
  const onRetry = vi.fn(
    () =>
      new Promise<void>((resolve) => {
        settle = resolve;
      })
  );
  return { onRetry, settle: () => settle() };
};

describe('QuickstartSamplePanelError', () => {
  it('says what failed and offers a retry', () => {
    render(<QuickstartSamplePanelError onRetry={vi.fn().mockResolvedValue(undefined)} />);

    expect(screen.getByText("Couldn't load the sample agent")).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled();
  });

  it('retries when the button is clicked', async () => {
    const onRetry = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<QuickstartSamplePanelError onRetry={onRetry} />);

    await user.click(screen.getByRole('button', { name: 'Retry' }));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('marks the button busy until the retry settles, so it cannot be sent twice', async () => {
    const { onRetry, settle } = pendingRetry();
    const user = userEvent.setup();
    render(<QuickstartSamplePanelError onRetry={onRetry} />);

    await user.click(screen.getByRole('button', { name: 'Retry' }));

    const button = screen.getByRole('button', { name: 'Retry' });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('aria-busy', 'true');

    settle();

    await waitFor(() => expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled());
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
