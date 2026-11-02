// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CancelJobModal } from '@nemo/common/src/components/CancelJobModal';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent, { UserEvent } from '@testing-library/user-event';

const JOB_NAME = 'test-job-abc123';
const BODY_COPY =
  "Cancelling stops this job and can't be undone. A cancelled job can't be resumed or rerun, but it can still be deleted. It may show as Cancelling briefly before it stops.";

const setup = (onConfirm: () => Promise<void>) => {
  const onClose = vi.fn();
  const user = userEvent.setup();
  render(
    <TestProviders>
      <CancelJobModal open jobName={JOB_NAME} onClose={onClose} onConfirm={onConfirm} />
    </TestProviders>
  );
  return { onClose, user };
};

const submit = () => screen.getByRole('button', { name: 'Cancel Job' });
const dismiss = () => screen.getByRole('button', { name: 'Cancel' });

describe('CancelJobModal', () => {
  it('names the job and shows the new copy', () => {
    setup(vi.fn().mockResolvedValue(undefined));
    expect(screen.getByText(`Cancel ${JOB_NAME}`)).toBeInTheDocument();
    expect(screen.getByText(BODY_COPY)).toBeInTheDocument();
  });

  it.each([
    ['dismiss button', (user: UserEvent) => user.click(dismiss())],
    ['Escape', (user: UserEvent) => user.keyboard('{Escape}')],
  ])('dismissing via %s calls onClose once and never onConfirm', async (_, act) => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    const { onClose, user } = setup(onConfirm);

    await act(user);

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it('calls onConfirm once on confirm and onClose once on success', async () => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    const { onClose, user } = setup(onConfirm);

    await user.click(submit());

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it('blocks double submit and disables controls while pending', async () => {
    const onConfirm = vi.fn(() => new Promise<void>(() => {}));
    const { onClose, user } = setup(onConfirm);

    await user.click(submit());
    await user.click(submit());

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(submit()).toBeDisabled();
    expect(dismiss()).toBeDisabled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('keeps the modal open with an inline error on failure and allows retry', async () => {
    const onConfirm = vi
      .fn()
      .mockRejectedValueOnce(new Error('Job is already finishing'))
      .mockResolvedValueOnce(undefined);
    const { onClose, user } = setup(onConfirm);

    await user.click(submit());

    expect(await screen.findByText('Job is already finishing')).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    await user.click(submit());

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(onConfirm).toHaveBeenCalledTimes(2);
  });
});
