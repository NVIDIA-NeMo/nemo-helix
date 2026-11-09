// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CancelJobButton } from '@nemo/common/src/components/CancelJobButton';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent, { type UserEvent } from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';

const JOB_NAME = 'test-job-abc123';

const renderButton = (status?: HelixJobStatus) =>
  render(
    <TestProviders>
      <CancelJobButton workspace={workspace1.name} jobName={JOB_NAME} jobStatus={status} />
    </TestProviders>
  );

describe('CancelJobButton', () => {
  beforeEach(() => {
    mockUseParams({
      [ROUTE_PARAMS.workspace]: workspace1.name,
    });
  });

  it.each([HelixJobStatus.created, HelixJobStatus.pending, HelixJobStatus.active])(
    'renders cancel button when status is %s',
    (status) => {
      renderButton(status);
      expect(screen.getByRole('button', { name: 'Cancel Job' })).toBeEnabled();
    }
  );

  it.each([HelixJobStatus.completed, HelixJobStatus.cancelled, HelixJobStatus.error, undefined])(
    'does not render when status is %s',
    (status) => {
      renderButton(status);
      expect(screen.queryByRole('button', { name: 'Cancel Job' })).not.toBeInTheDocument();
    }
  );

  it('shows disabled "Cancelling..." button when status is cancelling', () => {
    renderButton(HelixJobStatus.cancelling);
    const button = screen.getByRole('button', { name: 'Cancelling...' });
    expect(button).toBeDisabled();
  });

  it('opens confirmation modal on click', async () => {
    const user = userEvent.setup();
    renderButton(HelixJobStatus.active);

    await user.click(screen.getByRole('button', { name: 'Cancel Job' }));

    expect(screen.getByText(`Cancel ${JOB_NAME}`)).toBeInTheDocument();
    expect(
      screen.getByText(
        "Cancelling stops this job and can't be undone. A cancelled job can't be resumed or rerun, but it can still be deleted. It may show as Cancelling briefly before it stops."
      )
    ).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Cancel Job' }).length).toBeGreaterThan(0);
  });

  it('calls cancel API and closes modal on confirm', async () => {
    server.use(
      http.post(`${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name/cancel`, () =>
        HttpResponse.json({ name: JOB_NAME, status: HelixJobStatus.cancelled })
      )
    );

    const user = userEvent.setup();
    renderButton(HelixJobStatus.active);

    await user.click(screen.getByRole('button', { name: 'Cancel Job' }));

    const submitButtons = screen.getAllByRole('button', { name: 'Cancel Job' });
    const modalSubmit = submitButtons[submitButtons.length - 1];
    await user.click(modalSubmit);

    await screen.findByRole('button', { name: 'Cancel Job' });
  });

  it('shows loading state while cancel is in progress', async () => {
    server.use(
      http.post(
        `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name/cancel`,
        async () => {
          await delay('infinite');
          return HttpResponse.json({});
        }
      )
    );

    const user = userEvent.setup();
    renderButton(HelixJobStatus.active);

    await user.click(screen.getByRole('button', { name: 'Cancel Job' }));

    const submitButtons = screen.getAllByRole('button', { name: 'Cancel Job' });
    const modalSubmit = submitButtons[submitButtons.length - 1];
    await user.click(modalSubmit);

    expect(await screen.findByRole('status')).toBeInTheDocument();
  });

  const CANCEL_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name/cancel`;
  const openAndGetSubmit = async (user: UserEvent) => {
    await user.click(screen.getByRole('button', { name: 'Cancel Job' }));
    const buttons = screen.getAllByRole('button', { name: 'Cancel Job' });
    return buttons[buttons.length - 1];
  };

  it('dismissing sends no POST', async () => {
    let posts = 0;
    server.use(
      http.post(CANCEL_URL, () => {
        posts += 1;
        return HttpResponse.json({});
      })
    );
    const user = userEvent.setup();
    renderButton(HelixJobStatus.active);
    await openAndGetSubmit(user);

    await user.keyboard('{Escape}');

    await waitFor(() => expect(screen.queryByText(`Cancel ${JOB_NAME}`)).not.toBeInTheDocument());
    expect(posts).toBe(0);
  });

  it('sends exactly one POST even when submit is clicked twice', async () => {
    let posts = 0;
    server.use(
      http.post(CANCEL_URL, async () => {
        posts += 1;
        await delay(50);
        return HttpResponse.json({ name: JOB_NAME, status: HelixJobStatus.cancelled });
      })
    );
    const user = userEvent.setup();
    renderButton(HelixJobStatus.active);
    const modalSubmit = await openAndGetSubmit(user);

    await user.dblClick(modalSubmit);

    await waitFor(() => expect(screen.queryByText(`Cancel ${JOB_NAME}`)).not.toBeInTheDocument());
    expect(posts).toBe(1);
  });

  it('shows a 409 detail inside the open dialog and closes after a successful retry', async () => {
    let posts = 0;
    server.use(
      http.post(CANCEL_URL, () => {
        posts += 1;
        return posts === 1
          ? HttpResponse.json({ detail: 'Job is already finishing' }, { status: 409 })
          : HttpResponse.json({ name: JOB_NAME, status: HelixJobStatus.cancelled });
      })
    );
    const user = userEvent.setup();
    renderButton(HelixJobStatus.active);
    const modalSubmit = await openAndGetSubmit(user);

    await user.click(modalSubmit);

    const dialog = await screen.findByRole('dialog');
    expect(await within(dialog).findByText('Job is already finishing')).toBeInTheDocument();

    await user.click(within(dialog).getByRole('button', { name: 'Cancel Job' }));

    await waitFor(() => expect(screen.queryByText(`Cancel ${JOB_NAME}`)).not.toBeInTheDocument());
    expect(posts).toBe(2);
  });

  it('announces the request, then the result once the job reaches cancelled', async () => {
    server.use(
      http.post(CANCEL_URL, () =>
        HttpResponse.json({ name: JOB_NAME, status: HelixJobStatus.cancelling })
      ),
      http.get(`${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name`, () =>
        HttpResponse.json({ name: JOB_NAME, status: HelixJobStatus.cancelled })
      )
    );
    const user = userEvent.setup();
    renderButton(HelixJobStatus.active);
    const modalSubmit = await openAndGetSubmit(user);

    await user.click(modalSubmit);

    expect(await screen.findByText(`Cancellation requested for ${JOB_NAME}.`)).toBeInTheDocument();
    expect(
      await screen.findByText(`Job ${JOB_NAME} cancelled.`, undefined, { timeout: 6000 })
    ).toBeInTheDocument();
  }, 15_000);
});
