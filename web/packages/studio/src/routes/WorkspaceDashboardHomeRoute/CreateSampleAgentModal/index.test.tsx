// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SampleAgentResponse } from '@nemo/sdk/generated/agents/schema';
import { getWorkspaceDashboardRoute } from '@studio/routes/utils';
import { CreateSampleAgentModal } from '@studio/routes/WorkspaceDashboardHomeRoute/CreateSampleAgentModal';
import { streamCreateSampleAgent } from '@studio/routes/WorkspaceDashboardHomeRoute/CreateSampleAgentModal/streamCreateSampleAgent';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute } from '@studio/tests/util/render';
import { act, fireEvent, screen, waitFor } from '@testing-library/react';

vi.mock(
  '@studio/routes/WorkspaceDashboardHomeRoute/CreateSampleAgentModal/streamCreateSampleAgent',
  () => ({ streamCreateSampleAgent: vi.fn() })
);

vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({
    value,
    onValueChange,
    ...props
  }: {
    value: { model: string } | null;
    onValueChange: (next: { model: string }) => void;
    'aria-label': string;
  }) => (
    <input
      aria-label={props['aria-label']}
      value={value?.model ?? ''}
      onChange={(event) => onValueChange({ model: event.target.value })}
    />
  ),
}));

const streamMock = vi.mocked(streamCreateSampleAgent);

const RESULT: SampleAgentResponse = {
  status: 'created',
  workspace: 'sample-1a2b3c4d',
  studio_url: '/studio/workspaces/sample-1a2b3c4d/dashboard',
  agent: 'email-security-triage',
  deployment: 'email-security-triage',
  deployment_status: 'pending',
};

/** Keeps the stream pending until the test settles it. */
const controlledStream = () => {
  let resolve: (result: SampleAgentResponse) => void = () => undefined;
  let reject: (error: Error) => void = () => undefined;
  streamMock.mockImplementation(
    () =>
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
  );
  return {
    resolve: (result: SampleAgentResponse) => act(async () => resolve(result)),
    reject: (error: Error) => act(async () => reject(error)),
  };
};

const onClose = vi.fn();

const renderModal = () =>
  renderRoute(undefined, {
    history: '/workspaces/my-ws/dashboard',
    routes: [
      {
        path: '/workspaces/my-ws/dashboard',
        element: <CreateSampleAgentModal open onClose={onClose} workspace="my-ws" />,
      },
      { path: '/workspaces/:workspace/dashboard', element: <LocationDisplay /> },
    ],
  });

const selectModelAndSubmit = async () => {
  fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'my-ws/my-model' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create' }));
  // The mutation starts the stream on a later tick.
  await waitFor(() => expect(streamMock).toHaveBeenCalled());
};

beforeEach(() => {
  streamMock.mockReset();
  onClose.mockReset();
});

describe('CreateSampleAgentModal', () => {
  it('keeps Create disabled until a model is chosen', async () => {
    renderModal();

    expect(await screen.findByRole('button', { name: 'Create' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'my-ws/my-model' } });
    expect(screen.getByRole('button', { name: 'Create' })).toBeEnabled();
  });

  it('spins the submit button while creating, then toasts and navigates to the sample dashboard', async () => {
    const stream = controlledStream();
    renderModal();
    await screen.findByRole('button', { name: 'Create' });

    await selectModelAndSubmit();

    expect(streamMock).toHaveBeenCalledWith(
      { model: 'my-ws/my-model' },
      expect.anything(),
      expect.any(AbortSignal)
    );
    expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();

    await stream.resolve(RESULT);

    expect(onClose).toHaveBeenCalled();
    expect(screen.getByText('Sample workspace "sample-1a2b3c4d" created')).toBeInTheDocument();
    expect((await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).textContent).toBe(
      getWorkspaceDashboardRoute('sample-1a2b3c4d')
    );
  });

  it('shows the error and stays open so the user can retry', async () => {
    const stream = controlledStream();
    renderModal();
    await screen.findByRole('button', { name: 'Create' });

    await selectModelAndSubmit();
    await stream.reject(new Error('Already deployed with another model'));

    expect(
      await screen.findByText(
        "Couldn't create the sample workspace: Already deployed with another model"
      )
    ).toBeInTheDocument();
    expect(screen.queryByLabelText('Loading...')).not.toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Create' })).toBeEnabled();
  });
});
