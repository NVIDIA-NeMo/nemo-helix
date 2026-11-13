// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEntitiesCreateWorkspace } from '@nemo/sdk/generated/platform/entity-store';
import { WorkspaceCreateModal } from '@studio/components/WorkspaceCreateModal';
import { render } from '@studio/tests/util/render';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AxiosError, AxiosHeaders } from 'axios';

vi.mock('@nemo/sdk/generated/platform/entity-store', () => ({
  getEntitiesListWorkspacesQueryKey: () => ['/apis/entities/v2/workspaces'],
  useEntitiesCreateWorkspace: vi.fn(),
}));

const axiosErrorWithStatus = (status: number, detail: string) =>
  new AxiosError('request failed', undefined, undefined, undefined, {
    status,
    statusText: '',
    data: { detail },
    headers: {},
    config: { headers: new AxiosHeaders() },
  });

const submitWithError = async (error: AxiosError) => {
  vi.mocked(useEntitiesCreateWorkspace).mockReturnValue({
    mutateAsync: vi.fn().mockRejectedValue(error),
  } as never);
  render(
    <TestProviders>
      <WorkspaceCreateModal open onClose={vi.fn()} />
    </TestProviders>
  );
  await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'team-space');
  await userEvent.click(screen.getByRole('button', { name: 'Create' }));
};

describe('WorkspaceCreateModal', () => {
  it('explains a permission denial instead of showing the raw response', async () => {
    await submitWithError(axiosErrorWithStatus(403, 'Forbidden'));

    expect(await screen.findByTestId('mock-toast-error')).toHaveTextContent(
      "You don't have permission to create workspaces. Ask an administrator for access."
    );
  });

  it('shows the server detail for other failures', async () => {
    await submitWithError(axiosErrorWithStatus(500, 'database unavailable'));

    expect(await screen.findByTestId('mock-toast-error')).toHaveTextContent(
      'Failed to create workspace: database unavailable'
    );
  });
});
