// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixSecretResponse } from '@nemo/sdk/generated/platform/schema';
import { EditSecretModal } from '@studio/routes/SecretsListRoute/EditSecretModal';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const { mockGetSecret, mockUpdateSecret } = vi.hoisted(() => ({
  mockGetSecret: vi.fn(),
  mockUpdateSecret: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/secrets', async () => {
  const actual = await vi.importActual<typeof import('@nemo/sdk/generated/platform/secrets')>(
    '@nemo/sdk/generated/platform/secrets'
  );
  return {
    ...actual,
    useSecretsGetSecret: (workspace: string, name: string) => mockGetSecret(workspace, name),
    useSecretsUpdateSecret: () => ({
      mutateAsync: mockUpdateSecret,
      isPending: false,
      reset: vi.fn(),
      error: null,
    }),
  };
});

const renderModal = () =>
  render(
    <TestProviders>
      <EditSecretModal workspace="default" name="github-token" open onClose={vi.fn()} />
    </TestProviders>
  );

describe('EditSecretModal', () => {
  beforeEach(() => {
    mockUpdateSecret.mockResolvedValue({});
  });

  it('waits for the secret before showing the form', () => {
    mockGetSecret.mockReturnValue({ data: undefined, error: null, isLoading: true });

    renderModal();

    expect(screen.getByLabelText('Loading github-token')).toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText('Enter new value to replace the current one')
    ).not.toBeInTheDocument();
  });

  it('fetches the secret it was opened for and prefills the stored description', async () => {
    mockGetSecret.mockReturnValue({
      data: { name: 'github-token', description: 'Read-only PAT' } as HelixSecretResponse,
      error: null,
      isLoading: false,
    });

    renderModal();

    expect(mockGetSecret).toHaveBeenCalledWith('default', 'github-token');
    expect(await screen.findByDisplayValue('Read-only PAT')).toBeInTheDocument();
    expect(await screen.findByDisplayValue('github-token')).toBeDisabled();
  });

  it('replaces the value without touching the name', async () => {
    const user = userEvent.setup();
    mockGetSecret.mockReturnValue({
      data: { name: 'github-token', description: 'Read-only PAT' } as HelixSecretResponse,
      error: null,
      isLoading: false,
    });

    renderModal();

    await user.type(
      await screen.findByPlaceholderText('Enter new value to replace the current one'),
      'ghp_replacement'
    );
    await user.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() =>
      expect(mockUpdateSecret).toHaveBeenCalledWith({
        workspace: 'default',
        name: 'github-token',
        data: { description: 'Read-only PAT', value: 'ghp_replacement' },
      })
    );
  });

  it('surfaces a failure to load the secret', () => {
    mockGetSecret.mockReturnValue({
      data: undefined,
      error: new Error('Secret not found'),
      isLoading: false,
    });

    renderModal();

    expect(screen.getByText('Secret not found')).toBeInTheDocument();
  });
});
