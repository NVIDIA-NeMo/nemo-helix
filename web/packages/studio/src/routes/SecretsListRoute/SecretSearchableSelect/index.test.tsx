// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixSecretResponse } from '@nemo/sdk/generated/platform/schema';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';

const { mockListSecrets } = vi.hoisted(() => ({ mockListSecrets: vi.fn() }));

vi.mock('@nemo/sdk/generated/platform/secrets', async () => {
  const actual = await vi.importActual<typeof import('@nemo/sdk/generated/platform/secrets')>(
    '@nemo/sdk/generated/platform/secrets'
  );
  return { ...actual, secretsListSecrets: mockListSecrets };
});

const secret = (name: string): HelixSecretResponse =>
  ({ name, workspace: 'default' }) as HelixSecretResponse;

interface HarnessProps {
  defaultSecret?: string;
  onRequestEditSecret?: (secretName: string) => void;
}

const Harness = ({ defaultSecret = '', onRequestEditSecret }: HarnessProps) => {
  const { control } = useForm({ defaultValues: { secretKey: defaultSecret } });
  return (
    <SecretSearchableSelect
      workspace="default"
      useControllerProps={{ control, name: 'secretKey' }}
      onRequestNewSecret={vi.fn()}
      onRequestEditSecret={onRequestEditSecret}
      formFieldProps={{ slotLabel: 'Access token secret' }}
    />
  );
};

type User = ReturnType<typeof userEvent.setup>;

const openSelect = async (user: User, props: HarnessProps = {}) => {
  render(
    <TestProviders>
      <Harness {...props} />
    </TestProviders>
  );
  await user.click(await screen.findByRole('combobox'));
};

describe('SecretSearchableSelect', () => {
  beforeEach(() => {
    mockListSecrets.mockResolvedValue({
      data: [secret('github-token'), secret('hf-token')],
      pagination: { page: 1, total_pages: 1, total_results: 2 },
    });
  });

  it('always offers secret creation from the footer', async () => {
    await openSelect(userEvent.setup());

    expect(await screen.findByText('New Secret')).toBeInTheDocument();
  });

  it('does not offer editing until a secret is selected', async () => {
    await openSelect(userEvent.setup(), { onRequestEditSecret: vi.fn() });

    expect(await screen.findByText('New Secret')).toBeInTheDocument();
    expect(screen.queryByText('Edit selected secret')).not.toBeInTheDocument();
  });

  it('edits the secret the form currently holds', async () => {
    const onRequestEditSecret = vi.fn();
    const user = userEvent.setup();
    await openSelect(user, { defaultSecret: 'github-token', onRequestEditSecret });

    await user.click(await screen.findByText('Edit selected secret'));

    expect(onRequestEditSecret).toHaveBeenCalledWith('github-token');
  });

  it('omits the edit entry when the caller has nowhere to open it', async () => {
    await openSelect(userEvent.setup(), { defaultSecret: 'github-token' });

    expect(await screen.findByText('New Secret')).toBeInTheDocument();
    expect(screen.queryByText('Edit selected secret')).not.toBeInTheDocument();
  });
});
