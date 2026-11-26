// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { suppressConsoleError } from '@nemo/testing/utils/suppress-console';
import { listModelProvidersHandler } from '@studio/mocks/handlers/models';
import { server } from '@studio/mocks/node';
import { SafeSynthesizerNewRoute } from '@studio/routes/SafeSynthesizerNewRoute';
import {
  PII_MODEL_ID_REQUIRED,
  PII_MODEL_PROVIDER_REQUIRED,
} from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const WORKSPACE = 'test-workspace';
const PROVIDER_REF = `${WORKSPACE}/pii-provider`;
const mockMutate = vi.fn();

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => WORKSPACE,
}));

vi.mock('@nemo/sdk/generated/safe-synthesizer/safe-synthesizer', () => ({
  useSafeSynthesizerCreateJob: () => ({ mutate: mockMutate, isPending: false }),
}));

vi.mock('@studio/routes/SafeSynthesizerNewRoute/components/TrainingData', async () => {
  const { useEffect } = await import('react');
  const { useFormContext } = await import('react-hook-form');
  return {
    TrainingData: () => {
      const { setValue } = useFormContext();
      useEffect(() => {
        setValue('spec.data_source', `${WORKSPACE}/fileset#data.csv`);
      }, [setValue]);
      return null;
    },
  };
});

const setupRoute = async () => {
  const user = userEvent.setup();
  render(<SafeSynthesizerNewRoute />);
  await user.click(await screen.findByText('Show Advanced Parameters'));
  return user;
};

describe('SafeSynthesizerNewRoute PII discovery model', () => {
  beforeEach(() => {
    mockMutate.mockClear();
    suppressConsoleError('Form validation errors:');
    server.use(
      listModelProvidersHandler([
        {
          workspace: WORKSPACE,
          name: 'pii-provider',
          host_url: 'https://example.com',
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
        },
      ])
    );
  });

  it('blocks submit until a provider and model ID are set', async () => {
    const user = await setupRoute();

    await user.click(screen.getByRole('button', { name: 'Continue' }));

    expect(await screen.findByText(PII_MODEL_PROVIDER_REQUIRED)).toBeInTheDocument();
    expect(screen.getByText(PII_MODEL_ID_REQUIRED)).toBeInTheDocument();
    expect(mockMutate).not.toHaveBeenCalled();

    await user.click(screen.getByRole('combobox', { name: /^Model provider/ }));
    await user.click(await screen.findByRole('option', { name: PROVIDER_REF }));
    await user.click(screen.getByRole('button', { name: 'Continue' }));

    expect(await screen.findByText(PII_MODEL_ID_REQUIRED)).toBeInTheDocument();
    expect(mockMutate).not.toHaveBeenCalled();
  });

  it('submits the provider and model ID in the job spec', async () => {
    const user = await setupRoute();

    await user.click(screen.getByRole('combobox', { name: /^Model provider/ }));
    await user.click(await screen.findByRole('option', { name: PROVIDER_REF }));
    await user.type(screen.getByRole('textbox', { name: /^Model ID/ }), 'nvidia/nemotron');
    await user.click(screen.getByRole('button', { name: 'Continue' }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    const { workspace, data } = mockMutate.mock.calls[0][0];
    expect(workspace).toBe(WORKSPACE);
    expect(data.spec.pii_model_provider).toBe(PROVIDER_REF);
    expect(data.spec.config.enable_replace_pii).toBe(true);
    expect(data.spec.config.replace_pii).toEqual({ llm: { model_id: 'nvidia/nemotron' } });
  });

  it('submits without PII discovery settings when PII replacement is off', async () => {
    const user = await setupRoute();

    await user.click(screen.getByRole('switch'));
    await user.click(screen.getByRole('button', { name: 'Continue' }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    const { data } = mockMutate.mock.calls[0][0];
    expect(data.spec).not.toHaveProperty('pii_model_provider');
    expect(data.spec.config).not.toHaveProperty('replace_pii');
    expect(data.spec.config.enable_replace_pii).toBe(false);
  });
});
