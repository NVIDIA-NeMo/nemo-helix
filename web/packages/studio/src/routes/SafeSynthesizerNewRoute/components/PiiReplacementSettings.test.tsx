// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelProvider } from '@nemo/sdk/generated/platform/schema';
import { listModelProvidersHandler } from '@studio/mocks/handlers/models';
import { server } from '@studio/mocks/node';
import { PiiReplacementSettings } from '@studio/routes/SafeSynthesizerNewRoute/components/PiiReplacementSettings';
import {
  type SafeSynthesizerFormData,
  getSafeSynthesizerFormDefaults,
} from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'test-workspace',
}));

const provider = (workspace: string, name: string): ModelProvider => ({
  workspace,
  name,
  host_url: 'https://example.com',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
});

const FormWrapper = ({ children }: { children: ReactNode }) => {
  const methods = useForm<SafeSynthesizerFormData>({
    defaultValues: getSafeSynthesizerFormDefaults(),
  });
  return <FormProvider {...methods}>{children}</FormProvider>;
};

const PiiValuesProbe = () => {
  const providerRef = useWatch<SafeSynthesizerFormData>({ name: 'spec.pii_model_provider' });
  const modelId = useWatch<SafeSynthesizerFormData>({
    name: 'spec.config.replace_pii.llm.model_id',
  });
  return (
    <>
      <div data-testid="pii-model-provider">{String(providerRef)}</div>
      <div data-testid="pii-model-id">{String(modelId)}</div>
    </>
  );
};

const renderSettings = () =>
  render(
    <FormWrapper>
      <PiiReplacementSettings />
      <PiiValuesProbe />
    </FormWrapper>
  );

const providerSelect = () => screen.getByRole('combobox', { name: /^Model provider/ });

describe('PiiReplacementSettings', () => {
  beforeEach(() => {
    server.use(
      listModelProvidersHandler([
        provider('test-workspace', 'pii-provider'),
        provider('shared', 'nim-provider'),
      ])
    );
  });

  it('shows the provider select and model ID input while PII replacement is on', async () => {
    renderSettings();

    expect(await screen.findByRole('switch')).toBeChecked();
    expect(providerSelect()).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /^Model ID/ })).toBeInTheDocument();
  });

  it('hides the provider select and model ID input when PII replacement is turned off', async () => {
    const user = userEvent.setup();
    renderSettings();

    await user.click(await screen.findByRole('switch'));

    await waitFor(() => {
      expect(screen.queryByRole('combobox', { name: /^Model provider/ })).not.toBeInTheDocument();
    });
    expect(screen.queryByRole('textbox', { name: /^Model ID/ })).not.toBeInTheDocument();
  });

  it('lists workspace providers as workspace/name and writes the picks into the form', async () => {
    const user = userEvent.setup();
    renderSettings();

    await user.click(providerSelect());
    await waitFor(() => {
      expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
        'test-workspace/pii-provider',
        'shared/nim-provider',
      ]);
    });
    await user.click(screen.getByRole('option', { name: 'shared/nim-provider' }));
    await user.type(screen.getByRole('textbox', { name: /^Model ID/ }), 'nvidia/nemotron');

    await waitFor(() => {
      expect(screen.getByTestId('pii-model-provider')).toHaveTextContent('shared/nim-provider');
    });
    expect(screen.getByTestId('pii-model-id')).toHaveTextContent('nvidia/nemotron');
  });

  it('says so when the workspace has no model providers', async () => {
    server.use(listModelProvidersHandler([]));
    renderSettings();

    expect(
      await screen.findByText(
        'No model providers in this workspace. Add one under Inference Providers.'
      )
    ).toBeInTheDocument();
  });
});
