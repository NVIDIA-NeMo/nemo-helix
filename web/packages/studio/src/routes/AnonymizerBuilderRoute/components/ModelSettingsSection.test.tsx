// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelSettingsSection } from '@studio/routes/AnonymizerBuilderRoute/components/ModelSettingsSection';
import {
  activeRolesForStrategy,
  DETECTOR_ROLE,
  STRATEGY_SUBSTITUTE,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import {
  getAnonymizerFormDefaults,
  type AnonymizerFormData,
} from '@studio/routes/AnonymizerBuilderRoute/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, within } from '@testing-library/react';
import type { FC, ReactNode } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({ useWorkspaceFromPath: () => 'ws' }));

vi.mock('@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels', () => ({
  useAnonymizerModels: () => ({
    models: [],
    items: [{ label: 'nemotron-3-nano-30b-a3b', value: 'nvidia/nemotron-3-nano-30b-a3b' }],
    isLoading: false,
    applyModel: vi.fn(),
  }),
}));

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

const TestWrapper: FC<{ children: ReactNode }> = ({ children }) => {
  const methods = useForm<AnonymizerFormData>({ defaultValues: getAnonymizerFormDefaults() });
  return (
    <QueryClientProvider client={queryClient}>
      <FormProvider {...methods}>{children}</FormProvider>
    </QueryClientProvider>
  );
};

describe('ModelSettingsSection', () => {
  it('offers sampling params for every role except the GLiNER detector', async () => {
    render(
      <TestWrapper>
        <ModelSettingsSection />
      </TestWrapper>
    );

    const triggers = await screen.findAllByTestId('params-dropdown-trigger');
    expect(triggers).toHaveLength(activeRolesForStrategy(STRATEGY_SUBSTITUTE).length - 1);

    expect(
      within(screen.getByTestId(`role-settings-${DETECTOR_ROLE}`)).queryByTestId(
        'params-dropdown-trigger'
      )
    ).not.toBeInTheDocument();
    expect(
      within(screen.getByTestId('role-settings-entity_validator')).getByTestId(
        'params-dropdown-trigger'
      )
    ).toBeInTheDocument();
  });

  it('uses the shared model selector for every role except the GLiNER detector', () => {
    render(
      <TestWrapper>
        <ModelSettingsSection />
      </TestWrapper>
    );

    expect(screen.getAllByTestId('model-select-v2-trigger')).toHaveLength(
      activeRolesForStrategy(STRATEGY_SUBSTITUTE).length - 1
    );
    expect(
      within(screen.getByTestId(`role-settings-${DETECTOR_ROLE}`)).queryByTestId(
        'model-select-v2-trigger'
      )
    ).not.toBeInTheDocument();
  });
});
