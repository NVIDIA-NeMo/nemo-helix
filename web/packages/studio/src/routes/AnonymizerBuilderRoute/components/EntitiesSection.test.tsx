// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EntitiesSection } from '@studio/routes/AnonymizerBuilderRoute/components/EntitiesSection';
import { ENTITY_MODE_AUTO } from '@studio/routes/AnonymizerBuilderRoute/constants';
import {
  type AnonymizerFormData,
  getAnonymizerFormDefaults,
} from '@studio/routes/AnonymizerBuilderRoute/schema';
import { render, screen } from '@testing-library/react';
import type { FC, ReactNode } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

vi.mock('@nemo/sdk/generated/anonymizer/anonymizer', () => ({
  useAnonymizerListEntityLabels: () => ({
    data: { data: ['email', 'ssn', 'first_name'] },
    isLoading: false,
  }),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'default',
}));

const TestWrapper: FC<{ defaultValues?: Partial<AnonymizerFormData>; children: ReactNode }> = ({
  defaultValues,
  children,
}) => {
  const methods = useForm<AnonymizerFormData>({
    defaultValues: { ...getAnonymizerFormDefaults(), ...defaultValues },
  });
  return <FormProvider {...methods}>{children}</FormProvider>;
};

describe('EntitiesSection', () => {
  it('states that auto-detect includes the defaults and hides the defaults checkbox', async () => {
    render(
      <TestWrapper defaultValues={{ entityMode: ENTITY_MODE_AUTO }}>
        <EntitiesSection />
      </TestWrapper>
    );

    expect(
      await screen.findByText(/Auto-detect includes all 3 available entities/)
    ).toBeInTheDocument();
    expect(screen.queryByText(/Also include all 3 available entities/)).not.toBeInTheDocument();
  });

  it('offers the defaults checkbox in custom mode', async () => {
    render(
      <TestWrapper>
        <EntitiesSection />
      </TestWrapper>
    );

    expect(
      await screen.findByText(/Custom mode only outputs the labels selected below/)
    ).toBeInTheDocument();
    expect(screen.getByText('Also include all 3 available entities')).toBeInTheDocument();
  });
});
