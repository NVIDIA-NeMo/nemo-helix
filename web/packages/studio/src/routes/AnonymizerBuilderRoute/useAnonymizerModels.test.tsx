// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getAnonymizerFormDefaults,
  type AnonymizerFormData,
} from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useAnonymizerModels } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels';
import { act, renderHook } from '@testing-library/react';
import type { FC, ReactNode } from 'react';
import { FormProvider, useForm, type UseFormReturn } from 'react-hook-form';

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'default',
}));

vi.mock('@nemo/sdk/generated/platform/model-providers', () => ({
  useModelsListProviders: () => ({
    isLoading: false,
    data: {
      data: [
        {
          workspace: 'default',
          name: 'nim',
          served_models: [
            { model_entity_id: 'default/nano', served_model_name: 'nvidia/nano' },
            { model_entity_id: 'system/super', served_model_name: 'nvidia/super' },
          ],
        },
      ],
    },
  }),
}));

let form: UseFormReturn<AnonymizerFormData>;

const Wrapper: FC<{ children: ReactNode }> = ({ children }) => {
  form = useForm<AnonymizerFormData>({ defaultValues: getAnonymizerFormDefaults() });
  return <FormProvider {...form}>{children}</FormProvider>;
};

describe('useAnonymizerModels', () => {
  it("groups the providers' models by the workspace each one lives in", () => {
    const { result } = renderHook(() => useAnonymizerModels(), { wrapper: Wrapper });

    expect(
      result.current.groups.map((group) => [group.workspace, group.models.map((m) => m.name)])
    ).toEqual([
      ['default', ['nano']],
      ['system', ['super']],
    ]);
  });

  it('uses the served name and provider the providers list reports', () => {
    const { result } = renderHook(() => useAnonymizerModels(), { wrapper: Wrapper });

    act(() => result.current.applyModel('rewriter', 'default/nano'));

    expect(form.getValues('roleModels.rewriter.model')).toBe('nvidia/nano');
    expect(form.getValues('roleModels.rewriter.provider')).toBe('default/nim');
  });

  it('clears the role for an id it cannot resolve, such as the built-in detector', () => {
    const { result } = renderHook(() => useAnonymizerModels(), { wrapper: Wrapper });

    act(() => result.current.applyModel('entity_detector', '__builtin_gliner2__'));

    expect(form.getValues('roleModels.entity_detector.model')).toBe('');
    expect(form.getValues('roleModels.entity_detector.provider')).toBe('');
  });
});
