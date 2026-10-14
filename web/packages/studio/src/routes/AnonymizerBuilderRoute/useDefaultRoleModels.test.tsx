// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  BUILTIN_DETECTOR_VALUE,
  DETECTOR_ROLE,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import {
  anonymizerFormSchema,
  getAnonymizerFormDefaults,
  roleModelSchema,
  type AnonymizerFormData,
} from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useDefaultRoleModels } from '@studio/routes/AnonymizerBuilderRoute/useDefaultRoleModels';
import { renderHook } from '@testing-library/react';
import { type FC, type ReactNode } from 'react';
import { FormProvider, useForm, useFormContext } from 'react-hook-form';

const mockModels = vi.hoisted(() => ({ value: [] as { id: string; name: string }[] }));

vi.mock('@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels', () => ({
  useAnonymizerModels: () => ({
    models: mockModels.value,
    items: mockModels.value.map((m) => ({ label: m.name, value: m.id })),
    isLoading: false,
    // Mirror the real applyModel: unknown id (the sentinel) writes empty model/provider.
    applyModel: (role: string, id: string) => {
      const selected = mockModels.value.find((m) => m.id === id);
      formRef?.setValue(`roleModels.${role}.model`, selected ? selected.name : '');
      formRef?.setValue(`roleModels.${role}.provider`, selected ? 'default/nvidia' : '');
    },
  }),
}));

// Captured so the mocked applyModel can write into the same form instance.
let formRef: ReturnType<typeof useFormContext<AnonymizerFormData>> | null = null;

const Wrapper: FC<{ children: ReactNode }> = ({ children }) => {
  const methods = useForm<AnonymizerFormData>({ defaultValues: getAnonymizerFormDefaults() });
  formRef = methods;
  return <FormProvider {...methods}>{children}</FormProvider>;
};

describe('useDefaultRoleModels', () => {
  afterEach(() => {
    mockModels.value = [];
    formRef = null;
  });

  it('defaults the entity detector to a complete, individually-valid built-in selection with an empty workspace', () => {
    mockModels.value = [];
    renderHook(() => useDefaultRoleModels(), { wrapper: Wrapper });

    const detector = formRef?.getValues(`roleModels.${DETECTOR_ROLE}`);
    expect(detector?.modelId).toBe(BUILTIN_DETECTOR_VALUE);
    // The complete object must satisfy the base role schema (model/provider are required strings),
    // so the built-in detector never blocks submission on its own — the P1 regression guard.
    expect(detector?.model).toBe('');
    expect(detector?.provider).toBe('');
    expect(roleModelSchema.safeParse(detector).success).toBe(true);
  });

  it('yields a fully valid form once the other roles are filled, with the detector left built-in', () => {
    mockModels.value = [{ id: 'm1', name: 'nvidia/nemotron-3-nano-30b-a3b' }];
    renderHook(() => useDefaultRoleModels(), { wrapper: Wrapper });

    expect(formRef?.getValues(`roleModels.${DETECTOR_ROLE}.modelId`)).toBe(BUILTIN_DETECTOR_VALUE);
    const result = anonymizerFormSchema.safeParse({
      ...formRef?.getValues(),
      source: 'https://example.com/data.csv',
    });
    expect(result.success).toBe(true);
  });

  it('auto-picks a chat model for the other roles when the workspace has models', () => {
    mockModels.value = [{ id: 'm1', name: 'nvidia/nemotron-3-nano-30b-a3b' }];
    renderHook(() => useDefaultRoleModels(), { wrapper: Wrapper });

    expect(formRef?.getValues(`roleModels.${DETECTOR_ROLE}.modelId`)).toBe(BUILTIN_DETECTOR_VALUE);
    expect(formRef?.getValues('roleModels.entity_validator.modelId')).toBe('m1');
    expect(formRef?.getValues('roleModels.replacement_generator.modelId')).toBe('m1');
  });
});
