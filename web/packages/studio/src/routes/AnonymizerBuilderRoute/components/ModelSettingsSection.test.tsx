// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelSelectV2Props } from '@nemo/common/src/components/ModelSelectV2/types';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ModelSettingsSection } from '@studio/routes/AnonymizerBuilderRoute/components/ModelSettingsSection';
import {
  activeRolesForStrategy,
  ANONYMIZER_PARAM_METADATA,
  DETECTOR_ROLE,
  STRATEGY_SUBSTITUTE,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import {
  getAnonymizerFormDefaults,
  type AnonymizerFormData,
} from '@studio/routes/AnonymizerBuilderRoute/schema';
import { act, render, screen, within } from '@studio/tests/util/render';
import type { FC } from 'react';
import { FormProvider, useForm, type UseFormReturn } from 'react-hook-form';

const applyModel = vi.fn();
const systemModel = {
  workspace: 'system',
  name: 'nano',
  model_providers: ['system/nim'],
} as unknown as ModelEntity;

vi.mock('@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels', () => ({
  useAnonymizerModels: () => ({
    models: [],
    items: [{ label: 'nemotron-3-nano-30b-a3b', value: 'nvidia/nemotron-3-nano-30b-a3b' }],
    groups: [{ workspace: 'system', models: [systemModel] }],
    isLoading: false,
    applyModel,
  }),
}));

// The real picker renders; the spy only records its props so a test can drive a selection.
const selectSpy = vi.fn<(props: ModelSelectV2Props) => void>();

vi.mock('@nemo/common/src/components/ModelSelectV2', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/common/src/components/ModelSelectV2')>();
  return {
    ...actual,
    ModelSelectV2: (props: ModelSelectV2Props) => {
      selectSpy(props);
      return <actual.ModelSelectV2 {...props} />;
    },
  };
});

let form: UseFormReturn<AnonymizerFormData>;

const Harness: FC = () => {
  form = useForm<AnonymizerFormData>({ defaultValues: getAnonymizerFormDefaults() });
  return (
    <FormProvider {...form}>
      <ModelSettingsSection />
    </FormProvider>
  );
};

const lastSelectPropsFor = (label: string): ModelSelectV2Props =>
  selectSpy.mock.calls.filter(([props]) => props['aria-label'] === label).at(-1)![0];

describe('ModelSettingsSection', () => {
  beforeEach(() => {
    applyModel.mockClear();
    selectSpy.mockClear();
  });

  it('gives every sampling role the integrated model + params picker', async () => {
    render(<Harness />);

    const samplingRoles = activeRolesForStrategy(STRATEGY_SUBSTITUTE).length - 1;
    expect(await screen.findAllByTestId('model-select-v2-trigger')).toHaveLength(samplingRoles);
    expect(screen.getAllByTestId('params-dropdown-trigger')).toHaveLength(samplingRoles);
    expect(
      within(screen.getByTestId('role-settings-entity_validator')).getByTestId(
        'params-dropdown-trigger'
      )
    ).toBeInTheDocument();
  });

  it('keeps the GLiNER detector on its own select, with no sampling params', () => {
    render(<Harness />);
    const detector = screen.getByTestId(`role-settings-${DETECTOR_ROLE}`);

    expect(within(detector).queryByTestId('model-select-v2-trigger')).not.toBeInTheDocument();
    expect(within(detector).queryByTestId('params-dropdown-trigger')).not.toBeInTheDocument();
  });

  it("lists the providers' models, with the anonymizer bounds and defaults for sampling params", () => {
    render(<Harness />);

    expect(lastSelectPropsFor('Entity Validator')).toEqual(
      expect.objectContaining({
        showParams: true,
        paramsFieldMetadata: ANONYMIZER_PARAM_METADATA,
        groups: [{ workspace: 'system', models: [systemModel] }],
      })
    );
  });

  it('writes the picked model and its params into the role', async () => {
    render(<Harness />);
    await act(async () =>
      lastSelectPropsFor('Entity Validator').onValueChange({
        model: 'system/nano',
        entity: systemModel,
      })
    );
    await act(async () =>
      lastSelectPropsFor('Entity Validator').onInferenceParamsChange?.({ temperature: 0.5 })
    );

    expect(form.getValues('roleModels.entity_validator.modelId')).toBe('system/nano');
    expect(form.getValues('roleModels.entity_validator.params')).toEqual({ temperature: 0.5 });
    expect(applyModel).toHaveBeenCalledWith('entity_validator', 'system/nano');
    expect(lastSelectPropsFor('Entity Validator').value).toEqual({ model: 'system/nano' });
  });

  it('shows the required-model error on the role that is missing one', async () => {
    render(<Harness />);

    act(() =>
      form.setError('roleModels.entity_validator.modelId', {
        message: 'Pick a model for this role',
      })
    );

    expect(
      await within(screen.getByTestId('role-settings-entity_validator')).findByText(
        'Pick a model for this role'
      )
    ).toBeInTheDocument();
  });
});
