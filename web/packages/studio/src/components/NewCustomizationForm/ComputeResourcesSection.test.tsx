// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ComputeResourcesSection } from '@studio/components/NewCustomizationForm/ComputeResourcesSection';
import { render, screen } from '@studio/tests/util/render';
import { type CustomizationFormFields, FORM_DEFAULTS } from '@studio/util/forms/customization';
import userEvent from '@testing-library/user-event';
import type { FC } from 'react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';

const modelEntity = vi.hoisted(() => ({ current: undefined as ModelEntity | undefined }));
vi.mock('@nemo/common/src/api/models/useModelEntity', () => ({
  useModelEntity: () => modelEntity.current,
}));

const Harness: FC<{ expertParallelSize?: number }> = ({ expertParallelSize }) => {
  const form = useForm<CustomizationFormFields>({
    defaultValues: {
      ...FORM_DEFAULTS,
      automodel: {
        ...FORM_DEFAULTS.automodel,
        model: 'ws/model',
        parallelism: {
          ...FORM_DEFAULTS.automodel.parallelism,
          num_gpus_per_node: 4,
          expert_parallel_size: expertParallelSize,
        },
      },
    },
  });
  const ep = useWatch({
    control: form.control,
    name: 'automodel.parallelism.expert_parallel_size',
  });
  return (
    <FormProvider {...form}>
      <ComputeResourcesSection />
      <output data-testid="ep">{ep ?? 'unset'}</output>
    </FormProvider>
  );
};

const moeModel = {
  spec: {
    checkpoint_model_name: 'NemotronHForCausalLM',
    moe_config: { num_experts: 128, num_experts_per_tok: 6, num_expert_layers: 23 },
  },
} as ModelEntity;

const setGpusPerNode = async (gpus: number) => {
  const input = screen
    .getAllByRole('spinbutton')
    .find((el) => el.getAttribute('name') === 'automodel.parallelism.num_gpus_per_node');
  if (!input) throw new Error('GPUs per Node input not found');
  await userEvent.clear(input);
  await userEvent.type(input, String(gpus));
};

describe('ComputeResourcesSection expert parallel size', () => {
  it('replaces an invalid size when the GPU count changes', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={4} />);
    await setGpusPerNode(2);
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it('keeps a size that still fits the new GPU count', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={2} />);
    await setGpusPerNode(8);
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it('shows the field and a warning for an invalid size on a MoE model', () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={3} />);
    expect(screen.getByText('Expert Parallel Size')).toBeVisible();
    expect(screen.getByText(/need an expert parallel size of 2 or 4/)).toBeInTheDocument();
  });

  it('keeps the field in Advanced Parallelism for a dense model', () => {
    modelEntity.current = { spec: {} } as ModelEntity;
    render(<Harness />);
    expect(screen.queryByText('Expert Parallel Size')).not.toBeVisible();
  });
});
