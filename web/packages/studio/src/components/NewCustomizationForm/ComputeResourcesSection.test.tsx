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

const nemotronMoe = {
  spec: {
    checkpoint_model_name: 'NemotronHForCausalLM',
    moe_config: { num_experts: 128, num_experts_per_tok: 6, num_expert_layers: 23 },
  },
} as ModelEntity;

const Harness: FC<{ gpus: number; expertParallelSize?: number }> = ({
  gpus,
  expertParallelSize,
}) => {
  const form = useForm<CustomizationFormFields>({
    defaultValues: {
      ...FORM_DEFAULTS,
      automodel: {
        ...FORM_DEFAULTS.automodel,
        model: 'ws/nemotron',
        parallelism: {
          ...FORM_DEFAULTS.automodel.parallelism,
          num_gpus_per_node: gpus,
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
      {[1, 2, 8].map((n) => (
        <button
          key={n}
          type="button"
          onClick={() => form.setValue('automodel.parallelism.num_gpus_per_node', n)}
        >
          {`${n} GPUs`}
        </button>
      ))}
    </FormProvider>
  );
};

describe('ComputeResourcesSection expert parallelism', () => {
  beforeEach(() => {
    modelEntity.current = nemotronMoe;
  });

  it('pre-fills the largest valid size for a multi-GPU MoE run and follows the GPU count', async () => {
    render(<Harness gpus={4} />);
    expect(await screen.findByTestId('ep')).toHaveTextContent('4');
    expect(screen.getByText('Expert Parallel Size')).toBeVisible();

    await userEvent.click(screen.getByRole('button', { name: '2 GPUs' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('2');

    await userEvent.click(screen.getByRole('button', { name: '8 GPUs' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('8');

    await userEvent.click(screen.getByRole('button', { name: '1 GPUs' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('unset');
  });

  it('keeps an explicit size and flags it when it does not fit', async () => {
    render(<Harness gpus={4} expertParallelSize={8} />);
    expect(screen.getByTestId('ep')).toHaveTextContent('8');
    expect(
      await screen.findByText('With 4 GPUs, expert parallel size must be one of 2, 4.')
    ).toBeInTheDocument();
  });

  it('leaves the size unset for a model that does not need it', () => {
    modelEntity.current = { spec: { checkpoint_model_name: 'LlamaForCausalLM' } } as ModelEntity;
    render(<Harness gpus={4} />);
    expect(screen.getByTestId('ep')).toHaveTextContent('unset');
  });
});
