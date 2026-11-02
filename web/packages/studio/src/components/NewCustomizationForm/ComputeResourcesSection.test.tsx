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

const moeModel = {
  spec: { moe_config: { num_experts: 128, num_experts_per_tok: 6, num_expert_layers: 23 } },
} as ModelEntity;

interface HarnessProps {
  gpus: number;
  expertParallelSize?: number;
}

const Harness: FC<HarnessProps> = ({ gpus, expertParallelSize }) => {
  const form = useForm<CustomizationFormFields>({
    defaultValues: {
      ...FORM_DEFAULTS,
      automodel: {
        ...FORM_DEFAULTS.automodel,
        model: 'ws/model',
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
      {[2, 8].map((n) => (
        <button
          key={n}
          type="button"
          onClick={() => form.setValue('automodel.parallelism.num_gpus_per_node', n)}
        >
          {`${n} GPUs`}
        </button>
      ))}
      <button
        type="button"
        onClick={() => form.setValue('automodel.parallelism.expert_parallel_size', 3)}
      >
        EP 3
      </button>
    </FormProvider>
  );
};

describe('ComputeResourcesSection expert parallelism', () => {
  beforeEach(() => {
    modelEntity.current = moeModel;
  });

  it('pre-fills the largest valid size and replaces it only once the GPU count makes it invalid', async () => {
    render(<Harness gpus={4} />);
    expect(await screen.findByTestId('ep')).toHaveTextContent('4');

    await userEvent.click(screen.getByRole('button', { name: '2 GPUs' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('2');

    await userEvent.click(screen.getByRole('button', { name: '8 GPUs' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it('keeps a valid size from a template', async () => {
    render(<Harness gpus={4} expertParallelSize={2} />);
    expect(await screen.findByTestId('ep')).toHaveTextContent('2');
  });

  it('replaces a template size that does not fit the GPU count', async () => {
    render(<Harness gpus={4} expertParallelSize={8} />);
    expect(await screen.findByTestId('ep')).toHaveTextContent('4');
  });

  it('warns about an invalid size entered by the user instead of replacing it', async () => {
    render(<Harness gpus={4} />);
    await userEvent.click(screen.getByRole('button', { name: 'EP 3' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('3');
    expect(await screen.findByText(/without an expert parallel size of 2, 4/)).toBeInTheDocument();
  });

  it('leaves the size unset for a dense model', () => {
    modelEntity.current = { spec: {} } as ModelEntity;
    render(<Harness gpus={4} />);
    expect(screen.getByTestId('ep')).toHaveTextContent('unset');
  });
});
