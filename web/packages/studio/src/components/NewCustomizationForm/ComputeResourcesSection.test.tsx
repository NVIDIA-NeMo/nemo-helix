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

const Harness: FC<{ expertParallelSize?: number; numNodes?: number }> = ({
  expertParallelSize,
  numNodes = 1,
}) => {
  const form = useForm<CustomizationFormFields>({
    defaultValues: {
      ...FORM_DEFAULTS,
      automodel: {
        ...FORM_DEFAULTS.automodel,
        model: 'ws/model',
        parallelism: {
          ...FORM_DEFAULTS.automodel.parallelism,
          num_nodes: numNodes,
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

const input = (name: string) => {
  const el = screen
    .getAllByRole('spinbutton')
    .find((candidate) => candidate.getAttribute('name') === `automodel.parallelism.${name}`);
  if (!el) throw new Error(`${name} input not found`);
  return el;
};

const typeAndBlur = async (name: string, value: string) => {
  await userEvent.clear(input(name));
  if (value) await userEvent.type(input(name), value);
  await userEvent.tab();
};

const setGpusPerNode = (gpus: number) => typeAndBlur('num_gpus_per_node', String(gpus));

describe('ComputeResourcesSection expert parallel size', () => {
  it('replaces an invalid size when the GPU count changes', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={4} />);
    await setGpusPerNode(2);
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it('clears the size when the run drops to one GPU', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={4} />);
    await setGpusPerNode(1);
    expect(screen.getByTestId('ep')).toHaveTextContent('unset');
  });

  it('keeps a size that still fits the new GPU count', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={2} />);
    await setGpusPerNode(8);
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it('reconciles once typing ends, not on the digits on the way', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={8} numNodes={2} />);
    // At "1" (4 GPUs) 8 would not fit; at 12 nodes (48 GPUs) it does.
    await typeAndBlur('num_nodes', '12');
    expect(screen.getByTestId('ep')).toHaveTextContent('8');
  });

  it('reconciles when the node count is left empty', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={8} numNodes={2} />);
    await typeAndBlur('num_nodes', '');
    expect(screen.getByTestId('ep')).toHaveTextContent('4');
  });

  it('reconciles against the clamped node count', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={16} numNodes={4} />);
    // 20 clamps to 16 nodes (64 GPUs), where 16 still fits.
    await typeAndBlur('num_nodes', '20');
    expect(screen.getByTestId('ep')).toHaveTextContent('16');
  });

  it('accepts a size up to the total GPU count', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={2} numNodes={16} />);
    await typeAndBlur('expert_parallel_size', '64');
    expect(screen.getByTestId('ep')).toHaveTextContent('64');
  });

  it('reconciles when the tensor parallel size changes', async () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={4} />);
    await userEvent.click(screen.getByText('Advanced Parallelism'));
    await typeAndBlur('tensor_parallel_size', '2');
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it.each([
    ['a dense model', { spec: {} } as ModelEntity],
    ['a model still loading', undefined],
  ])('keeps a set size for %s when the GPU count changes', async (_, entity) => {
    modelEntity.current = entity;
    render(<Harness expertParallelSize={2} />);
    await setGpusPerNode(8);
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it('shows the field and a warning for an invalid size on a MoE model', () => {
    modelEntity.current = moeModel;
    render(<Harness expertParallelSize={3} />);
    expect(screen.getByText('Expert Parallel Size')).toBeVisible();
    expect(
      screen.getByText(/This model needs an expert parallel size of 2 or 4/)
    ).toBeInTheDocument();
  });

  it('keeps the field in Advanced Parallelism for a dense model', () => {
    modelEntity.current = { spec: {} } as ModelEntity;
    render(<Harness />);
    expect(screen.queryByText('Expert Parallel Size')).not.toBeVisible();
  });
});
