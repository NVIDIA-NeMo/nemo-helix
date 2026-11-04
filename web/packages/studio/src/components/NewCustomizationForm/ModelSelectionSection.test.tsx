// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ModelSelectionSection } from '@studio/components/NewCustomizationForm/ModelSelectionSection';
import { render, screen } from '@studio/tests/util/render';
import { type CustomizationFormFields, FORM_DEFAULTS } from '@studio/util/forms/customization';
import userEvent from '@testing-library/user-event';
import type { FC } from 'react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';

vi.mock('@nemo/common/src/api/models/useModelSearch', () => ({ useModelSearch: () => ({}) }));
vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({ useWorkspaceFromPath: () => 'ws' }));
vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  ModelSelectV2: ({ onValueChange }: { onValueChange: (selection: ModelSelection) => void }) => (
    <>
      <button
        type="button"
        onClick={() =>
          onValueChange({
            model: 'ws/moe',
            entity: {
              spec: {
                checkpoint_model_name: 'NemotronHForCausalLM',
                moe_config: { num_experts: 128, num_experts_per_tok: 6, num_expert_layers: 23 },
              },
            } as ModelEntity,
          })
        }
      >
        MoE
      </button>
      <button
        type="button"
        onClick={() => onValueChange({ model: 'ws/dense', entity: { spec: {} } as ModelEntity })}
      >
        Dense
      </button>
      <button
        type="button"
        onClick={() =>
          onValueChange({
            model: 'ws/nemotron-h-dense',
            entity: { spec: { checkpoint_model_name: 'NemotronHForCausalLM' } } as ModelEntity,
          })
        }
      >
        Dense Nemotron-H
      </button>
      <button
        type="button"
        onClick={() =>
          onValueChange({
            model: 'ws/mixtral',
            entity: {
              spec: {
                checkpoint_model_name: 'MixtralForCausalLM',
                moe_config: { num_experts: 8, num_experts_per_tok: 2, num_expert_layers: 32 },
              },
            } as ModelEntity,
          })
        }
      >
        Mixtral
      </button>
    </>
  ),
}));

interface HarnessProps {
  gpus: number;
  expertParallelSize?: number;
}

const Harness: FC<HarnessProps> = ({ gpus, expertParallelSize }) => {
  const form = useForm<CustomizationFormFields>({
    defaultValues: {
      ...FORM_DEFAULTS,
      backend: 'automodel',
      automodel: {
        ...FORM_DEFAULTS.automodel,
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
      <ModelSelectionSection />
      <output data-testid="ep">{ep ?? 'unset'}</output>
    </FormProvider>
  );
};

describe('ModelSelectionSection expert parallel size', () => {
  it('fills the largest valid size when a Nemotron-H MoE model is selected', async () => {
    render(<Harness gpus={4} />);
    await userEvent.click(screen.getByRole('button', { name: 'MoE' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('4');
  });

  it('keeps a valid size', async () => {
    render(<Harness gpus={4} expertParallelSize={2} />);
    await userEvent.click(screen.getByRole('button', { name: 'MoE' }));
    expect(screen.getByTestId('ep')).toHaveTextContent('2');
  });

  it.each([
    ['a dense model', 'Dense', 4],
    ['another MoE architecture', 'Mixtral', 4],
    ['a single GPU', 'MoE', 1],
  ])('leaves the size alone for %s', async (_, button, gpus) => {
    render(<Harness gpus={gpus} />);
    await userEvent.click(screen.getByRole('button', { name: button }));
    expect(screen.getByTestId('ep')).toHaveTextContent('unset');
  });

  it.each(['Dense', 'Dense Nemotron-H', 'Mixtral'])(
    'clears the Nemotron-H size when %s is selected next',
    async (button) => {
      render(<Harness gpus={4} />);
      await userEvent.click(screen.getByRole('button', { name: 'MoE' }));
      expect(screen.getByTestId('ep')).toHaveTextContent('4');
      await userEvent.click(screen.getByRole('button', { name: button }));
      expect(screen.getByTestId('ep')).toHaveTextContent('unset');
    }
  );
});
