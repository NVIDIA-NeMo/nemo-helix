// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ComputeResourcesSection } from '@studio/components/NewCustomizationForm/ComputeResourcesSection';
import { render, screen } from '@studio/tests/util/render';
import { type CustomizationFormFields, FORM_DEFAULTS } from '@studio/util/forms/customization';
import type { FC } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

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
  return (
    <FormProvider {...form}>
      <ComputeResourcesSection />
    </FormProvider>
  );
};

describe('ComputeResourcesSection expert parallel size', () => {
  it('shows the field and a warning for an invalid size on a MoE model', () => {
    modelEntity.current = {
      spec: { moe_config: { num_experts: 128, num_experts_per_tok: 6, num_expert_layers: 23 } },
    } as ModelEntity;
    render(<Harness expertParallelSize={3} />);
    expect(screen.getByText('Expert Parallel Size')).toBeVisible();
    expect(screen.getByText(/without an expert parallel size of 2, 4/)).toBeInTheDocument();
  });

  it('keeps the field in Advanced Parallelism for a dense model', () => {
    modelEntity.current = { spec: {} } as ModelEntity;
    render(<Harness />);
    expect(screen.queryByText('Expert Parallel Size')).not.toBeVisible();
  });
});
