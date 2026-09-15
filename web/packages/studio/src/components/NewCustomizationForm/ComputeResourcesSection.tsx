// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { ControlledJsonInput } from '@studio/components/NewCustomizationForm/ControlledJsonInput';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import {
  AUTOMODEL_SPEC_DEFAULTS,
  DPO_SPEC_DEFAULTS,
  specSliderProps,
} from '@studio/util/forms/specDefaults';
import { useFormContext } from 'react-hook-form';

const AutomodelParallelism = ({ disabled }: { disabled: boolean }) => {
  const { control } = useFormContext<CustomizationFormFields>();
  return (
    <Stack gap="density-lg">
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'automodel.parallelism.num_nodes', control }}
        formFieldProps={{ slotLabel: 'Nodes' }}
        {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_num_nodes')}
        min={1}
        max={16}
        step={1}
        disabled={disabled}
      />
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'automodel.parallelism.num_gpus_per_node', control }}
        formFieldProps={{ slotLabel: 'GPUs per Node' }}
        {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_num_gpus_per_node')}
        min={1}
        max={8}
        step={1}
        disabled={disabled}
      />
      <AccordionRoot multiple>
        <AccordionItem value="advanced-parallelism" className="border-b-0">
          <AccordionTrigger>
            <Text kind="label/bold/md">Advanced Parallelism</Text>
          </AccordionTrigger>
          <AccordionContent>
            <Stack gap="density-md" className="pt-density-md">
              <ControlledSliderWithTextInput
                useControllerProps={{ name: 'automodel.parallelism.tensor_parallel_size', control }}
                formFieldProps={{
                  slotLabel: 'Tensor Parallel Size',
                  slotInfo: 'Splits each layer across GPUs.',
                }}
                {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_tensor_parallel_size')}
                min={1}
                max={8}
                step={1}
                disabled={disabled}
              />
              <ControlledSliderWithTextInput
                useControllerProps={{
                  name: 'automodel.parallelism.pipeline_parallel_size',
                  control,
                }}
                formFieldProps={{
                  slotLabel: 'Pipeline Parallel Size',
                  slotInfo: 'Splits layers into stages across GPUs.',
                }}
                {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_pipeline_parallel_size')}
                min={1}
                max={8}
                step={1}
                disabled={disabled}
              />
              <ControlledSliderWithTextInput
                useControllerProps={{
                  name: 'automodel.parallelism.context_parallel_size',
                  control,
                }}
                formFieldProps={{
                  slotLabel: 'Context Parallel Size',
                  slotInfo:
                    'Splits the sequence dimension across GPUs, for long-sequence training.',
                }}
                {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_context_parallel_size')}
                min={1}
                max={8}
                step={1}
                disabled={disabled}
              />
              <ControlledSliderWithTextInput
                useControllerProps={{ name: 'automodel.parallelism.expert_parallel_size', control }}
                formFieldProps={{
                  slotLabel: 'Expert Parallel Size',
                  slotInfo:
                    'Expert parallel size for MoE models. Must divide data_parallel_size × context_parallel_size, and tensor parallel size must be 1 when this is above 1.',
                }}
                {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_expert_parallel_size')}
                min={1}
                max={8}
                step={1}
                disabled={disabled}
              />
              <ControlledSwitch
                useControllerProps={{ name: 'automodel.parallelism.sequence_parallel', control }}
                formFieldProps={{ slotLabel: 'Sequence Parallel', labelPosition: 'left' }}
                disabled={disabled}
              />
            </Stack>
          </AccordionContent>
        </AccordionItem>
      </AccordionRoot>
    </Stack>
  );
};

const RlParallelism = ({ disabled }: { disabled: boolean }) => {
  const { control } = useFormContext<CustomizationFormFields>();
  return (
    <Stack gap="density-lg">
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'rl.training.parallelism.num_nodes', control }}
        formFieldProps={{ slotLabel: 'Nodes' }}
        {...specSliderProps(DPO_SPEC_DEFAULTS, 'parallelism_num_nodes')}
        min={1}
        max={16}
        step={1}
        disabled={disabled}
      />
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'rl.training.parallelism.num_gpus_per_node', control }}
        formFieldProps={{ slotLabel: 'GPUs per Node' }}
        {...specSliderProps(DPO_SPEC_DEFAULTS, 'parallelism_num_gpus_per_node')}
        min={1}
        max={8}
        step={1}
        disabled={disabled}
      />
      <Text kind="label/bold/md">Parallelism</Text>
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'rl.training.parallelism.tensor_parallel_size', control }}
        formFieldProps={{
          slotLabel: 'Tensor Parallel Size',
          slotInfo: 'Splits each layer across GPUs. NeMo RL key: tensor_parallel_size.',
        }}
        {...specSliderProps(DPO_SPEC_DEFAULTS, 'parallelism_tensor_parallel_size')}
        min={1}
        max={8}
        step={1}
        disabled={disabled}
      />
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'rl.training.parallelism.pipeline_parallel_size', control }}
        formFieldProps={{
          slotLabel: 'Pipeline Parallel Size',
          slotInfo: 'Splits layers into stages across GPUs. NeMo RL key: pipeline_parallel_size.',
        }}
        {...specSliderProps(DPO_SPEC_DEFAULTS, 'parallelism_pipeline_parallel_size')}
        min={1}
        max={8}
        step={1}
        disabled={disabled}
      />
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'rl.training.parallelism.context_parallel_size', control }}
        formFieldProps={{
          slotLabel: 'Context Parallel Size',
          slotInfo:
            'Splits the sequence dimension across GPUs — how long-sequence GRPO becomes feasible at all. NeMo RL key: context_parallel_size.',
        }}
        {...specSliderProps(DPO_SPEC_DEFAULTS, 'parallelism_context_parallel_size')}
        min={1}
        max={8}
        step={1}
        disabled={disabled}
      />
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'rl.training.parallelism.expert_parallel_size', control }}
        formFieldProps={{
          slotLabel: 'Expert Parallel Size',
          slotInfo:
            'Expert parallel size for MoE models. GRPO only — a value above 1 selects the DTensor v2 backend. NeMo RL key: expert_parallel_size.',
        }}
        {...specSliderProps(DPO_SPEC_DEFAULTS, 'parallelism_expert_parallel_size')}
        min={1}
        max={8}
        step={1}
        disabled={disabled}
      />
      <ControlledSwitch
        useControllerProps={{ name: 'rl.training.parallelism.sequence_parallel', control }}
        formFieldProps={{
          slotLabel: 'Sequence Parallel',
          labelPosition: 'left',
          slotInfo:
            'Shards layer-norm and dropout activations along the sequence axis. NeMo RL key: sequence_parallel.',
        }}
        disabled={disabled}
      />
    </Stack>
  );
};

const UnslothHardware = ({ disabled }: { disabled: boolean }) => {
  const { control } = useFormContext<CustomizationFormFields>();
  return (
    <Stack gap="density-lg">
      <ControlledTextInput
        useControllerProps={{ name: 'unsloth.hardware.gpus', control }}
        label="GPU Indices"
        placeholder="0  or  0,1"
        disabled={disabled}
      />
      <ControlledJsonInput
        useControllerProps={{ name: 'unsloth.deployment_config', control }}
        formFieldProps={{ slotLabel: 'Deployment Config (name or JSON)' }}
        placeholder='"my-config"  or  { "gpu": 1 }'
        disabled={disabled}
      />
    </Stack>
  );
};

export const ComputeResourcesSection = () => {
  const { watch, formState } = useFormContext<CustomizationFormFields>();
  const backend = watch('backend');
  const disabled = formState.isSubmitting;

  return (
    <FormSection title="Compute Resources">
      {backend === 'automodel' ? (
        <AutomodelParallelism disabled={disabled} />
      ) : backend === 'rl' ? (
        <RlParallelism disabled={disabled} />
      ) : (
        <UnslothHardware disabled={disabled} />
      )}
    </FormSection>
  );
};
