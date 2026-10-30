// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelEntity } from '@nemo/common/src/api/models/useModelEntity';
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
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import {
  expertParallelError,
  expertParallelExpertCount,
  validExpertParallelSizes,
} from '@studio/util/forms/expertParallel';
import {
  AUTOMODEL_SPEC_DEFAULTS,
  DPO_SPEC_DEFAULTS,
  specSliderProps,
} from '@studio/util/forms/specDefaults';
import { useEffect, useRef } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const EXPERT_PARALLEL_FIELD = 'automodel.parallelism.expert_parallel_size';

const AutomodelParallelism = ({ disabled }: { disabled: boolean }) => {
  const { control, setValue } = useFormContext<CustomizationFormFields>();
  const modelUrn = useWatch({ control, name: 'automodel.model' });
  const parallelism = useWatch({ control, name: 'automodel.parallelism' }) ?? {};
  const numExperts = expertParallelExpertCount(useModelEntity(modelUrn));
  const validSizes = validExpertParallelSizes(numExperts, parallelism);
  const expertParallelSize = parallelism.expert_parallel_size;

  // The value this effect last filled in. Anything else came from the user or a template,
  // and is left for validation to flag rather than overwritten.
  const autoFilledSize = useRef<number | undefined>(undefined);
  const defaultSize = validSizes?.at(-1);
  useEffect(() => {
    const isAutoFilled =
      expertParallelSize === undefined || expertParallelSize === autoFilledSize.current;
    if (!isAutoFilled || expertParallelSize === defaultSize) return;
    if (defaultSize === undefined && autoFilledSize.current === undefined) return;
    autoFilledSize.current = defaultSize;
    setValue(EXPERT_PARALLEL_FIELD, defaultSize, { shouldValidate: true });
  }, [defaultSize, expertParallelSize, setValue]);

  const expertParallelField = (
    <ControlledSliderWithTextInput
      useControllerProps={{ name: EXPERT_PARALLEL_FIELD, control }}
      formFieldProps={{
        slotLabel: 'Expert Parallel Size',
        slotInfo: validSizes
          ? `This model routes each token to a few of its ${numExperts} experts. Training it on more than one GPU shards the experts across GPUs, so each holds only its share — the value must divide both the expert count and the data-parallel GPUs.`
          : 'Expert parallel size for MoE models. Must divide data_parallel_size × context_parallel_size, and tensor parallel size must be 1 when this is above 1.',
        slotError: expertParallelError(numExperts, parallelism),
      }}
      {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_expert_parallel_size')}
      min={1}
      max={Math.max(8, ...(validSizes ?? []))}
      step={1}
      disabled={disabled}
    />
  );

  return (
    <Stack gap="density-lg">
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'automodel.parallelism.num_nodes', control }}
        formFieldProps={{ slotLabel: 'Nodes', slotInfo: 'Number of nodes to train on.' }}
        {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_num_nodes')}
        min={1}
        max={16}
        step={1}
        disabled={disabled}
      />
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'automodel.parallelism.num_gpus_per_node', control }}
        formFieldProps={{
          slotLabel: 'GPUs per Node',
          slotInfo: 'GPUs used on each node. Total GPUs is nodes multiplied by this.',
        }}
        {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'parallelism_num_gpus_per_node')}
        min={1}
        max={8}
        step={1}
        disabled={disabled}
      />
      {validSizes && expertParallelField}
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
              {!validSizes && expertParallelField}
              <ControlledSwitch
                useControllerProps={{ name: 'automodel.parallelism.sequence_parallel', control }}
                formFieldProps={{
                  slotLabel: 'Sequence Parallel',
                  slotInfo:
                    'Shards layer-norm and dropout activations along the sequence axis to save memory.',
                  labelPosition: 'left',
                }}
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
        formFieldProps={{
          slotLabel: 'Nodes',
          slotInfo: 'Number of nodes (>1 → multi-node Ray cluster).',
        }}
        {...specSliderProps(DPO_SPEC_DEFAULTS, 'parallelism_num_nodes')}
        min={1}
        max={16}
        step={1}
        disabled={disabled}
      />
      <ControlledSliderWithTextInput
        useControllerProps={{ name: 'rl.training.parallelism.num_gpus_per_node', control }}
        formFieldProps={{
          slotLabel: 'GPUs per Node',
          slotInfo: 'GPUs used on each node. Total GPUs is nodes multiplied by this.',
        }}
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
