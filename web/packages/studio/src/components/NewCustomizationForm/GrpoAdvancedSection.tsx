// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import {
  BatchingStrategy,
  PolicyBackend,
  RlGRPOTrainingTruncatedImportanceSamplingType,
} from '@nemo/sdk/generated/customizer/schema';
import { Divider, FormField, Stack, TextInput } from '@nvidia/foundations-react-core';
import { ControlledJsonInput } from '@studio/components/NewCustomizationForm/ControlledJsonInput';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { GRPO_SPEC_DEFAULTS, specSliderProps } from '@studio/util/forms/specDefaults';
import { useFormContext } from 'react-hook-form';

const POLICY_BACKEND_ITEMS = [
  { value: PolicyBackend.automodel, label: 'Automodel' },
  { value: PolicyBackend.dtensor, label: 'DTensor' },
];

const BATCHING_STRATEGY_ITEMS = [
  { value: BatchingStrategy.dynamic, label: 'Dynamic' },
  { value: BatchingStrategy.static, label: 'Static' },
  { value: BatchingStrategy.sequence_packing, label: 'Sequence Packing' },
];

const TIS_TYPE_ITEMS = [
  { value: RlGRPOTrainingTruncatedImportanceSamplingType.tis, label: 'TIS' },
  { value: RlGRPOTrainingTruncatedImportanceSamplingType.icepop, label: 'IcePop' },
  { value: RlGRPOTrainingTruncatedImportanceSamplingType['seq-mask-tis'], label: 'Seq-Mask-TIS' },
];

/**
 * GRPO knobs the backend leaves off by default. Their sliders omit `defaultValue`, so ↺
 * clears the field back to unset and the feature stays off rather than resetting to a
 * number the user never chose.
 */
export const GrpoAdvancedSection = () => {
  const { control, watch, setValue, formState } = useFormContext<CustomizationFormFields>();
  const disabled = formState.isSubmitting;
  const executionProfile = watch('rl.training.execution_profile');

  return (
    <>
      <Divider />
      <FormSection
        collapsible
        title="Policy Update"
        description="How the policy update is computed and bounded. Leave a field blank to keep the backend default."
      >
        <Stack gap="density-md">
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'rl.training.ref_policy_kl_penalty', control }}
            formFieldProps={{
              slotLabel: 'KL Penalty',
              slotInfo:
                'KL penalty coefficient against the reference policy. Higher values keep the model closer to the reference.',
            }}
            {...specSliderProps(GRPO_SPEC_DEFAULTS, 'ref_policy_kl_penalty')}
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.ratio_clip_min', control }}
            formFieldProps={{
              slotLabel: 'Clip Min',
              slotInfo: 'Lower bound for PPO-style importance ratio clipping.',
            }}
            {...specSliderProps(GRPO_SPEC_DEFAULTS, 'ratio_clip_min')}
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.ratio_clip_max', control }}
            formFieldProps={{
              slotLabel: 'Clip Max',
              slotInfo: 'Upper bound for PPO-style importance ratio clipping.',
            }}
            {...specSliderProps(GRPO_SPEC_DEFAULTS, 'ratio_clip_max')}
            min={0}
            max={2}
            step={0.01}
            disabled={disabled}
          />
          <ControlledSwitch
            useControllerProps={{ name: 'grpo.use_leave_one_out_baseline', control }}
            formFieldProps={{
              slotLabel: 'Leave-One-Out Baseline',
              slotInfo:
                'Compare each rollout against the mean of the other rollouts in its group rather than against the group mean including itself.',
            }}
            disabled={disabled}
          />
          <ControlledSwitch
            useControllerProps={{ name: 'grpo.use_importance_sampling_correction', control }}
            formFieldProps={{
              slotLabel: 'Importance Sampling Correction',
              slotInfo:
                'Reweight each token by how much more or less likely the training policy is to produce it than the policy that generated the rollout.',
            }}
            disabled={disabled}
          />
          <ControlledSwitch
            useControllerProps={{ name: 'grpo.use_on_policy_kl_approximation', control }}
            formFieldProps={{
              slotLabel: 'On-Policy KL Approximation',
              slotInfo:
                'Estimate the KL term against the policy that generated the rollouts rather than the one currently training.',
            }}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.ratio_clip_c', control }}
            formFieldProps={{
              slotLabel: 'Dual-Clip Bound',
              slotInfo:
                'Second safety limit on how far one update can move the model. Must be above 1. Unset disables dual clipping.',
            }}
            unsetPlaceholder="Off"
            min={1.01}
            max={20}
            step={0.01}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.advantage_clip_low', control }}
            formFieldProps={{
              slotLabel: 'Advantage Clip Low',
              slotInfo:
                'Lower bound applied to advantages after normalization. Must be below the high bound.',
            }}
            unsetPlaceholder="Off"
            min={-20}
            max={20}
            step={0.1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.advantage_clip_high', control }}
            formFieldProps={{
              slotLabel: 'Advantage Clip High',
              slotInfo: 'Upper bound applied to advantages after normalization.',
            }}
            unsetPlaceholder="Off"
            min={-20}
            max={20}
            step={0.1}
            disabled={disabled}
          />
          <ControlledSelect
            useControllerProps={{ name: 'grpo.truncated_importance_sampling_type', control }}
            formFieldProps={{
              slotLabel: 'Truncated Importance Sampling',
              slotInfo:
                'Bound the rollout-vs-training importance weights so a policy that has drifted cannot blow up an update. Leave unselected to disable.',
            }}
            items={TIS_TYPE_ITEMS}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.truncated_importance_sampling_ratio', control }}
            formFieldProps={{
              slotLabel: 'TIS Ratio (Upper)',
              slotInfo: 'Upper bound on the importance weight.',
            }}
            unsetPlaceholder="Off"
            min={0.01}
            max={20}
            step={0.01}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{
              name: 'grpo.truncated_importance_sampling_ratio_min',
              control,
            }}
            formFieldProps={{
              slotLabel: 'TIS Ratio (Lower)',
              slotInfo: 'Lower bound on the importance weight.',
            }}
            unsetPlaceholder="Off"
            min={0}
            max={20}
            step={0.01}
            disabled={disabled}
          />
        </Stack>
      </FormSection>

      <Divider />
      <FormSection
        collapsible
        title="Execution & Overrides"
        description="Which trainer and rollout engine run the job, and model-specific configuration."
      >
        <Stack gap="density-md">
          <FormField
            slotLabel="Execution Profile"
            slotInfo="Operator-configured GPU profile for the training step, e.g. h100. Leave blank to use the service default."
          >
            <TextInput
              value={executionProfile ?? ''}
              placeholder="Service default"
              disabled={disabled}
              onValueChange={(next: string) =>
                setValue('rl.training.execution_profile', next, { shouldValidate: true })
              }
            />
          </FormField>
          <ControlledSelect
            useControllerProps={{ name: 'grpo.policy_backend', control }}
            formFieldProps={{
              slotLabel: 'Policy Backend',
              slotInfo:
                'NeMo-RL policy worker that trains the model. Automodel is the only backend supporting LoRA, expert parallelism and automodel kwargs.',
            }}
            items={POLICY_BACKEND_ITEMS}
            disabled={disabled}
          />
          <ControlledSelect
            useControllerProps={{ name: 'grpo.batching_strategy', control }}
            formFieldProps={{
              slotLabel: 'Batching Strategy',
              slotInfo: 'How rollouts are grouped into training micro-batches.',
            }}
            items={BATCHING_STRATEGY_ITEMS}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.sequence_length_round', control }}
            formFieldProps={{
              slotLabel: 'Length Rounding',
              slotInfo: 'Round bucketed micro-batch sequence lengths up to a multiple of this.',
            }}
            {...specSliderProps(GRPO_SPEC_DEFAULTS, 'sequence_length_round')}
            min={1}
            max={1024}
            step={1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.train_mb_tokens', control }}
            formFieldProps={{
              slotLabel: 'Micro-Batch Tokens',
              slotInfo:
                'Token budget per training micro-batch, read by the dynamic and sequence_packing strategies.',
            }}
            unsetPlaceholder="Auto"
            min={1}
            max={131072}
            step={128}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.vllm_tensor_parallel_size', control }}
            formFieldProps={{
              slotLabel: 'vLLM Tensor Parallel',
              slotInfo:
                "Tensor parallel size for the vLLM rollout engine, independent of the policy's.",
            }}
            unsetPlaceholder="Auto"
            min={1}
            max={64}
            step={1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.vllm_gpu_memory_utilization', control }}
            formFieldProps={{
              slotLabel: 'vLLM GPU Memory',
              slotInfo: 'Fraction of each GPU vLLM reserves for weights plus KV cache.',
            }}
            {...specSliderProps(GRPO_SPEC_DEFAULTS, 'vllm_gpu_memory_utilization')}
            min={0.05}
            max={1}
            step={0.05}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'grpo.router_aux_loss_coef', control }}
            formFieldProps={{
              slotLabel: 'Router Aux Loss',
              slotInfo:
                'MoE router auxiliary-loss coefficient, applied as a top-level HuggingFace config override.',
            }}
            unsetPlaceholder="Off"
            min={0}
            max={1}
            step={0.001}
            disabled={disabled}
          />
          <ControlledJsonInput
            useControllerProps={{ name: 'grpo.hf_config_overrides', control }}
            formFieldProps={{
              slotLabel: 'HF Config Overrides (JSON)',
              slotInfo:
                'Passed to the training model as HuggingFace config kwargs and to vLLM as hf_overrides. Nested keys are supported, e.g. {"text_config": {"router_aux_loss_coef": 0.0}} for Qwen3.5.',
            }}
            placeholder='{"text_config": {"router_aux_loss_coef": 0.0}}'
            disabled={disabled}
          />
          <ControlledJsonInput
            useControllerProps={{ name: 'grpo.automodel_kwargs', control }}
            formFieldProps={{
              slotLabel: 'Automodel Kwargs (JSON)',
              slotInfo:
                'Requires the automodel policy backend. force_hf loads the stock HuggingFace modules; a backend block picks the Transformer-Engine or DeepEP MoE implementation. Unset lets Automodel auto-detect.',
            }}
            placeholder='{"force_hf": true}'
            disabled={disabled}
          />
        </Stack>
      </FormSection>
    </>
  );
};
