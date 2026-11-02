// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  TrainingMemoryComponent,
  TrainingMemoryStatus,
  type TrainingMemoryEstimate,
} from '@nemo/sdk/generated/platform/schema';
import { Banner, Stack, Text } from '@nvidia/foundations-react-core';
import {
  toTrainingMemoryEstimateRequest,
  useTrainingMemoryEstimate,
} from '@studio/hooks/useTrainingMemoryEstimate';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import type { FC } from 'react';
import { useFormContext } from 'react-hook-form';

export interface TrainingMemoryWarningProps {
  estimate: TrainingMemoryEstimate | undefined;
  /** Whether the run already trains a LoRA adapter, so the advice does not suggest it again. */
  usesLora: boolean;
}

const formatGb = (gb: number) => `${gb.toFixed(1)} GB`;

/** The changes that shrink whichever component dominates the estimate. */
const suggestionsFor = (estimate: TrainingMemoryEstimate, usesLora: boolean): string[] => {
  if (estimate.limiting_factor === TrainingMemoryComponent.activations) {
    return ['lower the micro batch size', 'lower the max sequence length'];
  }
  // Framework overhead is sized from the model states, so the same changes shrink both.
  return [
    ...(usesLora ? [] : ['use LoRA instead of all-weights fine-tuning']),
    'raise the tensor parallel size',
    'add GPUs',
  ];
};

/**
 * Advisory banner for a configuration that is near or over GPU memory. It renders nothing
 * when the estimate fits or is unavailable, and never affects whether the form can submit.
 */
export const TrainingMemoryWarning: FC<TrainingMemoryWarningProps> = ({ estimate, usesLora }) => {
  if (!estimate || estimate.status === TrainingMemoryStatus.ok) return null;

  const exceeds = estimate.status === TrainingMemoryStatus.exceeds_capacity;
  const capacity = estimate.gpu_memory_assumed
    ? `the ${formatGb(estimate.gpu_memory_gb)} assumed per GPU`
    : `${formatGb(estimate.gpu_memory_gb)} per GPU`;
  const comparison = exceeds
    ? `more than ${capacity}`
    : `${Math.round(estimate.utilization * 100)}% of ${capacity}`;
  const suggestions = suggestionsFor(estimate, usesLora);

  return (
    <Banner kind="inline" status="warning">
      <Stack gap="density-xs">
        <Text kind="body/bold/sm">
          {exceeds ? 'Likely to run out of GPU memory' : 'Close to the GPU memory limit'}
        </Text>
        <Text kind="body/regular/sm">
          This job is estimated to need {formatGb(estimate.total_gb)} per GPU, {comparison}.
          {exceeds
            ? ''
            : ' Runs this close can fail partway through, when one long batch needs more memory than the rest.'}
        </Text>
        <Text kind="body/regular/sm">
          To use less memory, {suggestions.join(', or ')}. This is an estimate — you can still start
          the job.
        </Text>
      </Stack>
    </Banner>
  );
};

/** Estimates the automodel run in the surrounding form and warns when it may not fit. */
export const AutomodelTrainingMemoryWarning: FC = () => {
  const { watch } = useFormContext<CustomizationFormFields>();
  const automodel = watch('automodel');
  const estimate = useTrainingMemoryEstimate(
    automodel.model,
    toTrainingMemoryEstimateRequest(automodel)
  );
  const finetuningType = automodel.training?.finetuning_type;

  return (
    <TrainingMemoryWarning
      estimate={estimate}
      usesLora={finetuningType === 'lora' || finetuningType === 'lora_merged'}
    />
  );
};
