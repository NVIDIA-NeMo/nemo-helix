// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { customizationGetAutomodelJobStatus } from '@nemo/sdk/generated/customizer/automodel-jobs';
import { useQuery } from '@tanstack/react-query';

export interface TrainingProgress {
  /** 0-100, from steps when the trainer reports them, else its own percentage. */
  percent: number | undefined;
  detail: string;
}

const finite = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

/** What a training job reports while it runs: steps, epoch and loss. Polls until the node stops running. */
export const useTrainingProgress = (
  workspace: string,
  job: string | undefined,
  nodeRunning: boolean,
  enabled: boolean
): TrainingProgress | undefined => {
  const { data } = useQuery({
    queryKey: ['distillation-training-progress', workspace, job],
    queryFn: () => customizationGetAutomodelJobStatus(workspace, job ?? ''),
    enabled: enabled && !!job,
    refetchInterval: nodeRunning ? JOB_POLLING_INTERVAL_MS : false,
  });
  const details = data?.status_details;
  if (!details) return undefined;

  const step = finite(details.step);
  const maxSteps = finite(details.max_steps);
  const epoch = finite(details.epoch);
  const loss = finite(details.train_loss);
  const percent =
    step !== undefined && maxSteps ? Math.min(100, (step / maxSteps) * 100) : finite(details.percentage_done);
  const parts = [
    typeof details.phase === 'string' ? details.phase : undefined,
    step !== undefined && maxSteps ? `step ${step}/${maxSteps}` : undefined,
    epoch !== undefined ? `epoch ${epoch}` : undefined,
    loss !== undefined ? `loss ${loss.toFixed(3)}` : undefined,
  ].filter((part): part is string => part !== undefined);
  return { percent, detail: parts.join(' · ') };
};
