// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobStepStatusResponse } from '@nemo/sdk/generated/customizer/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import type { CustomizationJob } from '@studio/util/customizationBackend';
import { formatPipelineStepName } from '@studio/util/customizationFailure';

/** `training`, `grpo-training`, `dpo-training`: the step names the compilers emit for the training phase. */
const TRAINING_STEP_PATTERN = /(^|-)training$/;
/** The step that uploads the finished model's files; the RL, automodel and unsloth compilers all name it this. */
const MODEL_UPLOAD_STEP = 'model-upload';

/** Training has finished and the job is still running because the model's files are being transferred. */
export interface CustomizationTransfer {
  /** Display form of the transfer step, e.g. `model upload`. */
  stepLabel: string;
  uploadedFiles?: number;
  totalFiles?: number;
  totalBytes?: number;
}

const readNumber = (
  bag: Record<string, unknown> | null | undefined,
  key: string
): number | undefined => {
  const value = bag?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
};

/**
 * Resolves whether an active job is past training and now moving the model's files. The step
 * counters read 100% and the job still says Active during this phase, so without this the page
 * looks finished but is not. Returns `undefined` unless training is complete and the upload step
 * is running.
 *
 * File counts come from the upload task and only change as each whole file finishes, so they say
 * nothing about whether a large file is making progress.
 */
export const resolveCustomizationTransfer = (
  job: CustomizationJob | undefined,
  steps: HelixJobStepStatusResponse[] = []
): CustomizationTransfer | undefined => {
  if (job?.status !== HelixJobStatus.active) {
    return undefined;
  }

  const trainingComplete = steps.some(
    (step) => TRAINING_STEP_PATTERN.test(step.name) && step.status === HelixJobStatus.completed
  );
  const transferStep = steps.find(
    (step) => step.name === MODEL_UPLOAD_STEP && step.status === HelixJobStatus.active
  );
  if (!trainingComplete || !transferStep) {
    return undefined;
  }

  const taskDetails =
    transferStep.tasks?.find((task) => task.status === HelixJobStatus.active)?.status_details ??
    transferStep.status_details;
  const stepDetails = transferStep.status_details;

  return {
    stepLabel: formatPipelineStepName(transferStep.name),
    uploadedFiles:
      readNumber(taskDetails, 'uploaded_files') ?? readNumber(stepDetails, 'uploaded_files'),
    totalFiles: readNumber(taskDetails, 'total_files') ?? readNumber(stepDetails, 'total_files'),
    totalBytes: readNumber(taskDetails, 'total_size') ?? readNumber(stepDetails, 'total_size'),
  };
};
