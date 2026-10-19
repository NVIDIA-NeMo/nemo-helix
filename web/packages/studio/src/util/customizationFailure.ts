// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  HelixJobStepStatusResponse,
  HelixJobTaskStatusResponse,
} from '@nemo/sdk/generated/customizer/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import type { CustomizationJob } from '@studio/util/customizationBackend';

/**
 * A customization job's failure, assembled from the step/task status tree — the useful text lives
 * at `steps[].tasks[].error_details`, not on the job record itself.
 */
export interface CustomizationFailure {
  /** Best available user-facing text. Never empty. */
  message: string;
  /** Mapped exception class, e.g. `CudaError`. Absent for infrastructure failures. */
  errorType?: string;
  /** Raw exception line, `f"{type(exc).__name__}: {exc}"`. */
  detail?: string;
  /** Container terminated message, or the tail of its logs. Task level only. */
  errorStack?: string;
  /** Display form of the failing pipeline step's raw name, e.g. `GRPO training`. */
  failingStepLabel?: string;
  /** True when only generic infrastructure text was available — no mapped cause was found. */
  isGeneric: boolean;
}

const FALLBACK_MESSAGE = 'The customization job failed.';

/** Generic infrastructure text (pod-died boilerplate) the resolver should keep looking past. */
const GENERIC_MESSAGE_PATTERNS: readonly RegExp[] = [
  /^one or more tasks are in error state/i,
  /^job has errored pods/i,
  /^pod .+ is in error state$/i,
  /^job failed\.?$/i,
  /^job not found$/i,
  /^job exited with non-zero code \d+/i,
  /^job exited with code \d+/i,
];

const isGenericMessage = (message: string): boolean =>
  GENERIC_MESSAGE_PATTERNS.some((pattern) => pattern.test(message.trim()));

/**
 * `error_details` and `status_details` are `Record<string, unknown>` on the wire — the OpenAPI
 * schema declares them as free-form objects, so every read needs narrowing.
 */
const readString = (
  bag: Record<string, unknown> | null | undefined,
  key: string
): string | undefined => {
  const value = bag?.[key];
  return typeof value === 'string' && value.trim().length > 0 ? value : undefined;
};

/** Pipeline steps emitted by the RL, automodel and unsloth compilers. */
const STEP_LABELS: Record<string, string> = {
  'model-dataset-environment-download': 'model, dataset & environment download',
  'model-and-dataset-download': 'model & dataset download',
  'grpo-training': 'GRPO training',
  'dpo-training': 'DPO training',
  training: 'training',
  'model-upload': 'model upload',
  'model-entity-creation': 'model entity creation',
};

/** Known steps read as prose; anything new degrades to its de-kebabed name rather than vanishing. */
export const formatPipelineStepName = (step: string): string =>
  STEP_LABELS[step] ?? step.replace(/[-_]+/g, ' ').trim();

const isErrored = (status: HelixJobStatus | undefined): boolean => status === HelixJobStatus.error;

/** Prefers a task whose message actually explains something over a boilerplate sibling. */
const findFailingTask = (
  step: HelixJobStepStatusResponse | undefined
): HelixJobTaskStatusResponse | undefined => {
  const tasks = step?.tasks ?? [];
  const errored = tasks.filter((task) => isErrored(task.status));
  const explains = (task: HelixJobTaskStatusResponse) => {
    const message = readString(task.error_details, 'message');
    return message !== undefined && !isGenericMessage(message);
  };
  const hasMessage = (task: HelixJobTaskStatusResponse) =>
    readString(task.error_details, 'message') !== undefined;

  return (
    errored.find(explains) ??
    errored.find(hasMessage) ??
    // A step can be errored before its tasks' statuses catch up, so check active tasks too.
    tasks.find(explains) ??
    tasks.find(hasMessage) ??
    errored[0]
  );
};

/**
 * Resolves what to tell the user about a failed customization job. Returns `undefined` unless the
 * job is in `error`. Message candidates are tried most- to least-specific; the first non-generic one wins.
 */
export const resolveCustomizationFailure = (
  job: CustomizationJob | undefined,
  steps: HelixJobStepStatusResponse[] = []
): CustomizationFailure | undefined => {
  if (!job || !isErrored(job.status)) {
    return undefined;
  }

  const failingStep = steps.find((step) => isErrored(step.status));
  const failingTask = findFailingTask(failingStep);

  const candidates = [
    readString(failingTask?.error_details, 'message'),
    readString(failingStep?.error_details, 'message'),
    readString(failingStep?.status_details, 'message'),
    readString(job.error_details, 'message'),
    readString(job.status_details, 'message'),
  ].filter((candidate): candidate is string => candidate !== undefined);

  const specific = candidates.find((candidate) => !isGenericMessage(candidate));
  const message = specific ?? candidates[0] ?? FALLBACK_MESSAGE;

  return {
    message,
    errorType: readString(failingTask?.error_details, 'type'),
    detail: readString(failingTask?.error_details, 'detail'),
    errorStack: failingTask?.error_stack?.trim() ? failingTask.error_stack : undefined,
    failingStepLabel: failingStep ? formatPipelineStepName(failingStep.name) : undefined,
    isGeneric: specific === undefined,
  };
};
