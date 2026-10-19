// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobStepStatusResponse } from '@nemo/sdk/generated/customizer/schema';
import {
  failedGrpoCustomizationJob,
  failedGrpoJobSteps,
  failedVolcanoGrpoJobSteps,
  grpoCudaErrorMessage,
  grpoCustomizationJob,
} from '@studio/mocks/customizer/customization-jobs';
import type { CustomizationJob } from '@studio/util/customizationBackend';
import {
  formatPipelineStepName,
  resolveCustomizationFailure,
} from '@studio/util/customizationFailure';

const steps = (value: unknown) => value as HelixJobStepStatusResponse[];

describe('resolveCustomizationFailure', () => {
  it('returns undefined for a job that has not failed', () => {
    expect(
      resolveCustomizationFailure(grpoCustomizationJob, steps(failedGrpoJobSteps))
    ).toBeUndefined();
  });

  it('returns undefined when there is no job', () => {
    expect(resolveCustomizationFailure(undefined, steps(failedGrpoJobSteps))).toBeUndefined();
  });

  it('treats a cancelled job as not a failure', () => {
    const cancelled = { ...failedGrpoCustomizationJob, status: 'cancelled' } as CustomizationJob;
    expect(resolveCustomizationFailure(cancelled, steps(failedGrpoJobSteps))).toBeUndefined();
  });

  it('surfaces the mapped task message over the generic step and job text', () => {
    const failure = resolveCustomizationFailure(
      failedGrpoCustomizationJob,
      steps(failedGrpoJobSteps)
    );

    expect(failure?.message).toBe(grpoCudaErrorMessage);
    expect(failure?.isGeneric).toBe(false);
    expect(failure?.errorType).toBe('CudaError');
    expect(failure?.detail).toContain('CUDA out of memory');
    expect(failure?.errorStack).toContain('torch.OutOfMemoryError');
  });

  it('names the failing pipeline step', () => {
    const failure = resolveCustomizationFailure(
      failedGrpoCustomizationJob,
      steps(failedGrpoJobSteps)
    );

    expect(failure?.failingStepLabel).toBe('GRPO training');
  });

  it('falls through to a better candidate when the pod reconciler clobbers the task message', () => {
    // The reconciler replaces error_details wholesale, so the mapped cause can be lost from the
    // task while the step below still explains itself.
    const clobbered = steps([
      {
        ...failedGrpoJobSteps[1],
        error_details: { message: 'Training ran out of GPU memory on node 0.' },
        tasks: [
          {
            ...failedGrpoJobSteps[1].tasks[0],
            error_details: { message: 'Pod grpo-training-abc123 is in error state' },
          },
        ],
      },
    ]);

    const failure = resolveCustomizationFailure(failedGrpoCustomizationJob, clobbered);

    expect(failure?.message).toBe('Training ran out of GPU memory on node 0.');
    expect(failure?.isGeneric).toBe(false);
  });

  it('prefers an explanatory task over a boilerplate sibling in the same step', () => {
    const twoTasks = steps([
      {
        ...failedGrpoJobSteps[1],
        tasks: [
          {
            ...failedGrpoJobSteps[1].tasks[0],
            id: 'task-boilerplate',
            error_details: { message: 'Pod grpo-training-xyz789 is in error state' },
            error_stack: '',
          },
          failedGrpoJobSteps[1].tasks[0],
        ],
      },
    ]);

    const failure = resolveCustomizationFailure(failedGrpoCustomizationJob, twoTasks);

    expect(failure?.message).toBe(grpoCudaErrorMessage);
    expect(failure?.errorType).toBe('CudaError');
  });

  it('takes the mapped message from a task whose status has not caught up yet', () => {
    // The step is errored and one task has flipped with no detail, while the task that actually
    // reported the cause is still marked active.
    const lagging = steps([
      {
        ...failedGrpoJobSteps[1],
        tasks: [
          {
            ...failedGrpoJobSteps[1].tasks[0],
            id: 'task-silent',
            status: 'error',
            error_details: {},
            error_stack: '',
          },
          { ...failedGrpoJobSteps[1].tasks[0], status: 'active' },
        ],
      },
    ]);

    const failure = resolveCustomizationFailure(failedGrpoCustomizationJob, lagging);

    expect(failure?.message).toBe(grpoCudaErrorMessage);
    expect(failure?.errorType).toBe('CudaError');
    expect(failure?.isGeneric).toBe(false);
  });

  it('degrades to status_details.message on the multi-node Volcano path', () => {
    const failure = resolveCustomizationFailure(
      failedGrpoCustomizationJob,
      steps(failedVolcanoGrpoJobSteps)
    );

    // Volcano writes no error_details anywhere, so every candidate is generic.
    expect(failure?.message).toBe('Job failed');
    expect(failure?.isGeneric).toBe(true);
    expect(failure?.failingStepLabel).toBe('GRPO training');
    expect(failure?.errorType).toBeUndefined();
  });

  it('falls back to the job error_details when the status tree has not loaded yet', () => {
    const failure = resolveCustomizationFailure(failedGrpoCustomizationJob, []);

    expect(failure?.message).toBe('One or more tasks are in error state');
    expect(failure?.isGeneric).toBe(true);
    expect(failure?.failingStepLabel).toBeUndefined();
  });

  it('always produces a message, even with nothing to go on', () => {
    const bare = {
      ...failedGrpoCustomizationJob,
      status_details: {},
      error_details: {},
    } as CustomizationJob;

    const failure = resolveCustomizationFailure(bare, []);

    expect(failure?.message).toBe('The customization job failed.');
    expect(failure?.isGeneric).toBe(true);
  });

  it('ignores blank and non-string error text', () => {
    const noisy = {
      ...failedGrpoCustomizationJob,
      status_details: { message: '   ' },
      error_details: { message: 42 },
    } as CustomizationJob;

    expect(resolveCustomizationFailure(noisy, [])?.message).toBe('The customization job failed.');
  });
});

describe('formatPipelineStepName', () => {
  it.each([
    ['grpo-training', 'GRPO training'],
    ['dpo-training', 'DPO training'],
    ['training', 'training'],
    ['model-dataset-environment-download', 'model, dataset & environment download'],
    ['model-and-dataset-download', 'model & dataset download'],
    ['model-upload', 'model upload'],
    ['model-entity-creation', 'model entity creation'],
  ])('maps %s to %s', (step, expected) => {
    expect(formatPipelineStepName(step)).toBe(expected);
  });

  it('de-kebabs an unknown step rather than dropping it', () => {
    expect(formatPipelineStepName('reward-model-warmup')).toBe('reward model warmup');
  });
});
