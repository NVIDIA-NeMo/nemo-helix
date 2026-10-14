// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobStepStatusResponse } from '@nemo/sdk/generated/customizer/schema';
import { customizationJob1 } from '@studio/mocks/customizer/customization-jobs';
import type { CustomizationJob } from '@studio/util/customizationBackend';
import { resolveCustomizationTransfer } from '@studio/util/customizationTransfer';

const job = (status: string) => ({ ...customizationJob1, status }) as CustomizationJob;

const step = (name: string, status: string, tasks: unknown[] = []) =>
  ({
    id: `step-${name}`,
    name,
    status,
    status_details: {},
    error_details: {},
    tasks,
    created_at: '2025-06-25T21:41:12.100000',
    updated_at: '2025-06-25T21:42:02.100000',
  }) as unknown as HelixJobStepStatusResponse;

const uploadTask = (status_details: Record<string, unknown>) => ({
  id: 'task-upload',
  name: 'task-upload',
  status: 'active',
  status_details,
  error_details: {},
  created_at: '2025-06-25T21:42:02.100000',
  updated_at: '2025-06-25T21:42:02.100000',
});

describe('resolveCustomizationTransfer', () => {
  it('reports the transfer once training is complete and the upload is running, with its file counts', () => {
    const transfer = resolveCustomizationTransfer(job('active'), [
      step('model-and-dataset-download', 'completed'),
      step('training', 'completed'),
      step('model-upload', 'active', [
        uploadTask({ uploaded_files: 4, total_files: 9, total_size: 63174389156 }),
      ]),
    ]);

    expect(transfer).toEqual({
      stepLabel: 'model upload',
      uploadedFiles: 4,
      totalFiles: 9,
      totalBytes: 63174389156,
    });
  });

  it('still reports the transfer, without counts, before the upload task has reported any', () => {
    const transfer = resolveCustomizationTransfer(job('active'), [
      step('training', 'completed'),
      step('model-upload', 'active'),
    ]);

    expect(transfer).toEqual({
      stepLabel: 'model upload',
      uploadedFiles: undefined,
      totalFiles: undefined,
      totalBytes: undefined,
    });
  });

  it('recognises the RL pipeline step names', () => {
    expect(
      resolveCustomizationTransfer(job('active'), [
        step('grpo-training', 'completed'),
        step('model-upload', 'active'),
      ])
    ).toBeDefined();
  });

  it.each([
    ['training is still running', [step('training', 'active'), step('model-upload', 'pending')]],
    [
      'the upload has not started',
      [step('training', 'completed'), step('model-upload', 'pending')],
    ],
    [
      'the upload step has finished',
      [step('training', 'completed'), step('model-upload', 'completed')],
    ],
    ['no steps are loaded yet', []],
  ])('returns nothing while %s', (_label, steps) => {
    expect(resolveCustomizationTransfer(job('active'), steps)).toBeUndefined();
  });

  it.each(['completed', 'error', 'cancelled'])('returns nothing for a %s job', (status) => {
    expect(
      resolveCustomizationTransfer(job(status), [
        step('training', 'completed'),
        step('model-upload', 'active'),
      ])
    ).toBeUndefined();
  });

  it('returns nothing when there is no job', () => {
    expect(
      resolveCustomizationTransfer(undefined, [
        step('training', 'completed'),
        step('model-upload', 'active'),
      ])
    ).toBeUndefined();
  });
});
