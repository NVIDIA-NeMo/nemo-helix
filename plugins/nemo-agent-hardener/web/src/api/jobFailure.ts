// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CJobTerminalStatuses } from '@nemo/common';

export type JobFailure = { message: string; remediation?: string };

type JobLike =
  | { status?: string | null; error_details?: { [key: string]: unknown } | null }
  | null
  | undefined;

const text = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined;

// The job's own failure, as the backend wrote it (jobs/errors.py `as_error_details`); undefined when it left none.
export function jobFailure(job: JobLike): JobFailure | undefined {
  const message = text(job?.error_details?.message);
  if (!message) return undefined;
  return { message, remediation: text(job?.error_details?.remediation) };
}

// Why a job that has ended did not complete; undefined while it runs or once it completed.
export function endedJobFailure(job: JobLike): JobFailure | undefined {
  const status = job?.status;
  if (!status || status === 'completed' || !(CJobTerminalStatuses as string[]).includes(status))
    return undefined;
  if (status === 'cancelled') return { message: 'The job was cancelled.' };
  return (
    jobFailure(job) ?? { message: 'The job ended without completing and left no error message.' }
  );
}
