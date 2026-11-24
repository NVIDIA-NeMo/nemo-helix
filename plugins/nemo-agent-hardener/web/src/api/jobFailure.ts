// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type JobFailure = { message: string; remediation?: string };

type JobLike = { error_details?: { [key: string]: unknown } | null } | null | undefined;

const text = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined;

// The job's own failure, as the backend wrote it (jobs/errors.py `as_error_details`); undefined when it left none.
export function jobFailure(job: JobLike): JobFailure | undefined {
  const message = text(job?.error_details?.message);
  if (!message) return undefined;
  return { message, remediation: text(job?.error_details?.remediation) };
}
