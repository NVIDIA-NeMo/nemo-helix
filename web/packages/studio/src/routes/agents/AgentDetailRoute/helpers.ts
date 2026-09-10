// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Abbreviates a commit SHA; any other revision (a branch, a tag) is shown whole. */
export const shortRevision = (revision: string): string =>
  /^[0-9a-f]{40}$/.test(revision) ? revision.slice(0, 7) : revision;

export function deploymentStatusColor(status?: string): 'green' | 'red' | 'yellow' | undefined {
  if (status === 'running') return 'green';
  if (status === 'error' || status === 'failed') return 'red';
  if (status === 'pending' || status === 'starting' || status === 'deleting') return 'yellow';
  return undefined;
}
