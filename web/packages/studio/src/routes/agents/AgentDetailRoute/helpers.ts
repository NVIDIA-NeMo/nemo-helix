// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const COMMIT_SHA = /^[0-9a-f]{40}$/;

/** Abbreviates a commit SHA; any other revision (a branch, a tag) is shown whole. */
export const shortRevision = (revision: string): string =>
  COMMIT_SHA.test(revision) ? revision.slice(0, 7) : revision;

// Matches the Runtime select in CreateDeploymentModal, so one deployment is not
// named two things across the UI.
const DEPLOYMENT_MODE_LABELS: Record<string, string> = {
  subprocess: 'Subprocess',
  docker: 'Docker',
  k8s: 'Kubernetes',
};

/** The backend defaults this field to `subprocess`, so an absent mode is that. */
export const deploymentModeLabel = (mode?: string): string =>
  DEPLOYMENT_MODE_LABELS[mode ?? 'subprocess'] ?? (mode as string);

export function deploymentStatusColor(status?: string): 'green' | 'red' | 'yellow' | undefined {
  if (status === 'running') return 'green';
  if (status === 'error' || status === 'failed') return 'red';
  if (status === 'pending' || status === 'starting' || status === 'deleting') return 'yellow';
  return undefined;
}
