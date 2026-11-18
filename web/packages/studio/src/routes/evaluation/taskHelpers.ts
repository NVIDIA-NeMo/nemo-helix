// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Task } from '@nemo/sdk/generated/evals/schema';
import type { TaskRefParts } from '@studio/routes/evaluation/tasks/types';

export const RUNNER_LABELS = {
  harbor: 'Harbor',
  evaluator: 'Inline',
} as const satisfies Record<Task['spec']['kind'], string>;

export const RUNNER_OPTIONS = Object.entries(RUNNER_LABELS).map(([value, label]) => ({
  value,
  label,
}));

/** A one-line gist of what the task asks for: an inline task's intent, a Harbor instruction. */
export const taskSummary = ({ spec }: Pick<Task, 'spec'>): string | undefined =>
  spec.kind === 'harbor' ? spec.instruction : spec.intent;

/** Splits a `[workspace/]name[#revision]` ref; a bare name lives in `fallbackWorkspace`. */
export const parseTaskRef = (ref: string, fallbackWorkspace: string): TaskRefParts => {
  const [path, version] = ref.split('#');
  const slash = path.indexOf('/');
  return slash === -1
    ? { workspace: fallbackWorkspace, name: path, version }
    : { workspace: path.slice(0, slash), name: path.slice(slash + 1), version };
};

/** A task's workspace-qualified ref without a revision: the key members resolve against. */
export const taskRef = (task: Pick<Task, 'workspace' | 'name'>): string =>
  `${task.workspace}/${task.name}`;
