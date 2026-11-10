// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Task, TasksetTags } from '@nemo/sdk/generated/evals/schema';
import type {
  FilesRefParts,
  TaskFilters,
  TaskRefParts,
  TasksetTaskRow,
} from '@studio/routes/evaluation/tasksets/types';
import { getFilesetDetailRoute } from '@studio/routes/utils';

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

export const parseFilesRef = (ref: string): FilesRefParts => {
  const [path, file] = ref.split('#');
  const [workspace, fileset] = path.split('/');
  return { workspace, fileset, path: file };
};

export const getFilesetRoute = ({ workspace, fileset }: FilesRefParts): string =>
  getFilesetDetailRoute(workspace, fileset);

/** Tag names, `latest` first then alphabetical. */
export const tagNames = (tags: TasksetTags | undefined): string[] =>
  Object.keys(tags ?? {}).sort(
    (a, b) => Number(b === 'latest') - Number(a === 'latest') || a.localeCompare(b)
  );

export const formatValue = (value: unknown): string =>
  typeof value === 'string' ? value : JSON.stringify(value);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** A `task.toml` section (e.g. `environment`, `verifier`) from a Harbor task's config. */
export const configSection = (config: unknown, key: string): Record<string, unknown> => {
  const section = isRecord(config) ? config[key] : undefined;
  return isRecord(section) ? section : {};
};

/** DataView request status: an error wins over an in-flight fetch. */
export const requestStatus = (error: unknown, isFetching: boolean) => {
  if (error) return 'error';
  return isFetching ? 'loading' : undefined;
};

const includesText = (text: string, query: string) =>
  text.toLowerCase().includes(query.toLowerCase());

/** Whether a member row passes the task table's search and column filters. A row whose task
 *  didn't resolve has no runner or metadata, so it only survives the name search. */
export const matchesTaskFilters = (
  { name, task }: TasksetTaskRow,
  { name: query, runner, metadata }: TaskFilters
): boolean => {
  if (query && !includesText(name, query)) return false;
  if (runner && Object.keys(runner).length && !(task && runner[task.spec.kind])) return false;
  if (metadata) {
    const pairs = (task?.metadata ?? []).map(({ key, value }) => `${key}: ${formatValue(value)}`);
    if (!pairs.some((pair) => includesText(pair, metadata))) return false;
  }
  return true;
};
