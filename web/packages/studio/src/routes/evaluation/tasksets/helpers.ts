// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TasksetTags } from '@nemo/sdk/generated/evals/schema';
import type {
  FilesRefParts,
  TaskFilters,
  TasksetTaskRow,
} from '@studio/routes/evaluation/tasksets/types';

export const parseFilesRef = (ref: string): FilesRefParts => {
  const [path, file] = ref.split('#');
  const [workspace, fileset] = path.split('/');
  return { workspace, fileset, path: file };
};

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
