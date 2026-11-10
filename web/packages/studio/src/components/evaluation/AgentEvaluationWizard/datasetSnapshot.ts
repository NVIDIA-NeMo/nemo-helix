// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';

export const DATASET_SNAPSHOT_KEY = 'dataset_snapshot';

export interface DatasetFileStat {
  path: string;
  size: number;
}

export interface DatasetSnapshot {
  ref: string;
  files: DatasetFileStat[];
}

export type DatasetDrift = 'missing' | 'changed' | null;

const REGEX_SPECIAL = /[.*+?^${}()|[\]\\]/g;

const pathPattern = (objectPath: string): RegExp =>
  new RegExp(
    `^${objectPath
      .split('*')
      .map((part) => part.replace(REGEX_SPECIAL, '\\$&'))
      .join('[^/]*')}$`
  );

const listDatasetFiles = async (
  workspace: string,
  ref: string,
  signal?: AbortSignal
): Promise<DatasetFileStat[]> => {
  const location = parseFilesetLocation(ref, workspace);
  if (!location) {
    return [];
  }
  try {
    const { data } = await filesListFilesetFiles(
      location.workspace,
      location.name,
      undefined,
      signal
    );
    const pattern = pathPattern(location.objectPath);
    return data.filter((file) => pattern.test(file.path)).map(({ path, size }) => ({ path, size }));
  } catch (err) {
    if (isNotFoundError(err)) {
      return [];
    }
    throw err;
  }
};

export const takeDatasetSnapshot = async (
  workspace: string,
  ref: string,
  signal?: AbortSignal
): Promise<DatasetSnapshot | null> => {
  const files = await listDatasetFiles(workspace, ref, signal);
  return files.length > 0 ? { ref, files } : null;
};

const isDatasetSnapshot = (value: unknown): value is DatasetSnapshot => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const { ref, files } = value as Partial<DatasetSnapshot>;
  return (
    typeof ref === 'string' &&
    Array.isArray(files) &&
    files.every(
      (file: unknown) =>
        typeof file === 'object' &&
        file !== null &&
        typeof (file as DatasetFileStat).path === 'string' &&
        typeof (file as DatasetFileStat).size === 'number'
    )
  );
};

export const datasetSnapshotOf = (evaluation: EvaluationResponse): DatasetSnapshot | null => {
  const raw = evaluation.metadata?.[DATASET_SNAPSHOT_KEY];
  if (!raw) {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return isDatasetSnapshot(parsed) ? parsed : null;
  } catch {
    return null;
  }
};

export const datasetDrift = async (
  workspace: string,
  snapshot: DatasetSnapshot,
  signal?: AbortSignal
): Promise<DatasetDrift> => {
  const files = await listDatasetFiles(workspace, snapshot.ref, signal);
  if (files.length === 0) {
    return 'missing';
  }
  const recorded = new Map(snapshot.files.map((file) => [file.path, file.size]));
  const unchanged =
    files.length === recorded.size && files.every((file) => recorded.get(file.path) === file.size);
  return unchanged ? null : 'changed';
};
