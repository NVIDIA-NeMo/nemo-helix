// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import {
  CUSTOMIZATION_DATASET_DISCOVERY,
  CustomizationFileType,
} from '@studio/constants/customization';
import { parseFilesetUri } from '@studio/hooks/useCustomizationFiles/utils';

export interface DatasetFileDiscoveryResult {
  training: FilesetFileOutput[];
  validation: FilesetFileOutput[];
  hasRequiredFiles: boolean;
  isPending: boolean;
  /**
   * Set when listing the fileset's files failed (network/API/permissions).
   * Distinct from "the fileset is empty" — callers should render a retryable
   * load error rather than a dataset-quality error.
   */
  error: Error | null;
}

const DATASET_EXTENSIONS = ['.jsonl', '.json'] as const;

const segmentsOf = (path: string): string[] => path.split('/').filter(Boolean);

const isDatasetFile = (path: string): boolean => {
  const name = path.toLowerCase();
  return DATASET_EXTENSIONS.some((ext) => name.endsWith(ext));
};

/** The split a file holds by its name (root files) or its directory, or null when neither says. */
const splitOf = (segments: string[]): CustomizationFileType | null => {
  for (const type of [CustomizationFileType.Training, CustomizationFileType.Validation]) {
    const rule = CUSTOMIZATION_DATASET_DISCOVERY[type];
    if (segments.length === 1 && rule.filePatterns.some((re) => re.test(segments[0]))) return type;
    if (segments.length === 2 && rule.dirs.includes(segments[0])) return type;
  }
  return null;
};

/** Drop `.json` files when `.jsonl` files are present, as Customizer does. */
const preferJsonl = (files: FilesetFileOutput[]): FilesetFileOutput[] => {
  const jsonl = files.filter((f) => f.path.toLowerCase().endsWith('.jsonl'));
  return jsonl.length > 0 ? jsonl : files;
};

/**
 * Splits a fileset's files into the training and validation files Customizer reads, by the
 * rule every backend shares (see CUSTOMIZATION_DATASET_DISCOVERY).
 */
export const partitionDatasetFiles = (
  files: FilesetFileOutput[]
): { training: FilesetFileOutput[]; validation: FilesetFileOutput[] } => {
  // Sorted by code point, as Customizer sorts them, so both list the files in one order.
  const dataFiles = files
    .filter((f) => isDatasetFile(f.path))
    .sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  let training = dataFiles.filter(
    (f) => splitOf(segmentsOf(f.path)) === CustomizationFileType.Training
  );
  const validation = dataFiles.filter(
    (f) => splitOf(segmentsOf(f.path)) === CustomizationFileType.Validation
  );
  if (training.length === 0 && validation.length === 0) {
    training = dataFiles.filter(
      (f) => segmentsOf(f.path).length === 1 && f.path.endsWith('.jsonl')
    );
  }
  return { training: preferJsonl(training), validation: preferJsonl(validation) };
};

/**
 * Lists every file in a fileset (no path filter) and partitions them into
 * training / validation buckets with partitionDatasetFiles.
 */
export const useDatasetFileDiscovery = ({
  fileset,
}: {
  fileset?: string;
}): DatasetFileDiscoveryResult => {
  const { workspace, name } = parseFilesetUri(fileset ?? '');
  const enabled = !!workspace && !!name;

  const {
    data: filesResponse,
    isPending,
    error,
  } = useFilesListFilesetFiles(workspace, name, undefined, {
    query: {
      enabled,
      select: (res) => res.data,
    },
  });

  const { training, validation } = partitionDatasetFiles(filesResponse ?? []);

  return {
    training,
    validation,
    hasRequiredFiles: training.length > 0,
    isPending: enabled && isPending,
    error: error ?? null,
  };
};
