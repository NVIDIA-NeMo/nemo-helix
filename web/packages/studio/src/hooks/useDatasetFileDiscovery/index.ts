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

const isDatasetFile = (filename: string): boolean =>
  DATASET_EXTENSIONS.some((ext) => filename.endsWith(ext));

const matchesAnyPattern = (filename: string, patterns: readonly RegExp[]): boolean =>
  patterns.some((re) => re.test(filename));

interface PartitionedFiles {
  training: FilesetFileOutput[];
  validation: FilesetFileOutput[];
  unmatchedRootJsonl: FilesetFileOutput[];
}

export const partitionDatasetFiles = (files: FilesetFileOutput[]): PartitionedFiles => {
  const training: FilesetFileOutput[] = [];
  const validation: FilesetFileOutput[] = [];
  const unmatchedRootJsonl: FilesetFileOutput[] = [];
  const trainingRule = CUSTOMIZATION_DATASET_DISCOVERY[CustomizationFileType.Training];
  const validationRule = CUSTOMIZATION_DATASET_DISCOVERY[CustomizationFileType.Validation];

  for (const f of files) {
    const segments = f.path.split('/').filter(Boolean);
    if (segments.length === 0) continue;
    const filename = segments[segments.length - 1];
    if (!isDatasetFile(filename)) continue;

    const top = segments[0];
    if (segments.length > 1) {
      if (trainingRule.dirs.includes(top)) {
        training.push(f);
      } else if (validationRule.dirs.includes(top)) {
        validation.push(f);
      }
      continue;
    }

    // Root-level file
    if (matchesAnyPattern(filename, trainingRule.filePatterns)) {
      training.push(f);
    } else if (matchesAnyPattern(filename, validationRule.filePatterns)) {
      validation.push(f);
    } else if (filename.endsWith('.jsonl')) {
      // Only .jsonl is eligible for the lone-root fallback (matches customizer).
      unmatchedRootJsonl.push(f);
    }
  }

  return { training, validation, unmatchedRootJsonl };
};

/**
 * Lists every file in a fileset (no path filter) and partitions them into
 * training / validation buckets using the same discovery rules as the
 * Customizer service. See CUSTOMIZATION_DATASET_DISCOVERY and
 * services/customizer/src/nhx/customizer/tasks/training/datasets/preparation.py.
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

  const files = filesResponse ?? [];
  const { training, validation, unmatchedRootJsonl } = partitionDatasetFiles(files);

  // Customizer fallback (preparation.py:324-336): when no train/val patterns
  // matched at all, ALL unmatched root .jsonl files are claimed as training.
  // Single file is unambiguous; multiple files trigger a warning in customizer
  // but still get treated as training (the merged set is auto-split for val).
  const useRootFallback =
    training.length === 0 && validation.length === 0 && unmatchedRootJsonl.length > 0;

  const finalTraining = useRootFallback ? unmatchedRootJsonl : training;

  return {
    training: finalTraining,
    validation,
    hasRequiredFiles: finalTraining.length > 0,
    isPending: enabled && isPending,
    error: error ?? null,
  };
};
