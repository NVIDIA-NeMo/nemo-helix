// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CheckmarkBadge, CheckCircle, Play } from '@nemo/icons/icons';
import { Flex } from '@nvidia/foundations-react-core';
import { ReactNode } from 'react';

/**
 * An enum of the files in a Data Store fileset needed to create a customization.
 */
export enum CustomizationFileType {
  Training = 'Training',
  Validation = 'Validation',
}

/**
 * A wrapper around the native File type that includes what kind of CustomizationFileType
 * it is. The reason for a wrapper file instead of a Record is because there are some
 * instances where we may want to have information about the filetype where only one
 * file is relevant, and Records/keyof patterns require that all enum values be present.
 */
export type FileWithType = {
  type?: CustomizationFileType;
  file: File;
};

// https://developer.mozilla.org/en-US/docs/Web/HTTP/MIME_types/Common_types
export const CUSTOMIZATION_FILESET_FILE_ACCEPT = {
  'application/json': ['.jsonl'],
};

export const CUSTOMIZATION_FILES_ALLOWED_FILE_EXTENSIONS = ['.jsonl'];

/**
 * The "filepath" path arg to Data Store's upload fileset file endpoint that is expected
 * by Customizer for each of its training / testing / validation files.
 */
export const CUSTOMIZATION_FILESET_FILE_PREFIXES: Record<CustomizationFileType, string> = {
  [CustomizationFileType.Training]: 'training',
  [CustomizationFileType.Validation]: 'validation',
};

/**
 * Directories whose files hold each split, by the rule every Customizer backend shares
 * (packages/nhx_customization_common/src/nhx/customization_common/dataset_files.py):
 * a root file whose name contains `train` is training data; otherwise one whose name
 * contains `val` or starts with `dev` is validation data; so are the files directly in
 * these directories. With neither, every root data file is training data, or, with none
 * at the root, every nested one. useDatasetFileDiscovery applies the rule.
 */
export const CUSTOMIZATION_DATASET_DIRS: Record<CustomizationFileType, readonly string[]> = {
  [CustomizationFileType.Training]: ['train', 'training'],
  [CustomizationFileType.Validation]: ['val', 'validation', 'dev'],
};

/**
 * Share of the training rows Customizer holds out for validation when the dataset has no
 * validation files: the default `schedule.validation_split` of automodel and unsloth, and
 * the fixed split of rl.
 */
export const CUSTOMIZER_AUTO_VAL_SPLIT_RATIO = 0.1;

export const CUSTOMIZATION_FILESET_FILEPATHS: Record<CustomizationFileType, string> = {
  [CustomizationFileType.Training]: `${CUSTOMIZATION_FILESET_FILE_PREFIXES.Training}/training_file.jsonl`,
  [CustomizationFileType.Validation]: `${CUSTOMIZATION_FILESET_FILE_PREFIXES.Validation}/validation_file.jsonl`,
};

export const CUSTOMIZATION_FILESET_FILE_LABELS: Record<CustomizationFileType, string> = {
  [CustomizationFileType.Training]: 'Training File(s)',
  [CustomizationFileType.Validation]: 'Validation File(s)',
};

export const CUSTOMIZATION_FILESET_FILE_ICONS: Record<CustomizationFileType, ReactNode> = {
  [CustomizationFileType.Training]: <Play />,
  [CustomizationFileType.Validation]: <CheckmarkBadge />,
};

export const CUSTOMIZATION_FILESET_FILE_HELPERS: Record<CustomizationFileType, ReactNode> = {
  [CustomizationFileType.Training]: (
    <span>
      All valid .jsonl files within the <strong>/training</strong> subfolder will be concatenated
      during customization.
    </span>
  ),
  [CustomizationFileType.Validation]: (
    <span>
      All valid .jsonl files within the <strong>/validation</strong> subfolder will be concatenated
      during customization.
    </span>
  ),
};

export const CUSTOMIZATION_TYPE_BADGES: Record<CustomizationFileType, ReactNode> = {
  [CustomizationFileType.Training]: (
    <Flex align="center" gap="density-xs">
      <Play /> Training
    </Flex>
  ),
  [CustomizationFileType.Validation]: (
    <Flex align="center" gap="density-xs">
      <CheckCircle /> Validation
    </Flex>
  ),
};
