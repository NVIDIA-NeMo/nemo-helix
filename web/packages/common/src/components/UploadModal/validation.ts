/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import { UploadDataset, UploadFile } from '@nemo/common/src/components/UploadModal/types';
import { isFileTypeAllowed } from '@nemo/common/src/components/UploadModal/utils';
import { DATASET_NAME_REGEX } from '@nemo/common/src/datasets/constants';

interface ValidateUploadFormParams {
  selectedFiles: UploadFile[];
  selectedDataset: UploadDataset | undefined;
  isDatasetRequired: boolean;
  /** Extensions a selected file must match when `invalidFileMode` is 'disable' or 'hide'. */
  acceptableFileTypes?: string[];
  invalidFileMode?: 'show' | 'hide' | 'disable';
}

/**
 * Validates the upload modal form and returns any validation errors.
 * Returns an empty object if validation passes.
 *
 * @param params - The form parameters to validate
 * @returns Record of field names to error messages
 */
export const validateUploadForm = ({
  selectedFiles,
  selectedDataset,
  isDatasetRequired = false,
  acceptableFileTypes,
  invalidFileMode,
}: ValidateUploadFormParams): Record<string, string> => {
  const errors: Record<string, string> = {};

  // Check if file is selected
  if (selectedFiles.length === 0) {
    errors.file = 'File is required.';
  } else if (
    invalidFileMode !== undefined &&
    invalidFileMode !== 'show' &&
    selectedFiles.some((f) => !isFileTypeAllowed(f, acceptableFileTypes))
  ) {
    errors.file = `Only ${acceptableFileTypes?.join(', ')} files can be selected.`;
  }

  // Check if dataset is required and selected
  if (isDatasetRequired && !selectedDataset) {
    errors.dataset = 'Fileset is required.';
  }

  // Check if creating new dataset and name is provided
  if (selectedDataset && selectedDataset.type === 'new') {
    if (selectedDataset.name === '') {
      errors.datasetName = 'Fileset name is required.';
    } else if (!DATASET_NAME_REGEX.test(selectedDataset.name)) {
      errors.datasetName =
        'Fileset name must only contain alphanumeric characters, dashes, underscores, or dots.';
    }
  }

  return errors;
};
