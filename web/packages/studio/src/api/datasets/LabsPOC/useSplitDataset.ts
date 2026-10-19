// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getWorkspaceDatasetName } from '@nemo/common/src/datasets/constants';
import {
  parseFileContent,
  convertToFileFormat,
  SUPPORTED_FILE_EXTENSIONS,
  getFileMimeType,
  SupportedFileExtension,
} from '@nemo/common/src/LabsPOC/dataset-file-utils';
import { filesUploadFile } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import {
  FILESET_EVAL_FILE_NAME,
  FILESET_EVAL_FILE_PATH,
  FILESET_ICLS_FILE_NAME,
  FILESET_ICLS_FILE_PATH,
  FILESET_TRAINING_FILE_NAME,
  FILESET_TRAINING_FILE_PATH,
  FILESET_VALIDATION_FILE_NAME,
  FILESET_VALIDATION_FILE_PATH,
} from '@studio/api/datasets/LabsPOC/constants';
import type { SplitFormData } from '@studio/api/datasets/LabsPOC/formSchema';
import { getFileExtension } from '@studio/util/files';
import { useMutation, UseMutationOptions } from '@tanstack/react-query';

const splitFile = async (file: File, split: SplitFormData) => {
  let evalFile, trainingFile, validationFile, iclsFile;

  // Read file contents
  const text = await file.text();
  const fileExtension = getFileExtension(file) as SupportedFileExtension;

  if (!fileExtension || !SUPPORTED_FILE_EXTENSIONS.includes(fileExtension)) {
    throw new Error(`Unsupported file type: ${fileExtension}`);
  }

  const data = parseFileContent(text, fileExtension);
  const totalItems = data.length;

  // Calculate item counts for each split
  const evalItems = Math.floor(totalItems * (split.eval / 100));
  const trainItems = Math.floor(totalItems * (split.train / 100));
  const validateItems = Math.floor(totalItems * (split.validate / 100));
  const iclsItems = Math.floor(totalItems * (split.icls / 100));

  // Create array of indices in original order (no randomization)
  const indices = Array.from({ length: totalItems }, (_, i) => i);

  let currentIndex = 0;

  if (split.eval) {
    const evalData = indices.slice(currentIndex, currentIndex + evalItems).map((i) => data[i]);
    const evalContent = convertToFileFormat(evalData, fileExtension);
    evalFile = new File([evalContent], FILESET_EVAL_FILE_NAME, {
      type: getFileMimeType(fileExtension),
    });
    currentIndex += evalItems;
  }

  if (split.train) {
    const trainData = indices.slice(currentIndex, currentIndex + trainItems).map((i) => data[i]);
    const trainContent = convertToFileFormat(trainData, fileExtension);
    trainingFile = new File([trainContent], FILESET_TRAINING_FILE_NAME, {
      type: getFileMimeType(fileExtension),
    });
    currentIndex += trainItems;
  }

  if (split.validate) {
    const validateData = indices
      .slice(currentIndex, currentIndex + validateItems)
      .map((i) => data[i]);
    const validateContent = convertToFileFormat(validateData, fileExtension);
    validationFile = new File([validateContent], FILESET_VALIDATION_FILE_NAME, {
      type: getFileMimeType(fileExtension),
    });
    currentIndex += validateItems;
  }

  if (split.icls) {
    const iclsData = indices.slice(currentIndex, currentIndex + iclsItems).map((i) => data[i]);
    const iclsContent = convertToFileFormat(iclsData, fileExtension);
    iclsFile = new File([iclsContent], FILESET_ICLS_FILE_NAME, {
      type: getFileMimeType(fileExtension),
    });
  }

  return {
    files: {
      evalFile,
      trainingFile,
      validationFile,
      iclsFile,
    },
    counts: {
      evalLines: evalItems,
      trainLines: trainItems,
      validateLines: validateItems,
      iclsLines: iclsItems,
    },
  };
};

export interface SplitDatasetMutationVars {
  workspace: string;
  file: File;
  splitData: SplitFormData;
}

export interface SplitCounts {
  evalLines: number;
  trainLines: number;
  iclsLines: number;
  validateLines: number;
}

export interface SplitDatasetOutput {
  uploadResults: FilesetFileOutput[];
  counts: SplitCounts;
}

/**
 * @deprecated Use useUploadFilesetsFileV2 from the per-tag modules under '@nemo/sdk/generated/platform/' directly.
 * Implement split logic separately and upload each split file using the generated hook.
 */
export const useSplitDataset = (
  options?: UseMutationOptions<SplitDatasetOutput, Error, SplitDatasetMutationVars>
) => {
  return useMutation({
    ...options,
    mutationFn: async ({ workspace, file, splitData }) => {
      const datasetName = getWorkspaceDatasetName(workspace);

      const fileExtension = getFileExtension(file);

      const {
        files: { evalFile, trainingFile, validationFile, iclsFile },
        counts,
      } = await splitFile(file, splitData);

      const filesToUpload: { path: string; file: File }[] = [];

      if (evalFile) {
        filesToUpload.push({
          path: FILESET_EVAL_FILE_PATH + fileExtension,
          file: evalFile,
        });
      }

      if (trainingFile) {
        filesToUpload.push({
          path: FILESET_TRAINING_FILE_PATH + fileExtension,
          file: trainingFile,
        });
      }

      if (validationFile) {
        filesToUpload.push({
          path: FILESET_VALIDATION_FILE_PATH + fileExtension,
          file: validationFile,
        });
      }

      if (iclsFile) {
        filesToUpload.push({
          path: FILESET_ICLS_FILE_PATH + fileExtension,
          file: iclsFile,
        });
      }

      // Upload all files using v2 API
      const uploadResults = await Promise.all(
        filesToUpload.map(async ({ path, file: fileToUpload }) => {
          const blob = new Blob([await fileToUpload.arrayBuffer()], { type: fileToUpload.type });
          return filesUploadFile(workspace, datasetName, path, blob);
        })
      );

      return {
        uploadResults,
        counts,
      };
    },
  });
};
