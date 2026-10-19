// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { CustomizationFileType } from '@studio/constants/customization';
import { parseFilesetUri } from '@studio/hooks/useCustomizationFiles/utils';
import { useDatasetFileDiscovery } from '@studio/hooks/useDatasetFileDiscovery';
import { useQueries } from '@tanstack/react-query';

export { parseFilesetUri } from '@studio/hooks/useCustomizationFiles/utils';
export {
  useDatasetFileDiscovery,
  type DatasetFileDiscoveryResult,
} from '@studio/hooks/useDatasetFileDiscovery';

interface UseCustomizationFilesOptions {
  fileset?: string;
}

interface CustomizationFilesResult {
  training: {
    hasFiles: boolean;
    files: FilesetFileOutput[];
    total: number;
  };
  validation: {
    hasFiles: boolean;
    files: FilesetFileOutput[];
    total: number;
  };
  hasRequiredFiles: boolean;
  isPending: boolean;
}

/**
 * React Query hook for fetching the training and validation files inside a fileset.
 * Discovery rules mirror the Customizer service (see useDatasetFileDiscovery): files
 * inside train, training, val, validation, dev subfolders are matched, plus root-level
 * files whose names start with train, training, val, validation, or dev. A lone
 * unmatched root .jsonl is also treated as training (auto-split applies).
 */
export const useCustomizationFiles = ({
  fileset,
}: UseCustomizationFilesOptions): CustomizationFilesResult => {
  const { training, validation, hasRequiredFiles, isPending } = useDatasetFileDiscovery({
    fileset,
  });

  return {
    training: {
      hasFiles: training.length > 0,
      files: training,
      total: training.length,
    },
    validation: {
      hasFiles: validation.length > 0,
      files: validation,
      total: validation.length,
    },
    hasRequiredFiles,
    isPending,
  };
};

interface UseCustomizationFilesPreviewOptions extends UseCustomizationFilesOptions {
  previewLimit?: number;
}

/** React Query hook identical to useCustomizationFiles() hook,
 * except that it limits the number of files returned. There is no limit to how
 * many files are in "training" or "validation" folders, so using this hook spares
 * components in the presentation layer from having to handle large amounts of files.
 *
 * @param params - Object containing fileset and preview limit
 * @returns Query result containing limited number of JSONL files within the given
 * folder, plus a total amount of all files in the folder.
 */
export const useCustomizationFilesPreview = ({
  fileset,
  previewLimit = 1,
}: UseCustomizationFilesPreviewOptions): CustomizationFilesResult => {
  const result = useCustomizationFiles({ fileset });

  return {
    ...result,
    training: {
      ...result.training,
      files: result.training.files.slice(0, previewLimit),
    },
    validation: {
      ...result.validation,
      files: result.validation.files.slice(0, previewLimit),
    },
  };
};

/**
 * This hook returns the rows of the customization training and validation files as an array of objects.
 * Each object contains the type of the file (training or validation), the path of the file, the
 * size of the file, and the content of the file.
 *
 * `content` is the size-capped preview from `datasetFileContentQueryOptions`, not the whole file,
 * so it cannot be used to count the file's rows.
 * @param fileset - The fileset URI to fetch the customization files from (e.g. fileset://workspace/name).
 * @returns
 */
export const useCustomizationFilesAsRows = ({ fileset }: UseCustomizationFilesOptions) => {
  const { name, workspace } = parseFilesetUri(fileset ?? '');
  const { training, validation, ...queryResults } = useCustomizationFiles({ fileset });

  // Tag each file with its category up front so we don't need to re-derive the
  // type by path-includes — that approach broke for root-level files like
  // train_a.jsonl that don't contain the literal "training" segment.
  const taggedFiles = [
    ...training.files.map((file) => ({ file, type: CustomizationFileType.Training })),
    ...validation.files.map((file) => ({ file, type: CustomizationFileType.Validation })),
  ];

  const { rows, isFetching: isFetchingRows } = useQueries({
    queries: taggedFiles.map(({ file }) => ({
      enabled: Boolean(workspace && name),
      ...datasetFileContentQueryOptions({
        workspace,
        name,
        path: file.path,
      }),
      meta: { path: file.path },
    })),
    combine: (results) => ({
      rows: results.map((result, index) => {
        const { file, type } = taggedFiles[index];
        return {
          type,
          path: file.path,
          size: file.size,
          content: result.data ?? '',
        };
      }),
      isFetching: results.some((result) => result.isFetching),
    }),
  });
  return {
    rows,
    isFetchingRows,
    ...queryResults,
  };
};
