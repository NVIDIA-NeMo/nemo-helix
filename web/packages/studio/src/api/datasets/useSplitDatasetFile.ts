// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { isDefined } from '@nemo/common/src/utils/isDefined';
import { filesUploadFile } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { getFileExtension, parseFileContent } from '@studio/util/files';
import { splitRandomDistribution, splitSequentialDistribution } from '@studio/util/list';
import { useMutation, UseMutationOptions } from '@tanstack/react-query';
import { useCallback } from 'react';

interface FileSplitsProps {
  filepath: string;
  fileContent: string;
  workspace: string;
  datasetName: string;
  distributionType: 'random' | 'sequential';
  splits: string[];
  seed?: string;
  sortKey?: string;
}

type Props = Omit<UseMutationOptions<FilesetFileOutput[], Error, FileSplitsProps>, 'mutationFn'>;

const fileSuffix = ['training', 'testing', 'validation'];

export const useSplitDatasetFile = ({ onError, onSuccess }: Props) => {
  const toast = useToast();

  const mutationFn = useCallback(
    async ({
      fileContent,
      workspace,
      datasetName,
      filepath,
      splits,
      distributionType,
      seed,
      sortKey,
    }: FileSplitsProps) => {
      const { rows, failures } = parseFileContent({
        content: fileContent,
        fileType: getFileExtension(filepath) ?? '',
      });
      if (failures?.length) {
        toast.error(`${failures.length} Line(s) had parsing errors.`);
      }

      const splitList =
        distributionType === 'random'
          ? splitRandomDistribution(rows, splits, seed)
          : splitSequentialDistribution(rows, splits, { key: sortKey });

      const isJson = filepath.toLowerCase().endsWith('json');
      const sourceName = filepath.split('/').pop() ?? filepath;
      const baseName = sourceName.replace(/\.[^./]+$/, '');
      const outputName = `${baseName}.${isJson ? 'json' : 'jsonl'}`;
      const toUpload = splits
        .map((_, index) => {
          let content = splitList[index]
            .map((row) => JSON.stringify(row))
            .join(isJson ? ',\n' : '\n');
          if (isJson) {
            content = '[' + content + ']';
          }
          if (content.length === 0) {
            return undefined;
          }
          return {
            path: `${fileSuffix[index]}/${outputName}`,
            content,
          };
        })
        .filter(isDefined);

      const results = await Promise.all(
        toUpload.map(async (details) => {
          const blob = new Blob([details.content], { type: 'application/json' });
          return filesUploadFile(workspace, datasetName, details.path, blob);
        })
      );

      return results;
    },
    [toast]
  );

  return {
    ...useMutation({
      mutationFn,
      onError: (data, variables, onMutateResult, context) => {
        onError?.(data, variables, onMutateResult, context);
      },
      onSuccess: (data, variables, onMutateResult, context) => {
        invalidateDatasetCaches(variables.workspace, variables.datasetName, ['files']);
        onSuccess?.(data, variables, onMutateResult, context);
      },
    }),
  };
};
