// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesUploadFile } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { renderTemplate } from '@studio/components/transform/renderTemplate';
import { parseFileContent } from '@studio/util/files';
import { useMutation, type UseMutationOptions } from '@tanstack/react-query';
import { useCallback } from 'react';

interface MutationProps {
  workspace: string;
  datasetName: string;
  filepath: string;
  /** The `schema_transform` template each row is rewritten through. */
  template: Record<string, unknown>;
  fileContent: string;
  /**
   * Column the template references but the source file does not have. A Data
   * Designer job would declare it as a UUID sampler; here each row gets its own
   * UUID as it is rewritten.
   */
  generatedIdColumn?: string;
}

type Props = Omit<UseMutationOptions<FilesetFileOutput, Error, MutationProps>, 'mutationFn'>;

/**
 * The in-place transform rewrites rows and re-serializes them as JSONL, so it
 * only accepts files that are already JSONL. Anything else — CSV, a JSON array,
 * Parquet — would be overwritten in a format its extension does not describe.
 */
export const isTransformableFilePath = (filepath: string): boolean => /\.jsonl$/i.test(filepath);

/**
 * Rewrites a fileset file in place through a transform template. The mapping is
 * applied in the browser with the same renderer that drives the preview, so what
 * the user approved is exactly what is uploaded.
 */
export const useDatasetFileTransform = ({ onError, onSuccess }: Props) => {
  const mutationFn = useCallback(
    async ({
      fileContent,
      filepath,
      template,
      workspace,
      datasetName,
      generatedIdColumn,
    }: MutationProps) => {
      if (!isTransformableFilePath(filepath)) {
        throw new Error('Only JSONL files can be transformed in place.');
      }

      const { rows, failures } = parseFileContent({ content: fileContent, fileType: 'jsonl' });
      if (failures?.length) {
        throw new Error(
          `${failures.length} line(s) could not be parsed, so the file was left unchanged.`
        );
      }
      if (!rows.length) {
        throw new Error('No rows could be read from this file, so it was left unchanged.');
      }

      const transformed = rows.map((row) => {
        const input = generatedIdColumn
          ? { ...row, [generatedIdColumn]: crypto.randomUUID() }
          : row;
        return renderTemplate(template, input).row;
      });
      const blob = new Blob([transformed.map((row) => JSON.stringify(row)).join('\n')], {
        type: 'application/json',
      });

      return filesUploadFile(workspace, datasetName, filepath, blob);
    },
    []
  );

  return useMutation({
    mutationFn,
    onError: (data, variables, onMutateResult, context) => {
      onError?.(data, variables, onMutateResult, context);
    },
    onSuccess: (data, variables, onMutateResult, context) => {
      invalidateDatasetCaches(
        variables.workspace,
        variables.datasetName,
        ['files', 'content'],
        variables.filepath
      );
      onSuccess?.(data, variables, onMutateResult, context);
    },
  });
};
