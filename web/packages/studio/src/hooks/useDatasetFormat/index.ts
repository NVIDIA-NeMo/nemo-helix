// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { parseFilesetUri } from '@studio/hooks/useCustomizationFiles/utils';
import { useDatasetFileDiscovery } from '@studio/hooks/useDatasetFileDiscovery';
import {
  type CustomizerSchemaDetection,
  detectCustomizerSchema,
  inferRowSchema,
} from '@studio/util/customizerSchema';
import { useQuery, useQueryClient } from '@tanstack/react-query';

export interface DatasetFormat {
  /** Null when the first training row matches no format customizer accepts. */
  schema: CustomizerSchemaDetection | null;
  /**
   * Rows in the training files. File content arrives as a size-capped preview, so a file
   * larger than the preview is extrapolated from its byte size; see `rowCountIsEstimate`.
   */
  trainingRowCount: number;
  rowCountIsEstimate: boolean;
  hasValidation: boolean;
  /** Field names and types of the first training row — no values. Empty when there are no rows. */
  shape: string;
}

export interface UseDatasetFormatResult {
  format: DatasetFormat | null;
  isPending: boolean;
  error: Error | null;
}

const firstRow = (content: string): Record<string, unknown> | null => {
  const line = content.split('\n').find((candidate) => candidate.trim());
  if (!line) return null;
  try {
    const parsed: unknown = JSON.parse(line);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
};

const countRows = (content: string): number =>
  content.split('\n').filter((line) => line.trim()).length;

/**
 * Rows in a file, given the possibly size-capped preview of it. A preview shorter than the
 * file is scaled up by bytes, which assumes the rows read are typical of the whole file.
 */
export const estimateRows = (
  preview: string,
  fileSize: number
): { rows: number; estimated: boolean } => {
  const rows = countRows(preview);
  const bytesRead = new TextEncoder().encode(preview).length;
  return bytesRead > 0 && fileSize > bytesRead
    ? { rows: Math.round((rows * fileSize) / bytesRead), estimated: true }
    : { rows, estimated: false };
};

/**
 * A dataset's format and size, without a training type to check it against — unlike
 * `useCustomizationDatasetValidation`, which validates against the one the form has chosen.
 * Reads through the same cached file queries, so the two never download a file twice.
 */
export const useDatasetFormat = (fileset: string | null): UseDatasetFormatResult => {
  const queryClient = useQueryClient();
  const { workspace, name } = parseFilesetUri(fileset ?? '');
  const discovery = useDatasetFileDiscovery({ fileset: fileset ?? undefined });
  const paths = discovery.training.map((file) => file.path);

  const query = useQuery({
    enabled: !!fileset && !discovery.isPending && !discovery.error,
    queryKey: ['customization-dataset-format', workspace, name, [...paths].sort()],
    queryFn: async (): Promise<DatasetFormat> => {
      const contents = await Promise.all(
        discovery.training.map((file) =>
          queryClient.ensureQueryData(
            datasetFileContentQueryOptions({ workspace, name, path: file.path })
          )
        )
      );
      const counts = contents.map((content, index) =>
        estimateRows(content, discovery.training[index].size)
      );
      const row = contents.map(firstRow).find(Boolean) ?? null;
      return {
        // GRPO and DPO keys are distinctive; SFT's `prompt` also appears in preference rows.
        schema:
          detectCustomizerSchema(row, 'grpo') ??
          detectCustomizerSchema(row, 'dpo') ??
          detectCustomizerSchema(row, 'sft'),
        trainingRowCount: counts.reduce((total, { rows }) => total + rows, 0),
        rowCountIsEstimate: counts.some(({ estimated }) => estimated),
        hasValidation: discovery.validation.length > 0,
        shape: inferRowSchema(row),
      };
    },
  });

  if (!fileset) return { format: null, isPending: false, error: null };
  return {
    format: query.data ?? null,
    isPending: discovery.isPending || query.isPending,
    error: discovery.error ?? query.error ?? null,
  };
};
