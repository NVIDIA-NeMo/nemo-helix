// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesetRowsQueryOptions } from '@studio/api/datasets/filesetParquetRows';
import type { CustomizationTemplateDataset } from '@studio/constants/customizationTemplates';
import type { QueryClient } from '@tanstack/react-query';

/** Parquet INT64 (and similar) columns decode as BigInt; JSON.stringify rejects those by default. */
function jsonReplacer(_key: string, value: unknown): unknown {
  return typeof value === 'bigint' ? value.toString() : value;
}

const toJsonlBlob = (rows: Record<string, unknown>[]): Blob =>
  new Blob([rows.map((row) => JSON.stringify(row, jsonReplacer)).join('\n')], {
    type: 'application/x-ndjson',
  });

/** Which stage of the fetch is running, so the caller can word its own status copy. */
export type FetchDatasetPhase = 'locating' | 'downloading' | 'converting';

export type FetchDatasetProgress = (
  phase: FetchDatasetPhase,
  loadedBytes?: number,
  totalBytes?: number
) => void;

/**
 * Pulls a template's dataset rows through the files service and converts them to the JSONL
 * Customizer takes.
 *
 * Rows come from an external fileset the caller has already registered against the
 * HuggingFace repo, so the bytes are proxied server-side. That is deliberate: the browser
 * is not assumed to reach huggingface.co at all.
 */
export const fetchAndConvertDataset = async (
  queryClient: QueryClient,
  workspace: string,
  dataset: CustomizationTemplateDataset,
  onProgress: FetchDatasetProgress
): Promise<{ training: Blob; validation: Blob }> => {
  const total = dataset.trainingRowCount + dataset.validationRowCount;

  onProgress('locating');

  const rawRows = await queryClient.ensureQueryData(
    filesetRowsQueryOptions({
      workspace,
      filesetName: dataset.sourceFilesetName,
      pattern: dataset.filePattern,
      rowCount: total,
      onDownloadProgress: (loadedBytes, totalBytes) =>
        onProgress('downloading', loadedBytes, totalBytes),
    })
  );

  onProgress('converting');

  const convert = (raw: Record<string, unknown>[]): Record<string, unknown>[] =>
    raw
      .map((row) => dataset.convertRow(row))
      .filter((row): row is Record<string, unknown> => row !== null);

  const trainingRows = convert(rawRows.slice(0, dataset.trainingRowCount));
  const validationRows = convert(
    rawRows.slice(dataset.trainingRowCount, dataset.trainingRowCount + dataset.validationRowCount)
  );

  if (trainingRows.length < dataset.trainingRowCount) {
    throw new Error(
      `Not enough valid training rows: needed ${dataset.trainingRowCount}, found ${trainingRows.length}.`
    );
  }
  if (dataset.validationRowCount > 0 && validationRows.length < dataset.validationRowCount) {
    throw new Error(
      `Not enough valid validation rows: needed ${dataset.validationRowCount}, found ${validationRows.length}.`
    );
  }

  return { training: toJsonlBlob(trainingRows), validation: toJsonlBlob(validationRows) };
};
