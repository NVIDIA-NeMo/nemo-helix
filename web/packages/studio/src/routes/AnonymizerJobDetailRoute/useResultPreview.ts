// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { parseDataFile } from '@studio/components/FileRowEditor/parse';
import type { DataFileRow } from '@studio/components/FileRowEditor/types';
import {
  metadataTextColumn,
  parseArtifactUrl,
  RESULT_PREVIEW_ROWS,
} from '@studio/routes/AnonymizerJobDetailRoute/util';
import { useMemo } from 'react';

export interface ResultPreview {
  readonly rows: DataFileRow[];
  readonly textColumn: string | undefined;
  readonly isLoading: boolean;
  readonly error: Error | null;
}

export const useResultPreview = (
  workspace: string,
  artifactUrl: string | undefined
): ResultPreview => {
  const location = parseArtifactUrl(artifactUrl);
  const enabled = !!location;

  const { data: metadata, isLoading: metadataLoading } = useDatasetFileContent({
    workspace,
    name: location?.fileset ?? '',
    path: `${location?.basePath}/metadata.json`,
    enabled,
  });

  const {
    data: trace,
    isLoading: traceLoading,
    error,
  } = useDatasetFileContent({
    workspace,
    name: location?.fileset ?? '',
    path: `${location?.basePath}/trace.parquet`,
    range: [0, RESULT_PREVIEW_ROWS],
    enabled,
  });

  const rows = useMemo<DataFileRow[]>(() => {
    if (!trace) return [];
    try {
      return parseDataFile(trace, 'jsonl');
    } catch {
      return [];
    }
  }, [trace]);

  const textColumn = useMemo(() => metadataTextColumn(metadata), [metadata]);

  return { rows, textColumn, isLoading: traceLoading || metadataLoading, error };
};
