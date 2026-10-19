// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { queryOptions } from '@tanstack/react-query';

export interface FileEncodingResult {
  ok: boolean;
}

interface DatasetFileEncodingParams {
  workspace: string;
  name: string;
  path: string;
}

/**
 * Strict UTF-8 validation for a fileset file. Customizer training decodes files
 * with Python's strict UTF-8, so we reject the same bytes pre-submit via
 * `TextDecoder('utf-8', { fatal: true })`. Fetches the file a second time;
 * folding this into the content query is a follow-up.
 */
export const datasetFileEncodingQueryOptions = ({
  workspace,
  name,
  path,
}: DatasetFileEncodingParams) =>
  queryOptions<FileEncodingResult>({
    staleTime: Infinity,
    queryKey: ['fileset-encoding-utf8', workspace, name, path],
    queryFn: async () => {
      const blob = await filesDownloadFile(workspace, name, path);
      if (!blob) {
        throw new Error('Invalid response while downloading file for encoding check');
      }
      const buffer = await blob.arrayBuffer();
      try {
        new TextDecoder('utf-8', { fatal: true }).decode(buffer);
        return { ok: true };
      } catch {
        // The browser's TextDecoder error string is implementation-specific
        // and not user-friendly ("Failed to execute 'decode' on 'TextDecoder'").
        // Drop it on the floor — the panel only needs to know pass/fail and
        // surfaces the offending file path itself.
        return { ok: false };
      }
    },
  });
