// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { findEvalConfigFile } from '@studio/components/evaluation/experimentEvalConfig';
import {
  type DatasetEvalSpec,
  isDatasetEvalSpec,
  parseEvalConfig,
} from '@studio/components/evaluation/submitEvaluationJob';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useQuery } from '@tanstack/react-query';

/**
 * The config stored in a fileset, loaded for reuse.
 *
 * Reuses the same chain the agent flow reads a config with -- `findEvalConfigFile`
 * accepts either serialization and reports "unreadable" apart from "absent", and
 * `parseEvalConfig` validates the shape rather than trusting the file.
 */
export type SavedConfig = ReturnType<typeof useSavedConfig>;

export function useSavedConfig(filesetName: string | null) {
  const workspace = useWorkspaceFromPath();

  const {
    data: spec,
    isFetching,
    error,
  } = useQuery({
    queryKey: ['evaluation-new', 'saved-config', workspace, filesetName],
    enabled: Boolean(filesetName),
    staleTime: Infinity,
    // A missing or malformed config is a dead end, not a flake.
    retry: false,
    queryFn: async (): Promise<DatasetEvalSpec> => {
      const signal = new AbortController().signal;
      const path = await findEvalConfigFile(workspace, filesetName as string, signal);
      if (path === undefined) throw new Error('Could not read that configuration.');
      if (path === null) throw new Error('That fileset has no eval config in it.');

      const blob = await filesDownloadFile(workspace, filesetName as string, path, signal);
      if (!blob) throw new Error('Could not download that configuration.');

      const parsed = parseEvalConfig(await blob.text());
      if (!isDatasetEvalSpec(parsed)) {
        throw new Error('That configuration is task-driven, so it cannot be run here.');
      }
      return parsed;
    },
  });

  return {
    spec: spec ?? null,
    isLoading: isFetching,
    error: error ? ((error as Error).message ?? 'Could not read that configuration.') : null,
  };
}
