// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useDataDesignerListCreateJobResults } from '@nemo/sdk/generated/data-designer/data-designer';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { useDataDesignerJobFromRoute } from '@studio/routes/DataDesignerJobDetailsRoute/useDataDesignerJobFromRoute';
import { useMemo } from 'react';

/** Result name under which a job registers its output artifacts fileset. */
const ARTIFACTS_RESULT_NAME = 'artifacts';

/**
 * Resolves the output artifacts fileset for the current Data Designer job:
 * finds the `artifacts` result, parses its fileset location, and lists the
 * files inside it. Shared by the output-fileset section and the config panel
 * so both agree on which fileset (and files) belong to the job.
 */
export const useDataDesignerArtifactsFileset = () => {
  const { workspace, jobName, job } = useDataDesignerJobFromRoute();

  const isTerminal = job?.status != null && HelixJobTerminalStatuses.includes(job.status);

  const {
    data: resultsResponse,
    isLoading: isResultsLoading,
    isError: isResultsError,
    error: resultsError,
  } = useDataDesignerListCreateJobResults(workspace, jobName, {
    query: { refetchInterval: isTerminal ? false : 3000 },
  });

  const artifactsResult = useMemo(() => {
    const data = resultsResponse?.data;
    if (!data?.length) {
      return undefined;
    }
    const preferred = data.find((r) => r.name === ARTIFACTS_RESULT_NAME);
    if (preferred) {
      return preferred;
    }
    return data.find((r) => r.artifact_url && parseFilesetLocation(r.artifact_url, workspace));
  }, [resultsResponse?.data, workspace]);

  const filesetLoc = useMemo(
    () =>
      artifactsResult?.artifact_url
        ? parseFilesetLocation(artifactsResult.artifact_url, workspace)
        : null,
    [artifactsResult?.artifact_url, workspace]
  );

  const filesetWorkspace = filesetLoc?.workspace ?? '';
  const filesetName = filesetLoc?.name ?? '';
  const listFilesParams = useMemo(
    () => (filesetLoc?.filesListPathPrefix ? { path: filesetLoc.filesListPathPrefix } : undefined),
    [filesetLoc?.filesListPathPrefix]
  );

  const {
    data: listFilesResponse,
    isLoading: isFilesLoading,
    isError: isFilesError,
    error: filesError,
  } = useFilesListFilesetFiles(filesetWorkspace, filesetName, listFilesParams, {
    query: {
      enabled: Boolean(filesetWorkspace && filesetName),
    },
  });

  const files: FilesetFileOutput[] = useMemo(
    () => listFilesResponse?.data ?? [],
    [listFilesResponse?.data]
  );

  return {
    workspace,
    jobName,
    job,
    isTerminal,
    artifactsResult,
    filesetLoc,
    filesetWorkspace,
    filesetName,
    listFilesParams,
    files,
    isResultsLoading,
    isResultsError,
    resultsError,
    isFilesLoading,
    isFilesError,
    filesError,
  };
};
