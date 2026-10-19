// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEntityReference } from '@nemo/common/src/namedEntity';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import {
  FilesetPurpose,
  type FilesetFileOutput,
  type FilesetOutput,
  type ModelEntityFilter,
} from '@nemo/sdk/generated/platform/schema';
import { Stack } from '@nvidia/foundations-react-core';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { DatasetSamplePanel } from '@studio/routes/FilesetDetailRoute/FilesetCard/DatasetSamplePanel';
import { ReadmeBody } from '@studio/routes/FilesetDetailRoute/FilesetCard/ReadmeBody';
import { FilesetMetadataPanel } from '@studio/routes/FilesetDetailRoute/FilesetMetadataPanel';
import { isRootReadme, parseReadme } from '@studio/routes/FilesetDetailRoute/utils';
import { useMemo, type FC } from 'react';

export interface FilesetCardProps {
  workspace: string;
  filesetName: string;
  fileset: FilesetOutput;
  files: FilesetFileOutput[] | undefined;
  isFilesError: boolean;
}

/**
 * Purpose-agnostic card for a fileset detail page: renders the root README as
 * markdown alongside a metadata panel. Used for every fileset purpose — the
 * panel's README-frontmatter "Details" section simply collapses when those
 * fields (license, tags, base model, …) aren't present.
 */
export const FilesetCard: FC<FilesetCardProps> = ({
  workspace,
  filesetName,
  fileset,
  files,
  isFilesError,
}) => {
  const readmePath = useMemo(() => files?.find(isRootReadme)?.path, [files]);
  const isModel = fileset.purpose === FilesetPurpose.model;

  const { data: modelEntitiesResponse } = useModelsListModels(
    workspace,
    // fileset filter is valid on the backend but missing from the generated SDK type (stale SDK)
    {
      filter: {
        fileset: getEntityReference({ workspace, name: filesetName }),
      } as ModelEntityFilter,
    },
    { query: { enabled: isModel } }
  );
  const modelEntities = modelEntitiesResponse?.data ?? [];

  const {
    data: rawContent,
    isLoading: isContentLoading,
    isError: isContentError,
  } = useDatasetFileContent({
    workspace,
    name: filesetName,
    path: readmePath ?? '',
    enabled: Boolean(readmePath),
  });

  const parsed = useMemo(
    () => (rawContent !== undefined ? parseReadme(rawContent) : undefined),
    [rawContent]
  );

  const isDataset = fileset.purpose === FilesetPurpose.dataset;
  const hasReadme = Boolean(readmePath);
  const metaPanels = (
    <Stack gap="density-xl" className="h-full overflow-auto">
      <FilesetMetadataPanel
        fileset={fileset}
        readmeMetadata={parsed?.metadata}
        modelEntities={isModel ? modelEntities : undefined}
      />
      {isDataset && (
        <DatasetSamplePanel workspace={workspace} filesetName={filesetName} files={files} />
      )}
    </Stack>
  );
  if (!hasReadme) {
    return <div data-testid="fileset-card">{metaPanels}</div>;
  }

  return (
    <div
      className="grid w-full grid-cols-1 gap-density-xl pt-density-xl lg:grid-cols-3"
      data-testid="fileset-card"
    >
      <div className="lg:col-span-2">
        <ReadmeBody
          isFilesError={isFilesError}
          readmePath={readmePath}
          isContentLoading={isContentLoading}
          isContentError={isContentError}
          content={parsed?.content}
        />
      </div>
      <div className="lg:col-span-1">{metaPanels}</div>
    </div>
  );
};
