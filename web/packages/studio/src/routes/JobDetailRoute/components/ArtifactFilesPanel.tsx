// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { HelixJobStatus, type HelixJobResultResponse } from '@nemo/sdk/generated/platform/schema';
import { Flex, Spinner, Stack } from '@nvidia/foundations-react-core';
import { FilesetFilePreviewPanel } from '@studio/components/FilesetFilePreviewPanel';
import {
  ArtifactItemRows,
  type ArtifactPreviewState,
} from '@studio/routes/JobDetailRoute/components/ArtifactItemRows';
import { resolveArtifactItems } from '@studio/routes/JobDetailRoute/utils';
import { useMemo, useState, type FC } from 'react';

const ERROR_STATUSES: HelixJobStatus[] = [HelixJobStatus.error, HelixJobStatus.cancelled];

const emptyArtifactStateCopy = (
  jobStatus: HelixJobStatus | undefined
): { header: string; emptyMessage: string } => {
  if (!jobStatus || !HelixJobTerminalStatuses.includes(jobStatus)) {
    return {
      header: 'No artifacts yet',
      emptyMessage: 'Artifacts will appear once the job is complete.',
    };
  }
  if (ERROR_STATUSES.includes(jobStatus)) {
    return {
      header: 'No artifacts',
      emptyMessage: 'No artifacts loaded due to a job error.',
    };
  }
  return {
    header: 'No artifacts',
    emptyMessage: 'This job did not produce any artifacts.',
  };
};

export interface ArtifactFilesPanelProps {
  workspace: string;
  results: ReadonlyArray<HelixJobResultResponse>;
  isLoading: boolean;
  jobStatus?: HelixJobStatus;
}

/**
 * Renders the Artifacts panel for a generic job. For each registered job
 * result, renders one row if the result is a file, or N rows (one per child)
 * if the result is a directory. Clicking a row opens the standard dataset
 * file preview panel — same UX as the Data Designer detail page.
 */
export const ArtifactFilesPanel: FC<ArtifactFilesPanelProps> = ({
  workspace,
  results,
  isLoading,
  jobStatus,
}) => {
  const [preview, setPreview] = useState<ArtifactPreviewState | null>(null);
  const items = useMemo(() => resolveArtifactItems(results, workspace), [results, workspace]);

  if (isLoading && results.length === 0) {
    return (
      <Flex justify="center" align="center" className="min-h-[200px] w-full">
        <Spinner size="medium" aria-label="Loading artifacts..." />
      </Flex>
    );
  }
  if (items.length === 0) {
    const { header, emptyMessage } = emptyArtifactStateCopy(jobStatus);
    return (
      <Flex justify="center" align="center" className="h-full min-h-[200px] w-full">
        <TableEmptyState header={header} emptyMessage={emptyMessage} />
      </Flex>
    );
  }

  return (
    <>
      <Stack gap="density-md">
        {items.map((item) => (
          <ArtifactItemRows
            key={`${item.resultName}|${item.workspace}/${item.fileset}#${item.objectPath}`}
            item={item}
            onPreview={setPreview}
          />
        ))}
      </Stack>

      <FilesetFilePreviewPanel
        open={preview != null}
        onCloseClick={() => setPreview(null)}
        workspace={preview?.workspace ?? ''}
        filesetName={preview?.fileset ?? ''}
        filePath={preview?.file.path ?? ''}
        file={preview?.file ?? undefined}
      />
    </>
  );
};
