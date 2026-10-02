// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { triggerDownload } from '@nemo/common/src/utils/file';
import { anonymizerDownloadRunJobResult } from '@nemo/sdk/generated/anonymizer/anonymizer';
import type { HelixJobResultResponse } from '@nemo/sdk/generated/anonymizer/schema';
import { Banner, Button, Flex, Panel, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { Download } from 'lucide-react';
import { useState, type FC } from 'react';

interface ResultsPanelProps {
  readonly workspace: string;
  readonly jobName: string;
  readonly results: readonly HelixJobResultResponse[];
  readonly isLoading: boolean;
  readonly isTerminal: boolean;
  readonly loadError: boolean;
}

export const ResultsPanel: FC<ResultsPanelProps> = ({
  workspace,
  jobName,
  results,
  isLoading,
  isTerminal,
  loadError,
}) => {
  const [downloading, setDownloading] = useState<string | null>(null);
  const [error, setError] = useState<string | undefined>(undefined);

  const download = async (name: string) => {
    setDownloading(name);
    setError(undefined);
    try {
      const blob = await anonymizerDownloadRunJobResult(workspace, jobName, name);
      triggerDownload(blob, name);
    } catch {
      setError(`Could not download ${name}`);
    } finally {
      setDownloading(null);
    }
  };

  return (
    <Panel slotHeading="Results" elevation="high" density="compact">
      <Stack gap="density-md">
        {error ? (
          <Banner kind="inline" status="error">
            {error}
          </Banner>
        ) : null}
        {loadError ? (
          <Banner kind="inline" status="error">
            Could not load results for this job.
          </Banner>
        ) : isLoading ? (
          <Spinner aria-label="Loading results" />
        ) : results.length ? (
          results.map((result) => (
            <Flex key={result.name} align="center" justify="between" gap="density-md">
              <Text kind="body/regular/md">{result.name}</Text>
              <Button
                kind="tertiary"
                size="small"
                disabled={downloading === result.name}
                onClick={() => download(result.name)}
              >
                <Download />
                Download
              </Button>
            </Flex>
          ))
        ) : (
          <Text kind="body/regular/md">
            {isTerminal ? 'This job produced no results.' : 'Results appear once the job finishes.'}
          </Text>
        )}
      </Stack>
    </Panel>
  );
};
