// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Panel, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import type { DataFileRow } from '@studio/components/FileRowEditor/types';
import { RecordDetailModal } from '@studio/routes/AnonymizerJobDetailRoute/components/RecordDetailModal';
import { ResultsPreviewTable } from '@studio/routes/AnonymizerJobDetailRoute/components/ResultsPreviewTable';
import { useResultPreview } from '@studio/routes/AnonymizerJobDetailRoute/useResultPreview';
import { useCallback, useEffect, useState, type FC } from 'react';

interface ResultsPreviewPanelProps {
  readonly workspace: string;
  readonly artifactUrl: string | undefined;
}

export const ResultsPreviewPanel: FC<ResultsPreviewPanelProps> = ({ workspace, artifactUrl }) => {
  const { rows, textColumn, isLoading, error } = useResultPreview(workspace, artifactUrl);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  useEffect(() => setSelectedIndex(null), [workspace, artifactUrl]);
  const closeSelectedRow = useCallback(() => setSelectedIndex(null), []);
  const selectRow = useCallback(
    (_row: DataFileRow, indexInAllRows: number) => setSelectedIndex(indexInAllRows),
    []
  );

  return (
    <Panel slotHeading="Preview" elevation="high" density="compact">
      {error ? (
        <Banner kind="inline" status="error">
          Could not load the result preview.
        </Banner>
      ) : isLoading ? (
        <Spinner aria-label="Loading preview" />
      ) : rows.length ? (
        <Stack gap="density-md">
          <ResultsPreviewTable rows={rows} textColumn={textColumn} onRowClick={selectRow} />
          <Text kind="body/regular/sm">
            Showing the first {rows.length} records. Download the result for the full dataset.
            Select a row to view it in the record preview.
          </Text>
        </Stack>
      ) : (
        <Text kind="body/regular/md">No preview available for this job.</Text>
      )}
      <RecordDetailModal
        rows={rows}
        selectedIndex={selectedIndex}
        textColumn={textColumn}
        onIndexChange={setSelectedIndex}
        onClose={closeSelectedRow}
      />
    </Panel>
  );
};
