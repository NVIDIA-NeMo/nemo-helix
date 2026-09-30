// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Accordion,
  Banner,
  CodeSnippet,
  Flex,
  Panel,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { AnonymizerRecordSkeleton } from '@studio/components/AnonymizerRecordView/AnonymizerRecordSkeleton';
import { AnonymizerRecordView } from '@studio/components/AnonymizerRecordView/AnonymizerRecordView';
import {
  buildAnonymizerRecord,
  outputColumn,
  REWRITTEN_SUFFIX,
} from '@studio/components/AnonymizerRecordView/parse';
import { RecordPager } from '@studio/routes/AnonymizerBuilderRoute/components/RecordPager';
import type { UseAnonymizerPreview } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerPreview';
import {
  OUTPUT_HEADING_REPLACED,
  OUTPUT_HEADING_REWRITTEN,
} from '@studio/routes/AnonymizerBuilderRoute/utils';
import { useMemo, useState, type FC, type ReactNode } from 'react';

interface PreviewPanelProps {
  readonly preview: UseAnonymizerPreview;
  /** Shown while loading, before a record reveals which output column was written. */
  readonly pendingOutputHeading: string;
  readonly slotActions: ReactNode;
}

export const PreviewPanel: FC<PreviewPanelProps> = ({
  preview,
  pendingOutputHeading,
  slotActions,
}) => {
  const { result, logs, isPreviewing, error, hasRun, wasStopped } = preview;
  const { records, textColumn, failedRecords } = result;
  const [recordIndex, setRecordIndex] = useState(0);
  const [pagedRecords, setPagedRecords] = useState(records);

  // Reset during render, not in an effect, so a new result never paints the old index first.
  if (pagedRecords !== records) {
    setPagedRecords(records);
    setRecordIndex(0);
  }

  const activeRow = records[recordIndex];
  const { record, outputHeading } = useMemo(
    () => ({
      record: activeRow ? buildAnonymizerRecord(activeRow, textColumn) : undefined,
      outputHeading: outputColumn(activeRow ?? {}, textColumn)?.endsWith(REWRITTEN_SUFFIX)
        ? OUTPUT_HEADING_REWRITTEN
        : OUTPUT_HEADING_REPLACED,
    }),
    [activeRow, textColumn]
  );

  // Rebuilt per streamed log frame otherwise, collapsed accordion or not.
  const logItems = useMemo(
    () => [
      {
        value: 'logs',
        slotTrigger: 'Logs',
        slotContent: (
          <CodeSnippet
            attributes={{ CodeSnippetCode: { className: 'max-h-[240px]' } }}
            kind="block"
            value={logs.join('\n')}
          />
        ),
      },
    ],
    [logs]
  );

  return (
    <Panel
      className="flex-1 h-full min-w-0"
      density="standard"
      elevation="high"
      attributes={{ PanelContent: { className: 'flex-1 min-h-0 overflow-auto' } }}
      slotHeading={
        <Flex align="center" className="w-full" gap="density-md" justify="between">
          <Text kind="label/bold/xl">Preview</Text>
          {records.length > 0 ? (
            <RecordPager index={recordIndex} onChange={setRecordIndex} total={records.length} />
          ) : null}
          {slotActions}
        </Flex>
      }
      slotFooter={logs.length > 0 ? <Accordion className="w-full" items={logItems} /> : null}
    >
      <Stack className="h-full" gap="density-lg">
        {error ? (
          <Banner kind="inline" status="error">
            {error}
          </Banner>
        ) : null}
        {failedRecords.length > 0 ? (
          <Banner kind="inline" status="warning">
            {failedRecords.length} record(s) failed during the preview run.
          </Banner>
        ) : null}
        {record ? (
          <AnonymizerRecordView outputHeading={outputHeading} record={record} />
        ) : isPreviewing ? (
          <AnonymizerRecordSkeleton outputHeading={pendingOutputHeading} />
        ) : error ? null : (
          <Flex align="center" className="flex-1" justify="center">
            <Text className="text-secondary" kind="body/regular/md">
              {wasStopped
                ? 'Preview stopped before any records arrived.'
                : hasRun
                  ? 'The preview run returned no records.'
                  : 'Your records preview will appear here'}
            </Text>
          </Flex>
        )}
      </Stack>
    </Panel>
  );
};
