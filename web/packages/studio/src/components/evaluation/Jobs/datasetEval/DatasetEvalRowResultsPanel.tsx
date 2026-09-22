// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccordionPanel } from '@nemo/common/src/components/AccordionPanel';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import {
  TableExpandableCell,
  type TableExpandableCellState,
} from '@nemo/common/src/components/DataView/TableExpandableCell';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { Block, Button, Flex, Modal, Stack, Text } from '@nvidia/foundations-react-core';
import { MetricScoreChip } from '@studio/components/evaluation/MetricScoreChip';
import { Rows3 } from 'lucide-react';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

export interface DatasetEvalRow {
  row_index?: number;
  item?: Record<string, unknown>;
  sample?: { output_text?: string };
  metrics?: Record<string, { name?: string; value?: number | string }[]>;
  requests?: {
    request?: { messages?: { content?: string }[]; input_message?: string };
  }[];
}

interface DatasetEvalRowResultsPanelProps {
  rows: DatasetEvalRow[];
}

const EXPECTED_FIELDS = ['label', 'expected', 'reference', 'answer'];

const expectedValue = (item?: Record<string, unknown>): string | null => {
  if (!item) return null;
  for (const field of EXPECTED_FIELDS) {
    const value = item[field];
    if (typeof value === 'string' && value) return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  }
  return null;
};

const inputText = (row: DatasetEvalRow): string => {
  const request = row.requests?.[0]?.request;
  // `input_message` is the pre-chat-completions body; jobs submitted then still render.
  const rendered = request?.messages?.at(-1)?.content ?? request?.input_message;
  if (typeof rendered === 'string' && rendered) return rendered;
  return row.item ? JSON.stringify(row.item, null, 2) : '';
};

const scoreCells = (row: DatasetEvalRow): { label: string; value: number | string | undefined }[] =>
  Object.entries(row.metrics ?? {}).flatMap(([metricType, outputs]) =>
    (Array.isArray(outputs) ? outputs : []).map((output) => ({
      label: output?.name ? `${metricType}.${output.name}` : metricType,
      value: output?.value,
    }))
  );

const LongCell: FC<{
  content: string;
  title: string;
  onExpand: (state: TableExpandableCellState) => void;
}> = ({ content, title, onExpand }) =>
  content ? (
    <TableExpandableCell content={content} title={title} onExpand={onExpand} />
  ) : (
    <Text kind="body/regular/sm" color="secondary">
      —
    </Text>
  );

export const DatasetEvalRowResultsPanel: FC<DatasetEvalRowResultsPanelProps> = ({ rows }) => {
  const [expandedCell, setExpandedCell] = useState<TableExpandableCellState | null>(null);
  const dataViewState = useStudioDataViewState();

  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageRows = useMemo(
    () => rows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [rows, pageIndex, pageSize]
  );

  // The table only receives the current page, so a row's table index restarts
  // at 0 on every page; offset it to stay unique across pages.
  const absoluteIndex = useCallback(
    (relativeIndex: number) => pageIndex * pageSize + relativeIndex,
    [pageIndex, pageSize]
  );

  const makeColumns = useCallback<
    ComponentProps<typeof StudioDataView<DatasetEvalRow>>['makeColumns']
  >(
    (col) => [
      col.display({
        id: 'row',
        header: 'Row',
        size: 70,
        cell: ({ row }) => (
          <Text kind="body/semibold/sm">{row.original.row_index ?? absoluteIndex(row.index)}</Text>
        ),
      }),
      col.display({
        id: 'score',
        header: 'Score',
        size: 170,
        cell: ({ row }) => (
          <Stack gap="density-sm">
            {scoreCells(row.original).map((cell) => (
              <MetricScoreChip key={cell.label} label={cell.label} value={cell.value} />
            ))}
          </Stack>
        ),
      }),
      col.display({
        id: 'input',
        header: 'Input',
        size: 320,
        cell: ({ row }) => (
          <LongCell
            content={inputText(row.original)}
            title={`Row ${row.original.row_index ?? absoluteIndex(row.index)} — Input`}
            onExpand={setExpandedCell}
          />
        ),
      }),
      col.display({
        id: 'expected',
        header: 'Expected',
        size: 240,
        cell: ({ row }) => (
          <LongCell
            content={expectedValue(row.original.item) ?? ''}
            title={`Row ${row.original.row_index ?? absoluteIndex(row.index)} — Expected`}
            onExpand={setExpandedCell}
          />
        ),
      }),
      col.display({
        id: 'output',
        header: 'Agent Response',
        size: 320,
        cell: ({ row }) => (
          <LongCell
            content={row.original.sample?.output_text ?? ''}
            title={`Row ${row.original.row_index ?? absoluteIndex(row.index)} — Agent Response`}
            onExpand={setExpandedCell}
          />
        ),
      }),
    ],
    [absoluteIndex]
  );

  if (rows.length === 0) {
    return <Block className="text-subtle">No per-row results recorded for this evaluation.</Block>;
  }

  return (
    <AccordionPanel slotHeading={`Row Results (${rows.length})`} slotIcon={<Rows3 />}>
      <Stack className="min-h-[400px] max-h-[640px]">
        <StudioDataView
          dataViewState={dataViewState}
          makeColumns={makeColumns}
          maxTwoLines={false}
          attributes={{
            DataViewRoot: {
              data: pageRows,
              totalCount: rows.length,
              reactTableOptions: {
                getRowId: (row, relativeIndex) =>
                  String(row.row_index ?? absoluteIndex(relativeIndex)),
              },
            },
          }}
        />
      </Stack>
      <Modal
        open={expandedCell !== null}
        onOpenChange={(open) => {
          if (!open) setExpandedCell(null);
        }}
        slotHeading={expandedCell?.title ?? 'Cell Content'}
        className="w-[90vw] max-w-[1000px]"
        slotFooter={
          <Flex justify="end" align="center" className="w-full">
            <Button kind="tertiary" onClick={() => setExpandedCell(null)}>
              Close
            </Button>
          </Flex>
        }
      >
        <Block className="max-h-[60vh] overflow-auto whitespace-pre-wrap break-words">
          {expandedCell?.content}
        </Block>
      </Modal>
    </AccordionPanel>
  );
};
