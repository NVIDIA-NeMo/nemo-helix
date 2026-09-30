// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccordionPanel } from '@nemo/common/src/components/AccordionPanel';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import {
  TableExpandableCell,
  type TableExpandableCellState,
} from '@nemo/common/src/components/DataView/TableExpandableCell';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { Badge, Block, Button, Flex, Modal, Stack, Text } from '@nvidia/foundations-react-core';
import type { AgentEvalTaskDetail } from '@studio/api/evaluation/agent-evaluations';
import { MetricScoreChip } from '@studio/components/evaluation/MetricScoreChip';
import { isScalar } from '@studio/util/functions';
import { ListChecks } from 'lucide-react';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

interface AgentEvalTaskResultsPanelProps {
  tasks: AgentEvalTaskDetail[];
}

/** Reference entries worth rendering as labelled chips: every value is a scalar and each
 *  is short enough to read inline. Anything else (step lists, nested objects) stays a
 *  JSON blob in the expandable cell. */
const scalarReferenceEntries = (reference?: Record<string, unknown>): [string, string][] | null => {
  const entries = Object.entries(reference ?? {});
  if (entries.length === 0) return null;
  if (!entries.every(([, v]) => isScalar(v) && String(v).length <= EXPECTED_CHIP_MAX_LENGTH)) {
    return null;
  }
  return entries.map(([k, v]) => [k, String(v)]);
};

const referenceText = (reference?: Record<string, unknown>): string | null => {
  if (!reference || Object.keys(reference).length === 0) return null;
  const values = Object.values(reference);
  const only = values[0];
  if (
    values.length === 1 &&
    (typeof only === 'string' || typeof only === 'number' || typeof only === 'boolean')
  ) {
    return String(only);
  }
  return JSON.stringify(reference, null, 2);
};

/** Longest expected value that still reads as a chip in this column. Anything
 *  longer (an ordered step list, a nested object) gets the expandable cell the
 *  Input and Agent Response columns use. */
const EXPECTED_CHIP_MAX_LENGTH = 32;

const diagnosticsText = (diagnostics: unknown[]): string =>
  diagnostics.map((d) => JSON.stringify(d)).join('\n');

const LongCell: FC<{
  content: string;
  title: string;
  onExpand: (state: TableExpandableCellState) => void;
}> = ({ content, title, onExpand }) =>
  content ? (
    <TableExpandableCell content={content} title={title} onExpand={onExpand} />
  ) : (
    <Text className="text-secondary" kind="body/regular/sm">
      —
    </Text>
  );

export const AgentEvalTaskResultsPanel: FC<AgentEvalTaskResultsPanelProps> = ({ tasks }) => {
  const [expandedCell, setExpandedCell] = useState<TableExpandableCellState | null>(null);
  const dataViewState = useStudioDataViewState();

  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageRows = useMemo(
    () => tasks.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [tasks, pageIndex, pageSize]
  );

  const makeColumns = useCallback<
    ComponentProps<typeof StudioDataView<AgentEvalTaskDetail>>['makeColumns']
  >(
    (col) => [
      col.display({
        id: 'task',
        header: 'Task',
        size: 80,
        cell: ({ row }) => (
          <Text kind="body/semibold/sm">{pageIndex * pageSize + row.index + 1}</Text>
        ),
      }),
      col.display({
        id: 'score',
        header: 'Score',
        size: 150,
        cell: ({ row }) => (
          <Stack gap="density-sm">
            {row.original.scores.map((s) => (
              <MetricScoreChip key={s.name} label={s.name} value={s.value} />
            ))}
          </Stack>
        ),
      }),
      col.display({
        id: 'input',
        header: 'Input',
        size: 280,
        cell: ({ row }) => (
          <LongCell
            content={row.original.instruction ?? ''}
            title={`Task ${pageIndex * pageSize + row.index + 1} — Input`}
            onExpand={setExpandedCell}
          />
        ),
      }),
      col.display({
        id: 'expected',
        header: 'Expected',
        size: 140,
        cell: ({ row }) => {
          const expected = referenceText(row.original.reference);
          if (!expected) {
            return (
              <Text className="text-secondary" kind="body/regular/sm">
                —
              </Text>
            );
          }
          const entries = scalarReferenceEntries(row.original.reference);
          if (entries) {
            return (
              <Stack gap="density-sm">
                {entries.map(([key, val]) => (
                  <Stack key={key} gap="density-xs" className="min-w-0">
                    {entries.length > 1 && (
                      <Text kind="body/regular/sm" className="truncate text-secondary">
                        {key}
                      </Text>
                    )}
                    <Badge kind="outline" color="gray">
                      {val}
                    </Badge>
                  </Stack>
                ))}
              </Stack>
            );
          }
          return expected.length <= EXPECTED_CHIP_MAX_LENGTH ? (
            <Badge kind="outline" color="gray">
              {expected}
            </Badge>
          ) : (
            <LongCell
              content={expected}
              title={`Task ${pageIndex * pageSize + row.index + 1} — Expected`}
              onExpand={setExpandedCell}
            />
          );
        },
      }),
      col.display({
        id: 'response',
        header: 'Agent Response',
        size: 280,
        cell: ({ row }) => (
          <LongCell
            content={row.original.responseText ?? ''}
            title={`Task ${pageIndex * pageSize + row.index + 1} — Agent Response`}
            onExpand={setExpandedCell}
          />
        ),
      }),
      col.display({
        id: 'status',
        header: 'Status',
        size: 110,
        cell: ({ row }) => <StatusBadge status={row.original.status} />,
      }),
      col.display({
        id: 'diagnostics',
        header: 'Diagnostics',
        size: 220,
        cell: ({ row }) => (
          <LongCell
            content={diagnosticsText(row.original.diagnostics)}
            title={`Task ${pageIndex * pageSize + row.index + 1} — Diagnostics`}
            onExpand={setExpandedCell}
          />
        ),
      }),
    ],
    [pageIndex, pageSize]
  );

  if (tasks.length === 0) {
    return <Block className="text-subtle">No per-task results recorded for this evaluation.</Block>;
  }

  return (
    <AccordionPanel slotHeading={`Task Results (${tasks.length})`} slotIcon={<ListChecks />}>
      <div className="flex flex-col min-h-[400px] max-h-[640px]">
        <StudioDataView
          dataViewState={dataViewState}
          makeColumns={makeColumns}
          maxTwoLines={false}
          attributes={{
            DataViewRoot: {
              data: pageRows,
              totalCount: tasks.length,
              reactTableOptions: { getRowId: (row) => row.taskId },
            },
          }}
        />
      </div>
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
        <div className="max-h-[70vh] overflow-auto">
          <Text kind="body/regular/md" className="whitespace-pre-wrap">
            {expandedCell?.content}
          </Text>
        </div>
      </Modal>
    </AccordionPanel>
  );
};
