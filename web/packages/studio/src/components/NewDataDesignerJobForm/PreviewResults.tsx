// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import {
  TableExpandableCell,
  type TableExpandableCellState,
} from '@nemo/common/src/components/DataView/TableExpandableCell';
import { formatJsonlSampleCellValue } from '@nemo/common/src/utils/parseJsonlObjectSample';
import { Banner, Button, Flex, Modal, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import {
  orderPreviewColumns,
  type PreviewRecord,
} from '@studio/components/NewDataDesignerJobForm/previewApi';
import {
  type ComponentProps,
  type FC,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';

const PREVIEW_PAGE_SIZE = 10;
const INLINE_CELL_MAX_LENGTH = 60;

interface PreviewRow {
  rowNumber: number;
  values: PreviewRecord;
}

const PreviewCell: FC<{
  value: unknown;
  title: string;
  onExpand: (state: TableExpandableCellState) => void;
}> = ({ value, title, onExpand }) => {
  const content = formatJsonlSampleCellValue(value);
  if (!content) {
    return (
      <Text kind="body/regular/sm" className="text-secondary">
        —
      </Text>
    );
  }
  if (content.length > INLINE_CELL_MAX_LENGTH || content.includes('\n')) {
    return <TableExpandableCell content={content} title={title} onExpand={onExpand} />;
  }
  return <Text kind="body/regular/sm">{content}</Text>;
};

const PreviewRecordsTable: FC<{
  rows: PreviewRow[];
  columnKeys: string[];
  onExpand: (state: TableExpandableCellState) => void;
}> = memo(function PreviewRecordsTable({ rows, columnKeys, onExpand }) {
  const dataViewState = DataView.useDataViewState({
    pagination: { pageSize: PREVIEW_PAGE_SIZE },
  });
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const setPagination = dataViewState.pagination.set;

  useEffect(() => {
    setPagination((prev) => (prev.pageIndex === 0 ? prev : { ...prev, pageIndex: 0 }));
  }, [rows, setPagination]);

  const pageRows = useMemo(
    () => rows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [rows, pageIndex, pageSize]
  );

  const makeColumns = useCallback<ComponentProps<typeof StudioDataView<PreviewRow>>['makeColumns']>(
    (col) => [
      col.accessor('rowNumber', {
        header: '#',
        size: 56,
        cell: ({ getValue }) => (
          <Text kind="body/regular/sm" className="tabular-nums text-secondary">
            {getValue()}
          </Text>
        ),
      }),
      ...columnKeys.map((key) =>
        col.display({
          id: `preview-column-${key}`,
          header: key,
          size: 240,
          cell: ({ row }) => (
            <PreviewCell
              value={row.original.values[key]}
              title={`Row ${row.original.rowNumber} — ${key}`}
              onExpand={onExpand}
            />
          ),
        })
      ),
    ],
    [columnKeys, onExpand]
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <StudioDataView<PreviewRow>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        maxTwoLines={false}
        attributes={{
          DataViewRoot: {
            data: pageRows,
            totalCount: rows.length,
            reactTableOptions: { getRowId: (row) => String(row.rowNumber) },
          },
          DataViewPagination: { showWhileEmpty: false, showWhileLessThanPageSize: false },
        }}
      />
    </div>
  );
});

export interface PreviewResultsProps {
  records: PreviewRecord[] | undefined;
  /** Column names in config order; record keys not listed here follow in stream order. */
  columnOrder: readonly string[];
  isPreviewing: boolean;
  error: string | undefined;
}

export const PreviewResults: FC<PreviewResultsProps> = ({
  records,
  columnOrder,
  isPreviewing,
  error,
}) => {
  const [expandedCell, setExpandedCell] = useState<TableExpandableCellState | null>(null);

  const rows = useMemo(
    () => (records ?? []).map((values, index) => ({ rowNumber: index + 1, values })),
    [records]
  );
  const columnKeys = useMemo(
    () => orderPreviewColumns(records ?? [], columnOrder),
    [records, columnOrder]
  );

  const errorBanner = error ? (
    <Banner kind="inline" status="error">
      The preview failed: {error}
    </Banner>
  ) : null;

  if (rows.length === 0) {
    if (isPreviewing) {
      return (
        <Flex align="center" justify="center" className="py-density-2xl">
          <Spinner description="Generating preview rows…" />
        </Flex>
      );
    }
    return (
      errorBanner ?? (
        <Text kind="body/regular/sm" className="text-secondary">
          The preview didn&apos;t return any rows. Check the preview logs for details.
        </Text>
      )
    );
  }

  return (
    <Stack gap="density-sm" className="min-h-0 flex-1">
      {errorBanner}
      <PreviewRecordsTable rows={rows} columnKeys={columnKeys} onExpand={setExpandedCell} />
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
    </Stack>
  );
};
