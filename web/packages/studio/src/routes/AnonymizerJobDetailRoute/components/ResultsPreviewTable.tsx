// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { Button, Text } from '@nvidia/foundations-react-core';
import {
  parseReplacements,
  REPLACEMENT_MAP_COLUMN,
} from '@studio/components/AnonymizerRecordView/parse';
import type { DataFileRow } from '@studio/components/FileRowEditor/types';
import {
  RESULT_PREVIEW_ROWS,
  resolveTextColumn,
} from '@studio/routes/AnonymizerJobDetailRoute/util';
import { memo, useCallback, useMemo, type ComponentProps, type FC } from 'react';

interface ResultsPreviewTableProps {
  readonly rows: readonly DataFileRow[];
  readonly textColumn: string | undefined;
  readonly onRowClick: (row: DataFileRow, indexInAllRows: number) => void;
}

const cellText = (value: unknown): string =>
  typeof value === 'object' ? JSON.stringify(value) : String(value);

const describeRow = (row: DataFileRow, textColumn: string | undefined): string => {
  const value = row[resolveTextColumn(row, textColumn)];
  return value == null ? '' : cellText(value);
};

const replacementCount = (row: DataFileRow): number =>
  parseReplacements(row[REPLACEMENT_MAP_COLUMN]).length;

interface PreviewRow {
  readonly source: DataFileRow;
  readonly index: number;
  readonly text: string;
  readonly count: number;
}

export const ResultsPreviewTable: FC<ResultsPreviewTableProps> = memo(
  ({ rows, textColumn, onRowClick }) => {
    const dataViewState = useStudioDataViewState({
      defaultPageSize: RESULT_PREVIEW_ROWS,
      columnPinning: {},
    });

    const previewRows = useMemo<PreviewRow[]>(
      () =>
        rows.map((source, index) => ({
          source,
          index,
          text: describeRow(source, textColumn),
          count: replacementCount(source),
        })),
      [rows, textColumn]
    );

    const { pageIndex, pageSize } = dataViewState.pagination.state;
    const sortingState = dataViewState.sorting.state;
    const sortedRows = useMemo(() => {
      const [sort] = sortingState;
      if (!sort) return previewRows;
      const compare = (a: PreviewRow, b: PreviewRow): number =>
        sort.id === 'count' ? a.count - b.count : a.text.localeCompare(b.text);
      return [...previewRows].sort((a, b) => (sort.desc ? -compare(a, b) : compare(a, b)));
    }, [previewRows, sortingState]);

    const pageRows = useMemo(
      () => sortedRows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
      [sortedRows, pageIndex, pageSize]
    );

    const makeColumns = useCallback<
      ComponentProps<typeof StudioDataView<PreviewRow>>['makeColumns']
    >(
      (col) => [
        col.accessor('text', {
          header: 'Record',
          cell: ({ getValue }) => <Text kind="body/regular/sm">{getValue()}</Text>,
        }),
        col.accessor('count', {
          header: 'Count',
          size: 100,
          enableResizing: false,
          cell: ({ getValue }) => <Text kind="body/regular/sm">{getValue()}</Text>,
        }),
        col.display({
          id: 'details',
          header: '',
          size: 100,
          enableResizing: false,
          cell: ({ row }) => (
            <Button
              kind="tertiary"
              onClick={() => onRowClick(row.original.source, row.original.index)}
            >
              Details
            </Button>
          ),
        }),
      ],
      [onRowClick]
    );

    const handleRowClick = useCallback(
      (row: PreviewRow) => onRowClick(row.source, row.index),
      [onRowClick]
    );

    return (
      <div className="flex flex-col min-h-[400px] max-h-[640px]">
        <StudioDataView<PreviewRow>
          dataViewState={dataViewState}
          makeColumns={makeColumns}
          onRowClick={handleRowClick}
          attributes={{ DataViewRoot: { data: pageRows, totalCount: rows.length } }}
        />
      </div>
    );
  }
);

ResultsPreviewTable.displayName = 'ResultsPreviewTable';
