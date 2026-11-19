// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { type ComponentProps, useMemo } from 'react';

const PAGE_SIZE = 20;

export type MakeColumns<T> = ComponentProps<typeof StudioDataView<T>>['makeColumns'];

interface StaticDataViewProps<T> {
  rows: readonly T[];
  makeColumns: MakeColumns<T>;
  getRowId: (row: T) => string;
}

/**
 * A read-only table over rows already in memory. Pagination lives in component state rather than
 * the URL, so several of these can share a route.
 */
export const StaticDataView = <T,>({ rows, makeColumns, getRowId }: StaticDataViewProps<T>) => {
  const dataViewState = DataView.useDataViewState({ pagination: { pageSize: PAGE_SIZE } });
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageRows = useMemo(
    () => rows.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [rows, pageIndex, pageSize]
  );

  return (
    <StudioDataView<T>
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      maxTwoLines={false}
      attributes={{
        DataViewRoot: {
          data: pageRows,
          totalCount: rows.length,
          reactTableOptions: { getRowId },
        },
        DataViewPagination: { showWhileEmpty: false, showWhileLessThanPageSize: false },
      }}
    />
  );
};
