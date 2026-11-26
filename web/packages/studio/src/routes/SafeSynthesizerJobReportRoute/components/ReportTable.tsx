// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { type ComponentProps, useMemo } from 'react';

type MakeColumns<Row> = ComponentProps<typeof StudioDataView<Row>>['makeColumns'];

interface ReportTableProps<Row> {
  rows: readonly Row[];
  makeColumns: MakeColumns<Row>;
  getRowId: (row: Row) => string;
  pageSize: number;
}

/**
 * A read-only, client-paged table. State stays local rather than in the URL, since the report
 * shows several tables on one route and URL-synced ones would share the same params.
 */
export const ReportTable = <Row,>({
  rows,
  makeColumns,
  getRowId,
  pageSize,
}: ReportTableProps<Row>) => {
  const dataViewState = DataView.useDataViewState({ pagination: { pageSize } });
  const { pageIndex, pageSize: currentPageSize } = dataViewState.pagination.state;
  const pageRows = useMemo(
    () => rows.slice(pageIndex * currentPageSize, (pageIndex + 1) * currentPageSize),
    [rows, pageIndex, currentPageSize]
  );
  return (
    <StudioDataView<Row>
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
