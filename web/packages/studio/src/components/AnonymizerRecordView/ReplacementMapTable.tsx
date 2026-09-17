// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { Badge, Text } from '@nvidia/foundations-react-core';
import type { EntityReplacement } from '@studio/components/AnonymizerRecordView/types';
import { entityTagColor } from '@studio/routes/AnonymizerBuilderRoute/constants';
import { ArrowRight } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, type ComponentProps, type FC } from 'react';

const REPLACEMENTS_PAGE_SIZE = 20;
const ARROW_COLUMN_SIZE = 48;

interface ReplacementMapTableProps {
  readonly replacements: readonly EntityReplacement[];
}

export const ReplacementMapTable: FC<ReplacementMapTableProps> = memo(({ replacements }) => {
  const dataViewState = DataView.useDataViewState({
    pagination: { pageSize: REPLACEMENTS_PAGE_SIZE },
  });

  const { pageIndex: requestedPage, pageSize } = dataViewState.pagination.state;
  const lastPageIndex = Math.max(Math.ceil(replacements.length / pageSize) - 1, 0);
  const pageIndex = Math.min(requestedPage, lastPageIndex);

  const setPagination = dataViewState.pagination.set;
  useEffect(() => {
    if (requestedPage > lastPageIndex) {
      setPagination((prev) => ({ ...prev, pageIndex: lastPageIndex }));
    }
  }, [requestedPage, lastPageIndex, setPagination]);

  const pageRows = useMemo(
    () => replacements.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [replacements, pageIndex, pageSize]
  );

  const makeColumns = useCallback<
    ComponentProps<typeof StudioDataView<EntityReplacement>>['makeColumns']
  >(
    (col) => [
      col.accessor('label', {
        header: 'Label',
        cell: ({ getValue }) => (
          <Badge color={entityTagColor(getValue())} kind="outline">
            {getValue()}
          </Badge>
        ),
      }),
      col.accessor('original', {
        header: 'Original',
        cell: ({ getValue }) => <Text kind="body/regular/sm">{getValue()}</Text>,
      }),
      col.display({
        id: 'arrow',
        header: '',
        size: ARROW_COLUMN_SIZE,
        cell: () => <ArrowRight aria-hidden size={14} />,
      }),
      col.accessor('synthetic', {
        header: 'Replacement',
        cell: ({ getValue }) => <Text kind="body/regular/sm">{getValue()}</Text>,
      }),
    ],
    []
  );

  return (
    <StudioDataView<EntityReplacement>
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      maxTwoLines={false}
      attributes={{
        DataViewRoot: { data: pageRows, totalCount: replacements.length },
        DataViewPagination: { showWhileEmpty: false, showWhileLessThanPageSize: false },
      }}
    />
  );
});
