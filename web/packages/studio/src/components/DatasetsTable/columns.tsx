// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type * as DataView from '@nemo/common/src/components/DataView/internal';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { Cloud, Db } from '@nemo/icons/icons';
import type { FilesetOutput as Dataset } from '@nemo/sdk/generated/platform/schema';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { getStorageBackend } from '@studio/components/FilesetsTable/helpers';
import type { DatasetWithId } from '@studio/components/FilesetsTable/types';
import { formatStorageBackendLabel } from '@studio/util/storageBackend';
import type { ComponentProps, ReactNode } from 'react';

interface MakeDatasetsTableColumnsArgs {
  renderRowActions?: (dataset: Dataset) => ReactNode;
}

export function makeDatasetsTableColumns({
  renderRowActions,
}: MakeDatasetsTableColumnsArgs): ComponentProps<
  typeof DataView.Root<DatasetWithId>
>['makeColumns'] {
  const makeColumns: ComponentProps<typeof DataView.Root<DatasetWithId>>['makeColumns'] = (
    { accessor },
    { rowSelectionColumn, rowActionsColumn }
  ) =>
    [
      rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
      accessor('name', {
        header: 'Name',
        enableSorting: true,
        cell({ row }) {
          const name = row.original?.name;
          return name ? <Text className="whitespace-normal break-all">{name}</Text> : null;
        },
      }),
      accessor((row) => getStorageBackend(row.storage), {
        id: 'storage_type',
        header: 'Storage Type',
        enableSorting: true,
        size: 160,
        cell({ row }) {
          const backend = getStorageBackend(row.original?.storage);
          if (!backend) return null;
          const label = formatStorageBackendLabel(backend);
          const Icon = backend === 'local' ? Db : Cloud;
          return (
            <Flex align="center" gap="density-sm" className="min-w-0">
              <Icon className="flex-none text-fg-subdued" size="16" strokeWidth={0} />
              <Text className="truncate" title={label ?? undefined}>
                {label}
              </Text>
            </Flex>
          );
        },
      }),
      accessor('created_at', {
        header: 'Created',
        enableSorting: true,
        size: 150,
        maxSize: 150,
        minSize: 150,
        cell({ row }) {
          return row.original?.created_at ? (
            <RelativeTime datetime={row.original.created_at} />
          ) : null;
        },
      }),
      renderRowActions &&
        rowActionsColumn<DatasetWithId>({
          size: ROW_ACTIONS_COLUMN_SIZE,
          enableResizing: false,
          cell: ({ row }) => (
            <Flex justify="center" align="center">
              {renderRowActions(row.original)}
            </Flex>
          ),
        }),
    ].filter((col): col is DataView.TanstackTable.ColumnDef<DatasetWithId> => Boolean(col));

  return makeColumns;
}
