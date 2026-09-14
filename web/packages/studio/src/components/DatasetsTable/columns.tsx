// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import {
  FilesetPurpose,
  StorageConfigType,
  type FilesetOutput as Dataset,
} from '@nemo/sdk/generated/platform/schema';
import { type DropdownEntry, Flex, Text } from '@nvidia/foundations-react-core';
import { PURPOSE_LABELS } from '@studio/components/DatasetsTable/constants';
import { getStorageBackend, getStoragePath } from '@studio/components/DatasetsTable/helpers';
import {
  type DatasetWithId,
  type DatasetsTableProps,
  type ModalOpenState,
} from '@studio/components/DatasetsTable/types';
import { formatStorageBackendLabel } from '@studio/util/storageBackend';
import { Cloud, Database } from 'lucide-react';
import { type ComponentProps, type Dispatch, type SetStateAction } from 'react';

interface MakeDatasetsTableColumnsArgs {
  enableSelection: DatasetsTableProps['enableSelection'];
  selectionType: DatasetsTableProps['selectionType'];
  enableFilters: DatasetsTableProps['enableFilters'];
  enableActions: DatasetsTableProps['enableActions'];
  getDatasetRoute: DatasetsTableProps['getDatasetRoute'];
  renderRowActions: DatasetsTableProps['renderRowActions'];
  setModalDataset: Dispatch<SetStateAction<Dataset | undefined>>;
  setModalOpen: Dispatch<SetStateAction<ModalOpenState | undefined>>;
  handleDatasetDeleted: (deletedDataset: Dataset) => void;
}

export function makeDatasetsTableColumns({
  enableSelection,
  selectionType,
  enableFilters,
  enableActions,
  getDatasetRoute,
  renderRowActions,
  setModalDataset,
  setModalOpen,
  handleDatasetDeleted,
}: MakeDatasetsTableColumnsArgs): ComponentProps<
  typeof DataView.Root<DatasetWithId>
>['makeColumns'] {
  // Column definitions
  const makeColumns: ComponentProps<typeof DataView.Root<DatasetWithId>>['makeColumns'] = (
    { accessor },
    { rowSelectionColumn, rowActionsColumn }
  ) =>
    [
      enableSelection &&
        rowSelectionColumn({
          size: ROW_SELECTION_COLUMN_SIZE,
          ...(selectionType === 'single' && {
            headerProps: { className: 'invisible' },
          }),
        }),
      accessor('name', {
        header: 'Name',
        enableSorting: enableFilters,
        size: 175,
        cell({ row }) {
          const name = row.original?.name;
          return name ? <Text className="whitespace-normal break-all">{name}</Text> : null;
        },
      }),
      accessor((row) => getStorageBackend(row.storage), {
        id: 'storage_type',
        header: 'Storage Backend',
        size: 130,
        meta: {
          filter: {
            label: 'Storage Backend',
            type: 'single-select',

            options: [
              { value: '', label: 'All' },
              { value: StorageConfigType.local, label: 'Local' },
              { value: StorageConfigType.ngc, label: 'NGC' },
              { value: StorageConfigType.huggingface, label: 'Hugging Face' },
              { value: StorageConfigType.s3, label: 'S3' },
              { value: StorageConfigType.github, label: 'GitHub' },
            ],
          },
        },
        cell({ row }) {
          const backend = getStorageBackend(row.original?.storage);
          if (!backend) return null;
          const label = formatStorageBackendLabel(backend);
          const isLocal = backend === 'local';
          const Icon = isLocal ? Database : Cloud;
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
      accessor((row) => row.purpose, {
        id: 'purpose',
        header: 'Purpose',
        size: 110,
        meta: {
          filter: {
            label: 'Purpose',
            type: 'single-select',
            options: [
              { value: '', label: 'All' },
              { value: FilesetPurpose.generic, label: 'Generic' },
              { value: FilesetPurpose.dataset, label: 'Dataset' },
              { value: FilesetPurpose.environment, label: 'Environment' },
              { value: FilesetPurpose.model, label: 'Model' },
            ],
          },
        },
        cell({ row }) {
          const purpose = row.original?.purpose;
          return purpose ? <Text>{PURPOSE_LABELS[purpose] ?? purpose}</Text> : null;
        },
      }),
      accessor((row) => getStoragePath(row.storage), {
        id: 'path',
        header: 'Path',
        size: 200,
        cell({ row }) {
          const path = getStoragePath(row.original?.storage);
          return path ? (
            <Text className="whitespace-normal break-all" title={path}>
              {path}
            </Text>
          ) : null;
        },
      }),
      accessor('description', {
        header: 'Description',
        cell({ row }) {
          return (
            <Text className="truncate" title={row.original?.description}>
              {row.original?.description}
            </Text>
          );
        },
      }),
      accessor('created_at', {
        id: 'created_at',
        header: 'Created',
        enableSorting: enableFilters,
        size: 150,
        maxSize: 150,
        minSize: 150,
        meta: {
          filter: dateTimeFilter('Created At'),
        },
        cell({ row }) {
          return row.original?.created_at ? (
            <RelativeTime datetime={row.original.created_at} />
          ) : null;
        },
      }),
      enableActions &&
        rowActionsColumn({
          size: ROW_ACTIONS_COLUMN_SIZE,
          enableResizing: false,
          rowActions: (data: DatasetWithId): DropdownEntry[] => [
            ...(getDatasetRoute
              ? [
                  {
                    children: 'View',
                    onSelect: () => {
                      // Navigation handled by Link
                    },
                  },
                ]
              : []),
            {
              children: 'Edit',
              onSelect: () => {
                setModalDataset(data);
                setModalOpen('edit');
              },
            },
            {
              children: 'Delete',
              danger: true,
              onSelect: () => {
                setModalDataset(data);
                setModalOpen('delete');
              },
            },
          ],
          cell: renderRowActions
            ? ({ row }) => (
                <Flex justify="center" align="center">
                  {renderRowActions(row.original, {
                    onNavigate: () => {
                      /* handled by Link */
                    },
                    onEdit: () => {
                      setModalDataset(row.original);
                      setModalOpen('edit');
                    },
                    onDelete: () => {
                      setModalDataset(row.original);
                      setModalOpen('delete');
                    },
                    onDatasetDeleted: handleDatasetDeleted,
                  })}
                </Flex>
              )
            : undefined,
        }),
    ].filter((col): col is DataView.TanstackTable.ColumnDef<DatasetWithId> => Boolean(col));

  return makeColumns;
}
