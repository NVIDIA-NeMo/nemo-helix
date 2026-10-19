// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { useRowClick } from '@nemo/common/src/components/DataView/useRowClick';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { Stack, Tag, Text } from '@nvidia/foundations-react-core';
import {
  useInProgressCustomizations,
  type InProgressCustomization,
} from '@studio/hooks/useInProgressCustomizations';
import { getWorkspaceCustomizationJobDetailsRoute } from '@studio/routes/utils';
import { formatFinetuningType } from '@studio/util/customizations';
import { type ComponentProps, type FC } from 'react';
import { useNavigate } from 'react-router';

const makeColumns: ComponentProps<typeof DataView.Root<InProgressCustomization>>['makeColumns'] = ({
  accessor,
}) => [
  accessor('outputName', {
    header: 'Name',
    enableSorting: false,
    cell: ({ row }) => <Text className="truncate">{row.original.outputName}</Text>,
  }),
  accessor((row) => row.baseModel ?? '-', {
    id: 'baseModel',
    header: 'Base Model',
    enableSorting: false,
    cell: ({ row }) => <Text className="truncate">{row.original.baseModel || '-'}</Text>,
  }),
  accessor((row) => row.finetuningType ?? '', {
    id: 'kind',
    header: 'Kind',
    enableSorting: false,
    size: 180,
    cell: ({ row }) =>
      row.original.finetuningType ? (
        <Tag kind="solid" color="purple" density="compact">
          {formatFinetuningType(row.original.finetuningType)}
        </Tag>
      ) : (
        <Text>-</Text>
      ),
  }),
  accessor('createdAt', {
    header: 'Created',
    enableSorting: false,
    size: 150,
    cell: ({ row }) =>
      row.original.createdAt ? <RelativeTime datetime={row.original.createdAt} /> : <Text>-</Text>,
  }),
];

export interface InProgressCustomizationsTableProps {
  workspace: string;
}

/**
 * Fine-tuning runs that are still going, above the catalogue of finished ones.
 *
 * A run registers its model or adapter only in the last of its four steps, so until it
 * finishes there is nothing in the model list to show — and an empty list reads as "no
 * run was ever submitted" rather than "it is still training". These rows come from the
 * jobs themselves, which is the only evidence the output is on its way.
 *
 * The section is absent, not empty, when nothing is running: a permanent empty table
 * above the catalogue would be noise on every visit.
 */
export const InProgressCustomizationsTable: FC<InProgressCustomizationsTableProps> = ({
  workspace,
}) => {
  const navigate = useNavigate();
  const { runs, isError } = useInProgressCustomizations(workspace);
  const dataViewState = DataView.useDataViewState({});

  const { wrapColumns, onClick, onAuxClick, className } = useRowClick(
    (run: InProgressCustomization) =>
      navigate(getWorkspaceCustomizationJobDetailsRoute(workspace, run.jobName)),
    runs
  );

  // Nothing running, or the lookup failed: either way the catalogue below is the whole
  // story, and a broken-looking panel above it would not help.
  if (isError || runs.length === 0) return null;

  return (
    <Stack gap="density-md" className="min-w-0 shrink-0">
      <Text kind="label/bold/lg">In Progress</Text>
      <DataView.Root
        dataMode="auto"
        state={dataViewState}
        data={runs}
        makeColumns={wrapColumns(makeColumns)}
      >
        <div className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-base bg-surface-raised">
          <DataView.TableContent
            stickyTableHeader={false}
            onClick={onClick}
            onAuxClick={onAuxClick}
            className={`studio-data-view-table min-w-0 ${className}`}
          />
        </div>
      </DataView.Root>
    </Stack>
  );
};
