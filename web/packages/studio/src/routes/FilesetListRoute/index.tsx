// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { PageHeader, Stack } from '@nvidia/foundations-react-core';
import { FilesetsTable } from '@studio/components/FilesetsTable';
import { NewModelFilesetButton } from '@studio/components/NewModelFilesetButton';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { ActionMenu } from '@studio/routes/FilesetListRoute/ActionMenu';
import { getFilesetDetailRoute, getWorkspaceFilesetsRoute } from '@studio/routes/utils';
import { type FC, useCallback } from 'react';
import { Outlet, useNavigate } from 'react-router';

export const FilesetListRoute: FC = () => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [{ href: getWorkspaceFilesetsRoute(workspace), slotLabel: 'Filesets' }],
  });

  const getDatasetRoute = useCallback(
    (dataset: FilesetOutput) => getFilesetDetailRoute(workspace, dataset.name),
    [workspace]
  );

  const handleDatasetNav = useCallback(
    (dataset: FilesetOutput) => {
      navigate(getDatasetRoute(dataset));
    },
    [navigate, getDatasetRoute]
  );

  return (
    <AccessibleTitle title="Filesets">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Filesets"
          slotDescription="Filesets organize files by purpose — Generic, Dataset, or Model. Purpose determines which metadata fields are available and can't be changed after creation. Use Dataset for training and evaluation data, Model for model weights and checkpoints, and Generic for everything else."
          slotActions={<NewModelFilesetButton color="brand" />}
        />
        <FilesetsTable
          enableFilters
          enableBulkDelete
          enableSelection
          getDatasetRoute={getDatasetRoute}
          renderRowActions={(dataset, { onDatasetDeleted }) => (
            <ActionMenu
              entityLabel="Fileset"
              dataset={dataset}
              onNavigateToDetails={handleDatasetNav}
              onDatasetDeleted={onDatasetDeleted}
            />
          )}
        />
      </Stack>
      <Outlet />
    </AccessibleTitle>
  );
};
