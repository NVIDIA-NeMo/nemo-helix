// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { FilesetsTable } from '@studio/components/FilesetsTable';
import { NewModelFilesetButton } from '@studio/components/NewModelFilesetButton';
import { FILESET_DETAILS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { ActionMenu } from '@studio/routes/FilesetListRoute/ActionMenu';
import { PanelManagement } from '@studio/routes/FilesetListRoute/PanelManagement';
import {
  getFilesetDetailRoute,
  getFilesetDetailsRoute,
  getNewFilesetRoute,
  getWorkspaceFilesetsRoute,
} from '@studio/routes/utils';
import { type FC, useCallback } from 'react';
import { Link, Outlet, useNavigate } from 'react-router';

export const FilesetListRoute: FC = () => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [{ href: getWorkspaceFilesetsRoute(workspace), slotLabel: 'Filesets' }],
  });

  const getDatasetRoute = useCallback(
    (dataset: FilesetOutput) => {
      if (
        FILESET_DETAILS_ENABLED &&
        (dataset.purpose === 'dataset' || dataset.purpose === 'model')
      ) {
        return getFilesetDetailRoute(workspace, dataset.name);
      }
      return getFilesetDetailsRoute(workspace, getEntityReference(dataset));
    },
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
          slotActions={
            FILESET_DETAILS_ENABLED ? (
              <NewModelFilesetButton color="brand" />
            ) : (
              <Button asChild color="brand">
                <Link to={getNewFilesetRoute(workspace)}>Create Fileset</Link>
              </Button>
            )
          }
        />
        <FilesetsTable
          enableFilters
          enableBulkDelete
          enableSelection
          getDatasetRoute={getDatasetRoute}
          renderRowActions={(dataset, { onDatasetDeleted }) => (
            <ActionMenu
              dataset={dataset}
              onNavigateToDetails={handleDatasetNav}
              onDatasetDeleted={onDatasetDeleted}
            />
          )}
        />
      </Stack>
      <Outlet />
      <PanelManagement workspace={workspace} />
    </AccessibleTitle>
  );
};
