// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { PageHeader, Stack } from '@nvidia/foundations-react-core';
import { DatasetsTable } from '@studio/components/DatasetsTable';
import { NewDatasetButton } from '@studio/components/NewDatasetButton';
import { FILESET_DETAILS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { ActionMenu } from '@studio/routes/FilesetListRoute/ActionMenu';
import {
  getFilesetDetailRoute,
  getFilesetDetailsRoute,
  getWorkspaceDatasetsRoute,
} from '@studio/routes/utils';
import { type FC, useCallback } from 'react';
import { useNavigate } from 'react-router';

export const DatasetListRoute: FC = () => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [{ href: getWorkspaceDatasetsRoute(workspace), slotLabel: 'Datasets' }],
  });

  const getDatasetRoute = useCallback(
    (dataset: FilesetOutput) =>
      FILESET_DETAILS_ENABLED
        ? getFilesetDetailRoute(workspace, dataset.name)
        : getFilesetDetailsRoute(workspace, getEntityReference(dataset)),
    [workspace]
  );

  const handleDatasetNav = useCallback(
    (dataset: FilesetOutput) => {
      navigate(getDatasetRoute(dataset));
    },
    [navigate, getDatasetRoute]
  );

  return (
    <AccessibleTitle title="Datasets">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Datasets"
          slotDescription="View and manage datasets for model training and evaluation data."
          slotActions={<NewDatasetButton color="brand" />}
        />
        <DatasetsTable
          variant="datasets"
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
    </AccessibleTitle>
  );
};
