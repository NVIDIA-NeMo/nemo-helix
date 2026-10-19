// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import {
  useFilesListFilesetFiles,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import {
  Flex,
  PageHeader,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import { FilesetActionMenu } from '@studio/components/FilesetActionMenu';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { FilesetDetailTab, isFilesetDetailTab } from '@studio/routes/FilesetDetailRoute/constants';
import { FilesetCard } from '@studio/routes/FilesetDetailRoute/FilesetCard';
import { FilesTab } from '@studio/routes/FilesetDetailRoute/FilesTab';
import { getModelSource, isRootReadme } from '@studio/routes/FilesetDetailRoute/utils';
import { getWorkspaceFilesetsRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

export const FilesetDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const { filesetName } = useRequiredPathParams([ROUTE_PARAMS.filesetName]);
  const filesetId = getEntityReference({ namespace: workspace, name: filesetName });

  const { getQueryParam, setQueryParam } = useQueryParams();
  const tabFromUrl = getQueryParam(QUERY_PARAMETERS.tab) || undefined;

  useBreadcrumbs({
    items: [
      { href: getWorkspaceFilesetsRoute(workspace), slotLabel: 'Filesets' },
      { slotLabel: filesetName },
    ],
  });

  const queryEnabled = Boolean(workspace && filesetName);
  const {
    data: filesResponse,
    isPending: isFilesPending,
    isFetching: isFilesFetching,
    isError: isFilesError,
  } = useFilesListFilesetFiles(workspace, filesetName, undefined, {
    query: { enabled: queryEnabled },
  });
  const files = filesResponse?.data;

  const {
    data: fileset,
    isPending: isFilesetPending,
    isError: isFilesetError,
  } = useFilesRetrieveFileset(workspace, filesetName, {
    query: { enabled: queryEnabled },
  });

  const handleTabChange = (value: string) => {
    if (isFilesetDetailTab(value)) {
      setQueryParam(QUERY_PARAMETERS.tab, value);
    }
  };

  if (isFilesPending || isFilesetPending) {
    return <Loading description="Loading fileset..." />;
  }

  if (isFilesetError || !fileset) {
    return (
      <AccessibleTitle title={`Fileset ${filesetName}`}>
        <Stack className="w-full h-full min-h-0 p-density-2xl" gap="density-xl">
          <PageHeader slotHeading={filesetName} />
          <Text className="text-feedback-danger">Failed to load fileset.</Text>
        </Stack>
      </AccessibleTitle>
    );
  }

  const isDataset = fileset.purpose === FilesetPurpose.dataset;
  const isModel = fileset.purpose === FilesetPurpose.model;

  // Page/tab labelling follows the fileset's purpose so a single route reads
  // as "Model …" / "Dataset …" / "Fileset …" without separate components.
  const typeLabel = isDataset ? 'Dataset' : isModel ? 'Model' : 'Fileset';
  const cardLabel = isDataset ? 'Dataset Card' : isModel ? 'Model Card' : 'Card';

  // Datasets land on their (placeholder) card; everything else opens on the
  // README card when present and falls back to Files otherwise.
  const hasReadme = files?.some(isRootReadme) ?? false;
  const defaultTab = isDataset || hasReadme ? FilesetDetailTab.Card : FilesetDetailTab.Files;
  const currentTab: FilesetDetailTab = isFilesetDetailTab(tabFromUrl) ? tabFromUrl : defaultTab;

  // The source line (HF/NGC origin) is meaningful for model/generic filesets;
  // datasets don't surface it.
  const source = isDataset ? undefined : getModelSource(fileset);
  const descriptionText = <Text kind="body/regular/md">{fileset.description}</Text>;
  const description = isDataset ? (
    descriptionText
  ) : source ? (
    <Stack gap="1">
      <Flex gap="1" align="center">
        {creatorToIcon(source.creatorSlug, { className: 'w-4 h-4 flex-shrink-0' })}
        <span>{source.path}</span>
      </Flex>
      {descriptionText}
    </Stack>
  ) : undefined;

  return (
    <AccessibleTitle title={`${typeLabel} ${filesetName}`}>
      <Stack className="w-full min-h-full p-density-2xl" gap="density-xl">
        <PageHeader
          slotHeading={
            <Stack gap="density-sm">
              <Flex gap="density-sm" align="center" justify="between">
                <span>{filesetName}</span>
                <FilesetActionMenu
                  fileset={fileset}
                  onFilesetDeleted={() => navigate(getWorkspaceFilesetsRoute(workspace))}
                />
              </Flex>
            </Stack>
          }
          slotDescription={description}
        />
        <TabsRoot className="flex flex-col" value={currentTab} onValueChange={handleTabChange}>
          <TabsList>
            <TabsTrigger value={FilesetDetailTab.Card}>{cardLabel}</TabsTrigger>
            <TabsTrigger value={FilesetDetailTab.Files}>Files</TabsTrigger>
          </TabsList>

          <TabsContent value={FilesetDetailTab.Card} className="p-0">
            <FilesetCard
              workspace={workspace}
              filesetName={filesetName}
              fileset={fileset}
              files={files}
              isFilesError={isFilesError}
            />
          </TabsContent>

          <TabsContent value={FilesetDetailTab.Files} className="p-0">
            <FilesTab
              workspace={workspace}
              filesetName={filesetName}
              filesetId={filesetId}
              fileset={fileset}
              files={files}
              isFilesError={isFilesError}
              isFilesFetching={isFilesFetching}
            />
          </TabsContent>
        </TabsRoot>
      </Stack>
    </AccessibleTitle>
  );
};
