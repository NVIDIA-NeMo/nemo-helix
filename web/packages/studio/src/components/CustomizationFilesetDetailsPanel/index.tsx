// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { formatFileSize } from '@nemo/common/src/components/UploadModal/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useFilesRetrieveFileset as useGetDataset } from '@nemo/sdk/generated/platform/files';
import {
  Button,
  CodeSnippet,
  Grid,
  Panel,
  SidePanel,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { CUSTOMIZATION_TYPE_BADGES, CustomizationFileType } from '@studio/constants/customization';
import { parseFilesetUri, useCustomizationFilesAsRows } from '@studio/hooks/useCustomizationFiles';
import { Database, Eye } from 'lucide-react';
import { ComponentProps, useState } from 'react';

type Props = {
  filesetUri?: string;
};
type Row = {
  path: string;
  size: number;
  type: CustomizationFileType;
  content: string;
};

export const CustomizationFilesetDetailsPanel = ({ filesetUri }: Props) => {
  const toast = useToast();
  const { name, workspace } = parseFilesetUri(filesetUri ?? '');
  const { data, isLoading } = useGetDataset(workspace, name, {
    query: { enabled: Boolean(workspace && name) },
  });
  const [openJsonEditorSidePanel, setOpenJsonEditorSidePanel] = useState(false);
  const [jsonEditorContent, setJsonEditorContent] = useState<Row | undefined>(undefined);
  const dataViewState = DataView.useDataViewState({
    pagination: { pageSize: 10 },
  });
  const {
    rows,
    isPending: isFilesLoading,
    isFetchingRows,
  } = useCustomizationFilesAsRows({
    fileset: filesetUri,
  });

  const loading = isLoading || isFilesLoading;
  const makeColumns: ComponentProps<typeof DataView.Root<Row>>['makeColumns'] = (
    _,
    { rowActionsColumn }
  ) => [
    {
      id: 'type',
      header: 'Type',
      size: 200,
      cell: ({ row }) => CUSTOMIZATION_TYPE_BADGES[row.original?.type],
    },
    {
      id: 'path',
      header: 'Files',
      size: 200,
      cell: ({ row }) => <Text>{row.original?.path}</Text>,
    },
    {
      id: 'size',
      header: 'Size',
      size: 100,
      cell: ({ row }) => <Text>{formatFileSize(row.original?.size)}</Text>,
    },
    rowActionsColumn({
      size: 120,
      maxSize: 120,
      minSize: 120,
      cell: ({ row }) => {
        return (
          <Button
            kind="tertiary"
            disabled={loading || isFetchingRows}
            onClick={() => {
              setOpenJsonEditorSidePanel(true);
              setJsonEditorContent(row.original);
            }}
          >
            <Eye width={16} height={16} />
            View
          </Button>
        );
      },
    }),
  ];
  return (
    <>
      <Panel elevation="high" slotHeading="Dataset" slotIcon={<Database />}>
        <Stack gap="density-lg">
          <Grid cols={{ sm: 2 }} gap="density-lg">
            <KVPair label="Name" value={filesetUri} orientation="vertical" />
            <KVPair
              label="Dataset ID"
              loading={isLoading}
              value={data?.id}
              orientation="vertical"
            />
          </Grid>

          <DataView.Root
            data={rows}
            state={dataViewState}
            makeColumns={makeColumns}
            requestStatus={loading ? 'loading' : undefined}
          >
            <DataView.TableContent
              className="studio-data-view-table"
              renderEmptyState={() => (
                <EntityEmptyState entity="filesetFiles" variant="first-use" />
              )}
            />
            <DataView.Pagination />
          </DataView.Root>
        </Stack>
      </Panel>
      <SidePanel
        className="w-[800px]"
        bordered
        modal
        open={openJsonEditorSidePanel}
        onOpenChange={(open: boolean) => setOpenJsonEditorSidePanel(open)}
        slotHeading={jsonEditorContent?.path}
      >
        {jsonEditorContent?.content && (
          <div className="h-full">
            <CodeSnippet
              value={jsonEditorContent.content}
              language="json"
              kind="block"
              onCopySuccess={() => toast.success('Copied to clipboard')}
            />
          </div>
        )}
      </SidePanel>
    </>
  );
};
