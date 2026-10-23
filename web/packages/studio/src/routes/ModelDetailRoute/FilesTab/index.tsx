// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import {
  useFilesListFilesetFiles,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import { Flex } from '@nvidia/foundations-react-core';
import { Empty } from '@studio/components/Empty';
import { Loading } from '@studio/components/Layouts/Loading';
import type { FilesetRef } from '@studio/hooks/useModelLineage';
import { FilesTab as FilesetFilesTab } from '@studio/routes/FilesetDetailRoute/FilesTab';
import { FolderOpen } from '@nvidia/react-gui-icons/icons';
import type { FC, ReactNode } from 'react';

export interface FilesTabProps {
  fileset: FilesetRef | null;
}

export const FilesTab: FC<FilesTabProps> = ({ fileset: filesetRef }) => {
  const workspace = filesetRef?.workspace ?? '';
  const name = filesetRef?.name ?? '';
  const enabled = Boolean(filesetRef);

  const {
    data: files,
    isPending: isFilesPending,
    isFetching: isFilesFetching,
    isError: isFilesError,
  } = useFilesListFilesetFiles(workspace, name, undefined, { query: { enabled } });
  const {
    data: fileset,
    isPending: isFilesetPending,
    isError: isFilesetError,
  } = useFilesRetrieveFileset(workspace, name, { query: { enabled, retry: false } });

  let state: ReactNode;
  if (!filesetRef) {
    state = (
      <Empty
        icon={<FolderOpen className="size-12" />}
        title="No files"
        description="No fileset is linked to this model, so there are no files to browse."
      />
    );
  } else if (isFilesetPending || isFilesPending) {
    state = <Loading description="Loading files..." />;
  } else if (isFilesetError || !fileset) {
    state = (
      <ErrorMessage
        header="Fileset not found"
        message={`Couldn't load the fileset "${workspace}/${name}".`}
      />
    );
  } else {
    return (
      <FilesetFilesTab
        workspace={workspace}
        filesetName={name}
        filesetId={getEntityReference({ namespace: workspace, name })}
        fileset={fileset}
        files={files?.data}
        isFilesError={isFilesError}
        isFilesFetching={isFilesFetching}
        hideFileActions
      />
    );
  }

  return (
    <Flex direction="col" justify="center" className="flex-1">
      {state}
    </Flex>
  );
};
