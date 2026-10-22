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
import { FolderOpen } from 'lucide-react';
import type { FC, ReactNode } from 'react';

export interface FilesTabProps {
  /** Null when the model or adapter records no fileset, e.g. a base model served elsewhere. */
  fileset: FilesetRef | null;
}

/** Fills the scrolling tab panel so a state message sits in its middle. */
const Centered: FC<{ children: ReactNode }> = ({ children }) => (
  <Flex direction="col" justify="center" className="flex-1">
    {children}
  </Flex>
);

/** The model's or adapter's fileset, browsable in place with the fileset page's explorer. */
export const FilesTab: FC<FilesTabProps> = ({ fileset }) =>
  fileset ? (
    <FilesetFiles fileset={fileset} />
  ) : (
    <Centered>
      <Empty
        icon={<FolderOpen className="size-12" />}
        title="No files"
        description="No fileset is linked to this model, so there are no files to browse."
      />
    </Centered>
  );

const FilesetFiles: FC<{ fileset: FilesetRef }> = ({ fileset: { workspace, name } }) => {
  const {
    data: files,
    isPending: isFilesPending,
    isFetching: isFilesFetching,
    isError: isFilesError,
  } = useFilesListFilesetFiles(workspace, name);
  const {
    data: fileset,
    isPending: isFilesetPending,
    isError: isFilesetError,
  } = useFilesRetrieveFileset(workspace, name, { query: { retry: false } });

  if (isFilesetPending || isFilesPending) {
    return (
      <Centered>
        <Loading description="Loading files..." />
      </Centered>
    );
  }

  if (isFilesetError || !fileset) {
    return (
      <Centered>
        <ErrorMessage
          header="Fileset not found"
          message={`Couldn't load the fileset "${workspace}/${name}".`}
        />
      </Centered>
    );
  }

  return (
    <FilesetFilesTab
      workspace={workspace}
      filesetName={name}
      filesetId={getEntityReference({ namespace: workspace, name })}
      fileset={fileset}
      files={files?.data}
      isFilesError={isFilesError}
      isFilesFetching={isFilesFetching}
      // The preview's menu is absolutely positioned and would land on the page header's actions.
      hideFileActions
    />
  );
};
