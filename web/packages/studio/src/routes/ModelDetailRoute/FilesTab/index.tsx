// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import {
  useFilesListFilesetFiles,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import { Empty } from '@studio/components/Empty';
import { Loading } from '@studio/components/Layouts/Loading';
import type { FilesetRef } from '@studio/hooks/useModelLineage';
import { FilesTab as FilesetFilesTab } from '@studio/routes/FilesetDetailRoute/FilesTab';
import { FolderOpen } from 'lucide-react';
import type { FC } from 'react';

export interface FilesTabProps {
  /** Null when the model or adapter records no fileset, e.g. a base model served elsewhere. */
  fileset: FilesetRef | null;
}

/** The model's or adapter's fileset, browsable in place with the fileset page's explorer. */
export const FilesTab: FC<FilesTabProps> = ({ fileset }) =>
  fileset ? (
    <FilesetFiles fileset={fileset} />
  ) : (
    <Empty
      icon={<FolderOpen className="size-12" />}
      title="No files"
      description="No fileset is linked to this model, so there are no files to browse."
    />
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
    return <Loading description="Loading files..." />;
  }

  if (isFilesetError || !fileset) {
    return (
      <ErrorMessage
        header="Fileset not found"
        message={`Couldn't load the fileset "${workspace}/${name}".`}
      />
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
    />
  );
};
