// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTE_PARAMS } from '@studio/constants/routes';
import { FilesetDetailTab } from '@studio/routes/FilesetDetailRoute/constants';
import { FilesetCard } from '@studio/routes/FilesetDetailRoute/FilesetCard';
import { FilesTab, type FilesTabProps } from '@studio/routes/FilesetDetailRoute/FilesTab';
import type { FC } from 'react';
import { useOutletContext, useParams } from 'react-router';

/** What `FilesetDetailRoute` hands its tab through the outlet. */
export type FilesetDetailOutletContext = Omit<FilesTabProps, 'hideFileActions'>;

/** Renders the tab named in the path. The parent route redirects unknown tabs. */
export const FilesetDetailTabContent: FC = () => {
  const { [ROUTE_PARAMS.filesetTab]: tab } = useParams();
  const context = useOutletContext<FilesetDetailOutletContext>();

  return tab === FilesetDetailTab.Files ? (
    <FilesTab {...context} />
  ) : (
    <FilesetCard
      workspace={context.workspace}
      filesetName={context.filesetName}
      fileset={context.fileset}
      files={context.files}
      isFilesError={context.isFilesError}
    />
  );
};
