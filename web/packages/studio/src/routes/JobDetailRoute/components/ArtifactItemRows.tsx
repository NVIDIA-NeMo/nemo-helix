// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { Flex, Spinner, Text } from '@nvidia/foundations-react-core';
import type { FileSystemFile } from '@studio/components/FilesTable/utils';
import { ArtifactFileRow } from '@studio/routes/JobDetailRoute/components/ArtifactFileRow';
import type { ArtifactItem } from '@studio/routes/JobDetailRoute/utils';
import { getHumanReadableFileSize } from '@studio/util/files';
import type { FC } from 'react';

export interface ArtifactPreviewState {
  workspace: string;
  fileset: string;
  file: FileSystemFile;
}

export interface ArtifactItemRowsProps {
  item: ArtifactItem;
  onPreview: (state: ArtifactPreviewState) => void;
}

export const ArtifactItemRows: FC<ArtifactItemRowsProps> = ({ item, onPreview }) => {
  const { data, isLoading, error } = useFilesListFilesetFiles(item.workspace, item.fileset, {
    path: item.objectPath,
  });
  if (isLoading) {
    return (
      <Flex justify="center" align="center" className="min-h-[200px] w-full">
        <Spinner size="medium" aria-label={`Loading artifact ${item.resultName}...`} />
      </Flex>
    );
  }
  if (error) {
    return (
      <Text kind="label/regular/sm" className="text-secondary">
        Could not load artifact <code>{item.resultName}</code>:{' '}
        {error instanceof Error ? error.message : 'unknown error'}
      </Text>
    );
  }

  const descendantPrefix = `${item.objectPath}/`;
  const files = (data?.data ?? []).filter(
    (f) => f.path === item.objectPath || f.path.startsWith(descendantPrefix)
  );
  if (files.length === 0) {
    return (
      <Text kind="label/regular/sm" className="text-secondary">
        No files found for artifact <code>{item.resultName}</code>.
      </Text>
    );
  }

  return (
    <>
      {files.map((file) => (
        <ArtifactFileRow
          key={file.file_ref}
          onClick={() =>
            onPreview({
              workspace: item.workspace,
              fileset: item.fileset,
              file: { type: 'file', path: file.path, size: file.size, oid: file.file_ref },
            })
          }
        >
          <Text kind="body/semibold/md">{file.path}</Text>
          <Text kind="label/regular/sm" className="text-secondary">
            {getHumanReadableFileSize(file.size)}
          </Text>
        </ArtifactFileRow>
      ))}
    </>
  );
};
