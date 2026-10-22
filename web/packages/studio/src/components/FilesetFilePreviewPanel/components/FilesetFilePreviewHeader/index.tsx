// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@nvidia/foundations-react-core';
import { FolderOpen } from '@nvidia/react-gui-icons/icons';
import { FileActions } from '@studio/components/FilesetFilePreviewPanel/components/FileActions';
import { FileBreadcrumbs } from '@studio/components/FilesetFilePreviewPanel/components/FileBreadcrumbs';
import type { FileSystemFile } from '@studio/components/FilesTable/utils';
import type { FC } from 'react';

export interface FilesetFilePreviewHeaderProps {
  workspace: string;
  filesetName: string;
  filePath: string;
  /** Resolved file used to render delete/rename/split actions. Omit to hide actions. */
  file?: FileSystemFile;
  onFilesetClick?: () => void;
  onFolderClick?: (folderPath: string) => void;
  onDeleteSuccess?: () => void;
  onRenameSuccess?: (newPath: string) => void;
}

export const FilesetFilePreviewHeader: FC<FilesetFilePreviewHeaderProps> = ({
  workspace,
  filesetName,
  filePath,
  file,
  onFilesetClick,
  onFolderClick,
  onDeleteSuccess,
  onRenameSuccess,
}) => (
  <Flex justify="between" align="center" gap="density-sm" className="shrink-0 w-full">
    <Flex gap="density-sm" align="center" className="min-w-0">
      <FolderOpen className="shrink-0" width={16} height={16} />
      <FileBreadcrumbs
        filesetName={filesetName}
        filePath={filePath}
        onFilesetClick={onFilesetClick}
        onFolderClick={onFolderClick}
      />
    </Flex>
    {file && (
      <FileActions
        file={file}
        workspace={workspace}
        filesetName={filesetName}
        onDeleteSuccess={onDeleteSuccess}
        onRenameSuccess={onRenameSuccess}
      />
    )}
  </Flex>
);
