// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileContentPreview } from '@nemo/common/src/components/FileContentPreview';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { FilesetFilePreviewHeader } from '@studio/components/FilesetFilePreviewPanel/components/FilesetFilePreviewHeader';
import { FilesetImagePreview } from '@studio/components/FilesetFilePreviewPanel/FilesetImagePreview';
import { useIsBinaryFile } from '@studio/components/filesets/hooks/useIsBinaryFile';
import type { FileSystemFile } from '@studio/components/FilesTable/utils';
import { isImageExtension } from '@studio/util/binaryFile';
import { useMemo, type FC } from 'react';

export interface FilesetFilePreviewContentProps {
  // Fileset context
  workspace: string;
  filesetName: string;
  filePath: string;

  // Navigation callbacks
  onFilesetClick?: () => void;
  onFolderClick?: (folderPath: string) => void;

  // File actions
  onDeleteSuccess?: () => void;
  onRenameSuccess?: (newPath: string) => void;

  // Optional: pre-fetched data (parent already has the file + content)
  file?: FileSystemFile;
  fileContent?: string;
  isLoading?: boolean;
  error?: Error;

  /**
   * When true, the inline header (breadcrumbs + file actions) is suppressed.
   * The host is responsible for rendering equivalent affordances elsewhere
   * (e.g. a SidePanel slotHeading). Defaults to false.
   */
  hideHeader?: boolean;

  /** Whether the host considers this content active for fetching purposes. Defaults to true. */
  enabled?: boolean;
}

/**
 * Content-only variant of the fileset file preview:
 * breadcrumbs + file actions header + read-only file viewer.
 *
 * No panel chrome. Used inline from the dataset / model detail Files tabs,
 * and composed by `FilesetFilePreviewPanel` (the side-panel wrapper).
 */
export const FilesetFilePreviewContent: FC<FilesetFilePreviewContentProps> = ({
  workspace,
  filesetName,
  filePath,
  onFilesetClick,
  onFolderClick,
  onDeleteSuccess,
  onRenameSuccess,
  file: externalFile,
  fileContent: externalContent,
  isLoading: externalLoading,
  error: externalError,
  hideHeader = false,
  enabled = true,
}) => {
  const { isBinary: binary, isLoading: isBinaryLoading } = useIsBinaryFile(filePath);
  const isImage = isImageExtension(filePath);

  const {
    data: internalContent,
    isLoading: internalLoading,
    error: internalError,
  } = useDatasetFileContent({
    workspace,
    name: filesetName,
    path: filePath,
    enabled: externalContent === undefined && enabled && !binary && !isImage && !isBinaryLoading,
  });

  const { data: allFilesResponse } = useFilesListFilesetFiles(workspace, filesetName, undefined, {
    query: { enabled: !externalFile && enabled },
  });
  const allFiles = allFilesResponse?.data;

  const fileContent = externalContent ?? internalContent;
  const isLoading = externalLoading ?? internalLoading;
  const error = externalError ?? internalError;
  const file =
    externalFile ?? (allFiles?.find((f) => f.path === filePath) as FileSystemFile | undefined);

  const body = useMemo(
    () =>
      isImage ? (
        <FilesetImagePreview
          workspace={workspace}
          filesetName={filesetName}
          filePath={filePath}
          enabled={enabled}
        />
      ) : binary ? (
        <Flex align="center" justify="center" className="h-full">
          <Text kind="body/regular/md" className="text-fg-subdued">
            Text preview not available for binary files.
          </Text>
        </Flex>
      ) : (
        <FileContentPreview
          file={{ path: filePath }}
          content={fileContent}
          isLoading={isBinaryLoading || isLoading}
          error={error ?? null}
        />
      ),
    [
      binary,
      enabled,
      error,
      fileContent,
      filePath,
      filesetName,
      isBinaryLoading,
      isImage,
      isLoading,
      workspace,
    ]
  );

  if (hideHeader) {
    return body;
  }

  return (
    <Stack gap="density-sm" className="h-full min-h-0">
      <FilesetFilePreviewHeader
        workspace={workspace}
        filesetName={filesetName}
        filePath={filePath}
        file={file}
        onFilesetClick={onFilesetClick}
        onFolderClick={onFolderClick}
        onDeleteSuccess={onDeleteSuccess}
        onRenameSuccess={onRenameSuccess}
      />
      <div className="flex-1 min-h-0">{body}</div>
    </Stack>
  );
};
