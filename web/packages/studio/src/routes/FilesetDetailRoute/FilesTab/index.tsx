// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { isSchemaAssignableFile } from '@nemo/common/src/utils/jsonSchema';
import {
  FilesetPurpose,
  type FilesetFileOutput,
  type FilesetOutput,
} from '@nemo/sdk/generated/platform/schema';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { FilesetFilePreviewContent } from '@studio/components/FilesetFilePreviewPanel/FilesetFilePreviewContent';
import {
  FilesetFileExplorer,
  type ExtraColumn,
} from '@studio/components/filesets/FilesetFileExplorer';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { DatasetSchemaEditor } from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor';
import { getSchemaCellLabel } from '@studio/routes/FilesetDetailRoute/FilesTab/schemaColumn';
import { useCallback, useMemo, type FC } from 'react';

export interface FilesTabProps {
  workspace: string;
  filesetName: string;
  filesetId: string;
  fileset: FilesetOutput;
  files: FilesetFileOutput[] | undefined;
  isFilesError: boolean;
  isFilesFetching: boolean;
}

/**
 * Files tab for a fileset detail page. The left column is the shared
 * `FilesetFileExplorer`. The right column is a purpose-specific panel —
 * today only the `DatasetSchemaEditor` for dataset filesets. For model and
 * generic filesets the right column is omitted so the explorer spans the
 * full width.
 */
export const FilesTab: FC<FilesTabProps> = ({
  workspace,
  filesetName,
  filesetId,
  fileset,
  files,
  isFilesError,
  isFilesFetching,
}) => {
  const { getQueryParam, setQueryParam, setQueryParams } = useQueryParams();
  const currentFolder = getQueryParam(QUERY_PARAMETERS.filesetFolder) ?? undefined;
  const selectedFilePath = getQueryParam(QUERY_PARAMETERS.file) || undefined;

  const handleFileSelect = useCallback(
    (filePath: string) => {
      setQueryParam(QUERY_PARAMETERS.file, filePath);
    },
    [setQueryParam]
  );

  const handleClosePreview = useCallback(() => {
    // Closing the preview by clicking the fileset breadcrumb means "back to the
    // top of the fileset." Clear both the file selection and the folder scope
    // so the explorer renders at root and the URL reflects that.
    setQueryParams({
      [QUERY_PARAMETERS.file]: undefined,
      [QUERY_PARAMETERS.filesetFolder]: undefined,
    });
  }, [setQueryParams]);

  const handleFolderChange = useCallback(
    (folderPath: string) => {
      // Folder breadcrumb click inside the preview: clear the file selection
      // and navigate the explorer to that folder, atomically.
      setQueryParams({
        [QUERY_PARAMETERS.file]: undefined,
        [QUERY_PARAMETERS.filesetFolder]: folderPath || undefined,
      });
    },
    [setQueryParams]
  );

  const handleFolderToggle = useCallback(
    (folderPath: string, isExpanded: boolean) => {
      // When the user collapses the folder that's currently named in the URL
      // (or any ancestor of it), clear `?filesetFolder=` so URL and visual
      // state stay in sync. Expansions never write to the URL — that would
      // churn `?filesetFolder=` on every folder click.
      if (isExpanded || !currentFolder) return;
      const isCurrentOrAncestor =
        currentFolder === folderPath || currentFolder.startsWith(`${folderPath}/`);
      if (isCurrentOrAncestor) {
        setQueryParams({ [QUERY_PARAMETERS.filesetFolder]: undefined });
      }
    },
    [currentFolder, setQueryParams]
  );

  const showSchemaEditor = fileset.purpose === FilesetPurpose.dataset;

  // Schema column for dataset filesets only. Resolves each file's mapping
  // from `metadata.dataset` so the user can see, at a glance, which schema
  // each file uses (or the root schema when it falls back).
  const datasetMetadata = fileset.metadata?.dataset;
  const extraColumns = useMemo<ExtraColumn[] | undefined>(() => {
    if (!showSchemaEditor) return undefined;
    return [
      {
        header: 'Schema',
        width: 140,
        cell: (node) =>
          node.type === 'file' ? getSchemaCellLabel(node.path, datasetMetadata) : null,
      },
    ];
  }, [showSchemaEditor, datasetMetadata]);

  if (isFilesError) {
    return (
      <Flex
        className="w-full min-h-80"
        align="center"
        justify="center"
        data-testid="fileset-files-tab"
      >
        <Text className="text-feedback-danger">Failed to load files.</Text>
      </Flex>
    );
  }

  return (
    <Flex
      direction="row"
      gap="density-md"
      className="w-full h-full min-h-0"
      data-testid="fileset-files-tab"
    >
      <Flex direction="col" className="flex-1 min-w-0 min-h-0 h-full">
        {selectedFilePath ? (
          <div className="w-full h-full min-h-0" data-testid="fileset-files-tab-preview">
            <FilesetFilePreviewContent
              workspace={workspace}
              filesetName={filesetName}
              filePath={selectedFilePath}
              onFilesetClick={handleClosePreview}
              onFolderClick={handleFolderChange}
              onDeleteSuccess={handleClosePreview}
              onRenameSuccess={(newPath) => setQueryParam(QUERY_PARAMETERS.file, newPath)}
            />
          </div>
        ) : (
          <div className="h-full min-h-0">
            <FilesetFileExplorer
              workspace={workspace}
              datasetName={filesetName}
              datasetId={filesetId}
              currentFolder={currentFolder}
              filesList={files}
              isLoading={false}
              isFilesFetching={isFilesFetching}
              onFileSelect={handleFileSelect}
              onFolderToggle={handleFolderToggle}
              extraColumns={extraColumns}
            />
          </div>
        )}
      </Flex>
      {showSchemaEditor && (
        <div
          className="w-[480px] shrink-0 h-full min-h-0 flex flex-col px-density-lg pb-density-lg"
          data-testid="fileset-files-tab-right-panel"
        >
          {!(selectedFilePath && !isSchemaAssignableFile(selectedFilePath)) && (
            <DatasetSchemaEditor
              workspace={workspace}
              datasetName={filesetName}
              fileset={fileset}
              filesList={files}
              selectedFilePath={selectedFilePath}
            />
          )}
        </div>
      )}
    </Flex>
  );
};
