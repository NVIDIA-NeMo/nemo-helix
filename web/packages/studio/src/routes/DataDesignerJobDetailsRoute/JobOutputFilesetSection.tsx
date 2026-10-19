// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import {
  getFilesListFilesetFilesQueryKey,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { Anchor, Banner, Card, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { Empty } from '@studio/components/Empty';
import { FilesetFilePreviewPanel } from '@studio/components/FilesetFilePreviewPanel';
import type { FileSystemFile } from '@studio/components/FilesTable/utils';
import { useDataDesignerArtifactsFileset } from '@studio/routes/DataDesignerJobDetailsRoute/useDataDesignerArtifactsFileset';
import { getFilesetDetailsRoute } from '@studio/routes/utils';
import { getHumanReadableFileSize } from '@studio/util/files';
import { useQueryClient } from '@tanstack/react-query';
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ComponentProps,
  type FC,
  type ReactNode,
} from 'react';
import { Link, useNavigate } from 'react-router';

type FileRow = FilesetFileOutput & { id: string };

function fileRowToSystemFile(row: FileRow): FileSystemFile {
  return {
    type: 'file',
    path: row.path,
    size: row.size,
    oid: row.file_ref,
  };
}

const centeredCard = (children: ReactNode) => (
  <Card
    className="min-w-0 w-full"
    attributes={{ CardContent: { className: 'flex justify-center items-center' } }}
  >
    {children}
  </Card>
);

export const JobOutputFilesetSection: FC = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [previewFile, setPreviewFile] = useState<FileSystemFile | null>(null);

  const {
    isTerminal,
    artifactsResult,
    filesetLoc,
    filesetWorkspace,
    filesetName,
    listFilesParams,
    files,
    isResultsLoading,
    isResultsError,
    resultsError,
    isFilesLoading,
    isFilesError: isListFilesError,
    filesError: listFilesError,
  } = useDataDesignerArtifactsFileset();

  const {
    data: filesetMeta,
    isLoading: isFilesetLoading,
    isError: isFilesetError,
    error: filesetError,
  } = useFilesRetrieveFileset(filesetWorkspace, filesetName, {
    query: {
      enabled: Boolean(filesetWorkspace && filesetName),
    },
  });

  const dataViewState = useStudioDataViewState({
    defaultPageSize: 10,
  });

  const rows: FileRow[] = useMemo(() => files.map((f) => ({ ...f, id: f.file_ref })), [files]);

  const makeColumns: ComponentProps<typeof StudioDataView<FileRow>>['makeColumns'] = useMemo(
    () => (helpers) => [
      helpers.accessor('path', {
        header: 'Path',
        cell: ({ row }) => (
          <Text kind="body/regular/md" className="font-mono text-sm">
            {row.original.path}
          </Text>
        ),
      }),
      helpers.accessor('size', {
        header: 'Size',
        size: 120,
        cell: ({ row }) => (
          <Text kind="body/regular/md">{getHumanReadableFileSize(row.original.size)}</Text>
        ),
      }),
    ],
    []
  );

  const invalidateFilesetFileQueries = useCallback(() => {
    void queryClient.invalidateQueries({
      queryKey: getFilesListFilesetFilesQueryKey(filesetWorkspace, filesetName, listFilesParams),
    });
  }, [queryClient, filesetWorkspace, filesetName, listFilesParams]);

  const handlePreviewClose = useCallback(() => {
    setPreviewFile(null);
  }, []);

  const handleRowClick = useCallback((row: FileRow) => {
    setPreviewFile(fileRowToSystemFile(row));
  }, []);

  const filesetId = `${filesetWorkspace}/${filesetName}`;

  const handleDatasetClickFromPreview = useCallback(() => {
    navigate(getFilesetDetailsRoute(filesetWorkspace, filesetId));
  }, [navigate, filesetWorkspace, filesetId]);

  const handleFolderClickFromPreview = useCallback(
    (folderPath: string) => {
      navigate(getFilesetDetailsRoute(filesetWorkspace, filesetId, folderPath));
    },
    [navigate, filesetWorkspace, filesetId]
  );

  const handleFileDeleteSuccess = useCallback(() => {
    setPreviewFile(null);
    invalidateFilesetFileQueries();
  }, [invalidateFilesetFileQueries]);

  const handleFileRenameSuccess = useCallback(
    (newPath: string) => {
      setPreviewFile((prev) => (prev ? { ...prev, path: newPath } : null));
      invalidateFilesetFileQueries();
    },
    [invalidateFilesetFileQueries]
  );

  useEffect(() => {
    if (!filesetWorkspace || !filesetName) {
      setPreviewFile(null);
    }
  }, [filesetWorkspace, filesetName]);

  if (isResultsLoading && !artifactsResult) {
    return centeredCard(<Spinner description="Loading job results..." />);
  }

  if (isResultsError) {
    const errorTitle =
      resultsError instanceof Error ? 'Error loading job results' : 'Could not load job results';
    const errorMessage =
      resultsError instanceof Error
        ? resultsError.message
        : 'The job results list could not be loaded.';
    return centeredCard(<Empty title={errorTitle} description={errorMessage} />);
  }

  if (!artifactsResult) {
    const emptyTitle = isTerminal
      ? 'No artifacts result was returned for this job.'
      : 'Output files will appear here once the job registers its artifacts result.';
    const emptyDescription = isTerminal
      ? 'This job completed but did not produce an artifacts result.'
      : 'Check back after the job registers its artifacts result.';
    return centeredCard(<Empty title={emptyTitle} description={emptyDescription} />);
  }

  if (!filesetLoc) {
    return (
      <Card>
        <Stack gap="4">
          <KVPair
            label="Artifact URL"
            value={artifactsResult.artifact_url}
            orientation="vertical"
          />
        </Stack>
      </Card>
    );
  }

  const filesErr = filesetError ?? listFilesError;
  const filesServiceError =
    isFilesetError || isListFilesError
      ? filesErr instanceof Error
        ? filesErr.message
        : 'Could not load fileset or file list from the Files service.'
      : null;

  return (
    <>
      <Card>
        <Stack gap="2">
          <Stack gap="2">
            <Text kind="body/bold/lg">Output fileset</Text>
            <Text kind="body/regular/sm" className="text-muted">
              Generated data for this job is stored in the following workspace fileset.
            </Text>
          </Stack>

          {filesServiceError != null && (
            <Banner kind="inline" status="error" title="Files service error">
              {filesServiceError}
            </Banner>
          )}

          <Stack gap="4">
            <KVPair
              label="Fileset"
              value={
                <Anchor>
                  <Link to={getFilesetDetailsRoute(filesetWorkspace, filesetId)}>
                    {filesetName}
                  </Link>
                </Anchor>
              }
            />
            {filesetMeta?.description?.trim() ? (
              <KVPair label="Description" value={filesetMeta.description} orientation="vertical" />
            ) : null}
          </Stack>

          <Stack gap="4">
            <StudioDataView<FileRow>
              dataViewState={dataViewState}
              makeColumns={makeColumns}
              onRowClick={handleRowClick}
              attributes={{
                DataViewRoot: {
                  data: rows,
                  totalCount: rows.length,
                  requestStatus: isFilesetLoading || isFilesLoading ? 'loading' : undefined,
                },
                DataViewTableContent: {
                  className: 'studio-data-view-table',
                  renderEmptyState: () => (
                    <TableEmptyState
                      className="py-4"
                      header="No files yet"
                      emptyMessage="This fileset has no files, or they are not visible yet."
                    />
                  ),
                },
              }}
            />
          </Stack>
        </Stack>
      </Card>

      <FilesetFilePreviewPanel
        open={previewFile != null}
        onCloseClick={handlePreviewClose}
        onOutsideClick={handlePreviewClose}
        workspace={filesetWorkspace}
        filesetName={filesetName}
        filePath={previewFile?.path ?? ''}
        file={previewFile ?? undefined}
        onFilesetClick={handleDatasetClickFromPreview}
        onFolderClick={handleFolderClickFromPreview}
        onDeleteSuccess={handleFileDeleteSuccess}
        onRenameSuccess={handleFileRenameSuccess}
      />
    </>
  );
};
