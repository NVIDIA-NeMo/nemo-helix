// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { DatasetFileManagementSidePanel } from '@studio/components/DatasetFileManagementSidePanel';
import { FilesetFilePreviewPanel } from '@studio/components/FilesetFilePreviewPanel';
import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { getFilesetDetailsRoute, getFilesetFileRoute } from '@studio/routes/utils';
import { FC, useEffect, useState } from 'react';
import { generatePath, useNavigate, useParams } from 'react-router';

interface PanelManagementProps {
  workspace: string;
}

/**
 * Dataset and file preview panels.
 *
 * **Mount** follows the URL; close navigation is deferred until the SidePanel exit
 * animation finishes, so panel data stays mounted for the whole close.
 *
 * **Open** is separate. SidePanel skips enter animation when mounting already open, so
 * we mount closed and flip open one commit later (effect). Close must flip open false
 * synchronously in the handler — an effect-driven close races SidePanel's isClosing
 * reset and causes a visible flash.
 */
export const PanelManagement: FC<PanelManagementProps> = ({ workspace }) => {
  const navigate = useNavigate();
  const { getQueryParam } = useQueryParams();

  // ==========================================
  // URL PARAMS & OPEN STATE
  // ==========================================

  // `useParams()` already returns decoded values, and `getFilesetFileRoute()`
  // encodes them when building the URL — decoding again here is redundant and
  // throws on valid names containing a literal `%`.
  const {
    [ROUTE_PARAMS.filesetId]: datasetIdFromUrl,
    [ROUTE_PARAMS.filePathEncoded]: filePathFromUrl,
  } = useParams();
  const currentFolder = getQueryParam(QUERY_PARAMETERS.filesetFolder);

  // Mount as soon as the URL references the panel (see component doc).
  const showDatasetPanel = !!datasetIdFromUrl;
  const showFilePanel = !!filePathFromUrl;

  // Open flag — lags the URL on open (drives the slide-in), flips synchronously
  // on close (see close handlers below).
  const [isDatasetPanelOpen, setIsDatasetPanelOpen] = useState(false);
  const [isFilePanelOpen, setIsFilePanelOpen] = useState(false);

  useEffect(() => {
    setIsDatasetPanelOpen(!!datasetIdFromUrl);
  }, [datasetIdFromUrl]);

  useEffect(() => {
    setIsFilePanelOpen(!!filePathFromUrl);
  }, [filePathFromUrl]);

  // ==========================================
  // DATASET PANEL LOGIC
  // ==========================================

  // Parse dataset info from URL
  const datasetFullName = datasetIdFromUrl ?? '';
  const { workspace: datasetworkspace, name: datasetName } = getPartsFromReference(datasetFullName);

  // Fetch dataset files
  const {
    data: filesResponse,
    isPending: isFilesPending,
    isFetching: isFilesFetching,
  } = useFilesListFilesetFiles(datasetworkspace || '', datasetName || '', undefined, {
    query: { enabled: showDatasetPanel },
  });
  const filesList = filesResponse?.data;

  // Dataset panel handlers
  const handleDatasetPanelClose = () => {
    setIsDatasetPanelOpen(false);
    navigate(generatePath(ROUTES.workspace.filesets, { workspace }), { flushSync: true });
  };

  const handleFileSelect = (filePath: string) => {
    const to = getFilesetFileRoute(workspace, datasetFullName, filePath);
    navigate(to);
  };

  // ==========================================
  // FILE PANEL LOGIC
  // ==========================================

  const decodedFilePath = filePathFromUrl ?? '';

  const handleFilePanelClosing = () => {
    setIsFilePanelOpen(false);
  };

  const handleFilePanelClose = () => {
    const folderPathFromFile = decodedFilePath.split('/').slice(0, -1).join('/');
    navigate(getFilesetDetailsRoute(workspace, datasetFullName, folderPathFromFile), {
      flushSync: true,
    });
  };

  const handleFilePanelOutsideClick = () => {
    navigate(generatePath(ROUTES.workspace.filesets, { workspace }), { flushSync: true });
  };

  const handleDatasetClick = () => {
    navigate(getFilesetDetailsRoute(workspace, datasetFullName));
  };

  const handleFolderClick = (folderPath?: string) => {
    navigate(getFilesetDetailsRoute(workspace, datasetFullName, folderPath || undefined));
  };

  const handleFileDeleteSuccess = () => {
    // Navigate back to the folder or fileset root after deletion
    const folderPathFromFile = decodedFilePath.split('/').slice(0, -1).join('/');
    navigate(getFilesetDetailsRoute(workspace, datasetFullName, folderPathFromFile));
  };

  const handleFileRenameSuccess = (newPath: string) => {
    // Navigate to the renamed file
    navigate(getFilesetFileRoute(workspace, datasetFullName, newPath));
  };

  // ==========================================
  // RENDER
  // ==========================================

  return (
    <>
      {/* Dataset Panel - renders first (bottom layer) */}
      {showDatasetPanel && (
        <DatasetFileManagementSidePanel
          open={isDatasetPanelOpen}
          workspace={datasetworkspace || ''}
          datasetName={datasetName || ''}
          datasetId={datasetFullName}
          currentFolder={currentFolder}
          filesList={filesList}
          isLoading={isFilesPending}
          isFilesFetching={isFilesFetching}
          onFolderChange={handleFolderClick}
          onFileSelect={handleFileSelect}
          onClose={handleDatasetPanelClose}
        />
      )}

      {/* File Panel - renders last (top layer) */}
      {showFilePanel && (
        <FilesetFilePreviewPanel
          open={isFilePanelOpen}
          onCloseClick={handleFilePanelClose}
          onOutsideClick={handleFilePanelOutsideClick}
          onClosing={handleFilePanelClosing}
          workspace={datasetworkspace || ''}
          filesetName={datasetName || ''}
          filePath={decodedFilePath}
          onFilesetClick={handleDatasetClick}
          onFolderClick={handleFolderClick}
          onDeleteSuccess={handleFileDeleteSuccess}
          onRenameSuccess={handleFileRenameSuccess}
        />
      )}
    </>
  );
};
