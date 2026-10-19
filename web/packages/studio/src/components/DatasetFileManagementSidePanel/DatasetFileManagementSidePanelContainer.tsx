// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { DatasetFileManagementSidePanel } from '@studio/components/DatasetFileManagementSidePanel';
import { type FC } from 'react';

/**
 * Props for DatasetFileManagementSidePanelContainer
 */
export interface DatasetFileManagementSidePanelContainerProps {
  /** Full dataset identifier in "workspace/name" format */
  datasetId: string;
  /** Whether the sidepanel is open */
  open: boolean;
  /** Current folder path to display (e.g., "training/", "validation/") */
  currentFolder?: string;
  /** Callback when the sidepanel is closed */
  onClose: () => void;
  /** Callback when the folder path changes */
  onFolderChange: (folderPath?: string) => void;
  /** Callback when a file is selected for viewing */
  onFileSelect?: (filePath: string) => void;
}

/**
 * Smart wrapper component for DatasetFileManagementSidePanel
 *
 * This container component handles all data fetching internally, making it
 * easy to use the dataset file management sidepanel without managing the
 * data fetching boilerplate.
 *
 * @example
 * ```tsx
 * const [open, setOpen] = useState(false);
 * const [folder, setFolder] = useState<string | undefined>();
 *
 * <DatasetFileManagementSidePanelContainer
 *   datasetId="default/my-dataset"
 *   open={open}
 *   currentFolder={folder}
 *   onClose={() => setOpen(false)}
 *   onFolderChange={setFolder}
 * />
 * ```
 *
 * @example With state hook
 * ```tsx
 * const sidepanel = useDatasetFileManagementState();
 *
 * <DatasetFileManagementSidePanelContainer
 *   datasetId="default/my-dataset"
 *   open={sidepanel.isOpen}
 *   currentFolder={sidepanel.currentFolder}
 *   onClose={sidepanel.close}
 *   onFolderChange={sidepanel.setFolder}
 * />
 * ```
 */
export const DatasetFileManagementSidePanelContainer: FC<
  DatasetFileManagementSidePanelContainerProps
> = ({ datasetId, open, currentFolder, onClose, onFolderChange, onFileSelect }) => {
  // Parse dataset ID into workspace and name
  const { workspace, name } = getPartsFromReference(datasetId);

  const {
    data: filesResponse,
    isPending: isFilesPending,
    isFetching: isFilesFetching,
  } = useFilesListFilesetFiles(workspace ?? '', name ?? '', undefined, {
    query: { enabled: open && !!workspace && !!name },
  });
  const filesList = filesResponse?.data;

  const isLoading = isFilesPending;

  return (
    <DatasetFileManagementSidePanel
      open={open}
      workspace={workspace}
      datasetName={name}
      datasetId={datasetId}
      currentFolder={currentFolder}
      filesList={filesList}
      isLoading={isLoading}
      isFilesFetching={isFilesFetching}
      onFolderChange={onFolderChange}
      onFileSelect={onFileSelect ?? (() => {})}
      onClose={onClose}
    />
  );
};
