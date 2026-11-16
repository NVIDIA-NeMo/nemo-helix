// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import {
  QuickActionsMenuRoot,
  type QuickActionItem,
} from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useDatasetDirectoryDelete } from '@studio/api/datasets/useDatasetDirectoryDelete';
import type { FileSystemDirectory } from '@studio/components/FilesTable/utils';
import { getDeleteFoldersDescription } from '@studio/constants/filesets';
import { useSelectedDatasetId } from '@studio/hooks/useSelectedDatasetId';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { resolveDatasetFilePath } from '@studio/util/files';
import { type FC, useState } from 'react';

interface Props {
  datasetId?: string;
  directory: FileSystemDirectory;
  /** When set (e.g. file side panel), overrides URL query folder for path resolution */
  currentFolder?: string;
  /** Whether to offer expanding/collapsing every folder beneath this one.
   *  Hosts pass false for leaf folders and for flattened (search) views. */
  showExpandSubtree?: boolean;
  /** Whether this folder and all folders beneath it are already expanded. */
  isSubtreeExpanded?: boolean;
  onToggleExpandSubtree?: () => void;
}

export const DirectoryQuickActions: FC<Props> = ({
  datasetId,
  directory,
  currentFolder,
  showExpandSubtree = false,
  isSubtreeExpanded = false,
  onToggleExpandSubtree,
}) => {
  const [openModal, setOpenModal] = useState<'delete'>();
  const toast = useToast();
  const datasetFullName = useSelectedDatasetId({ datasetId });
  const { workspace, name } = getPartsFromReference(datasetFullName);

  const { mutateAsync: deleteDirectory } = useDatasetDirectoryDelete();
  const { getQueryParam } = useQueryParams();

  const folderFromQuery = getQueryParam(QUERY_PARAMETERS.filesetFolder);
  const path = resolveDatasetFilePath(
    directory.path,
    currentFolder ?? folderFromQuery ?? undefined
  );

  const handleDeleteDirectory = async () => {
    if (!workspace || !name) {
      toast.error('Failed to delete file: invalid dataset name');
      return false;
    }

    const response = await deleteDirectory({ workspace, datasetName: name, path });
    return Boolean(response);
  };

  const actions: QuickActionItem[] = [
    ...(showExpandSubtree && onToggleExpandSubtree
      ? [
          {
            label: isSubtreeExpanded ? 'Collapse all' : 'Expand all',
            onSelect: onToggleExpandSubtree,
          },
        ]
      : []),
  ];

  return (
    <>
      <QuickActionsMenuRoot
        actions={actions}
        deleteAction={{ onSelect: () => setOpenModal('delete') }}
      />
      <DeleteConfirmationModal
        open={openModal === 'delete'}
        entity="folder"
        ids={[path]}
        description={getDeleteFoldersDescription(1)}
        onDelete={handleDeleteDirectory}
        onClose={() => setOpenModal(undefined)}
      />
    </>
  );
};
