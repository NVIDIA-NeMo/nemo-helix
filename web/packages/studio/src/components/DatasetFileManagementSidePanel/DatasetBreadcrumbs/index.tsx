// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Breadcrumbs } from '@nvidia/foundations-react-core';
import { type FC, useMemo } from 'react';

interface Props {
  datasetName: string;
  currentFolder?: string;
  onFolderChange: (folderPath?: string) => void;
}

/**
 * This component renders breadcrumbs that represent folders in the given dataset.
 */
export const DatasetBreadcrumbs: FC<Props> = ({ datasetName, currentFolder, onFolderChange }) => {
  const folders = useMemo(() => {
    const rootBreadcrumb = { label: datasetName, onClick: () => onFolderChange() };
    // if no filesetFolder query param, return the root breadcrumb, which is the dataset name
    if (!currentFolder) {
      return [rootBreadcrumb];
    }

    // Keeps track of the current path as we traverse each folder
    let currentPath = '';
    const folderPaths = currentFolder
      .split('/')
      .filter(Boolean)
      .map((folderPath) => {
        // Update the current path to include this folder
        currentPath = currentPath ? `${currentPath}/${folderPath}` : folderPath;
        const localCurrentPath = currentPath;
        return {
          label: folderPath,
          onClick: () => {
            onFolderChange(localCurrentPath);
          },
        };
      });

    return [rootBreadcrumb, ...folderPaths];
  }, [datasetName, currentFolder, onFolderChange]);

  return (
    <Breadcrumbs
      data-testid="dataset-breadcrumbs"
      items={folders.map(({ onClick, label }) => ({
        children: (
          <button
            type="button"
            onClick={onClick}
            className="text-inherit hover:underline cursor-pointer"
          >
            {label}
          </button>
        ),
      }))}
    />
  );
};
