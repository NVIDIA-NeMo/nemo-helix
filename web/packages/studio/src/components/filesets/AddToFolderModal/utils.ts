// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FileSystemNode } from '@studio/components/FilesTable/utils';

/**
 * Gets sibling folder names at the current folder level.
 * Only returns folder names (not full paths) that are direct children of the current folder.
 * @param filesList - Array of file system nodes (the current folder's contents)
 * @returns Sorted array of sibling folder names
 */
export const getSiblingFolders = (filesList: FileSystemNode[] | undefined): string[] => {
  if (!filesList || filesList.length === 0) {
    return [];
  }

  const siblingFolders: string[] = [];

  for (const node of filesList) {
    if (node.type === 'directory' && node.path) {
      // Get just the folder name (last segment of the path)
      const folderName = node.path.split('/').pop();
      if (folderName) {
        siblingFolders.push(folderName);
      }
    }
  }

  return siblingFolders.sort();
};

/**
 * Calculates the parent folder path from a given folder path.
 * @param currentFolder - The current folder path
 * @returns The parent folder path, or empty string if at root
 */
export const getParentFolder = (currentFolder: string | undefined): string => {
  if (!currentFolder) {
    return '';
  }
  const lastSlashIndex = currentFolder.lastIndexOf('/');
  return lastSlashIndex > 0 ? currentFolder.substring(0, lastSlashIndex) : '';
};
