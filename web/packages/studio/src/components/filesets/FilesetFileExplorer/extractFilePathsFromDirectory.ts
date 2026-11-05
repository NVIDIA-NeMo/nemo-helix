// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FileSystemNode } from '@studio/components/FilesTable/utils';

/**
 * Function to recursively extract all file paths from a directory
 */
export const extractFilePathsFromDirectory = (directory: FileSystemNode): string[] => {
  const filePaths: string[] = [];

  if (directory.type === 'file') {
    filePaths.push(directory.path);
  } else if (directory.type === 'directory' && directory.children) {
    for (const child of Object.values(directory.children)) {
      filePaths.push(...extractFilePathsFromDirectory(child));
    }
  }

  return filePaths;
};
