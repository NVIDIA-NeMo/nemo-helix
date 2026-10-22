// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GridColDef } from '@mui/x-data-grid';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { Flex } from '@nvidia/foundations-react-core';
import { FolderClosed, IconDocument } from '@nvidia/react-gui-icons/icons';
import { DirectoryQuickActions } from '@studio/components/FilesTable/DirectoryQuickActions';
import { FileQuickActions } from '@studio/components/FilesTable/FileQuickActions';
import { getHumanReadableFileSize } from '@studio/util/files';

/**
 * Base file entry interface that works with both HuggingFace ListFileEntry and v2 FilesetFileOutput.
 * Uses `oid` for backward compatibility with existing code that references it.
 */
export interface FileEntry {
  path: string;
  size: number;
  /** IconDocument identifier - maps to `oid` from HF or `file_ref` from v2 API */
  oid: string;
}

export interface FileSystemFile extends FileEntry {
  type: 'file';
}

export interface FileSystemDirectory extends FileEntry {
  type: 'directory';
  children: { [key: string]: FileSystemNode };
}

export type FileSystemNode = FileSystemFile | FileSystemDirectory;

/**
 * Placeholder filename used to persist otherwise-empty directories on the
 * backing storage. These markers are implementation details and should never
 * be surfaced in the UI.
 */
export const GITKEEP_FILENAME = '.gitkeep';

export const isGitkeepPath = (path: string): boolean => {
  const basename = path.split('/').pop() ?? '';
  return basename === GITKEEP_FILENAME;
};

/**
 * Converts FilesetFileOutput to FileEntry for compatibility with file tree utilities.
 */
export const toFileEntry = (file: FilesetFileOutput): FileEntry => ({
  path: file.path,
  size: file.size,
  oid: file.file_ref,
});

/**
 * This method takes a list of files in a dataset and builds a corresponding file tree.
 *
 * Datasets are currently returning files as list of file-like objects that contain path, sha, and size.
 * This can roughly create a filesystem by utilizing the forward slash "/" delimitter as a folder path.
 *
 * `.gitkeep` placeholder files are never added as leaves — their parent directory
 * chain is still created so the (otherwise empty) folder remains visible.
 */
export const mapFileListToFileTree = (files: FileEntry[]): FileSystemNode => {
  const root: FileSystemNode = {
    type: 'directory',
    size: 0,
    path: '',
    oid: '',
    children: {},
  };

  for (const file of files) {
    const pathParts = file.path.split('/');
    const isGitkeep = isGitkeepPath(file.path);
    let currentDir = root;

    for (let i = 0; i < pathParts.length; i++) {
      const part = pathParts[i];

      if (i === pathParts.length - 1) {
        if (!isGitkeep) {
          currentDir.children![part] = {
            ...file,
            type: 'file',
          };
        }
        break;
      }

      // If we get here, we're processing a directory
      if (!currentDir.children![part]) {
        currentDir.children![part] = {
          type: 'directory',
          size: 0,
          path: pathParts.slice(0, i + 1).join('/'),
          oid: '',
          children: {},
        };
      }
      currentDir = currentDir.children![part] as FileSystemDirectory;
    }
  }

  return root;
};

export const getFoldersFilesAtPath = (fileTree: FileSystemNode, path: string) => {
  const emptyEntries = { entries: [] };
  const pathParts = path.split('/').filter(Boolean);
  let currentDirectory: FileSystemNode = fileTree;

  // Navigate to the requested path
  for (const part of pathParts) {
    // Return empty if:
    if (
      // - We landed on a file
      currentDirectory.type === 'file' ||
      // - The directory we're looking for doesn't exist
      !(currentDirectory as FileSystemDirectory).children?.[part]
    ) {
      return emptyEntries;
    }
    currentDirectory = (currentDirectory as FileSystemDirectory).children[part];
  }

  // If we ended up on a file, return empty
  if (currentDirectory.type === 'file') {
    return emptyEntries;
  }

  const directory = currentDirectory as FileSystemDirectory;
  return {
    entries: Object.values(directory.children),
  };
};

export interface TreeRow {
  node: FileSystemNode;
  depth: number;
}

export interface FlattenTreeSortOrder {
  sortBy: 'name' | 'size';
  order: 'asc' | 'desc';
}

/**
 * Flattens a file tree into a list of rows with depth for indentation.
 * Only includes children of expanded directories.
 */
export function flattenFileTree(
  root: FileSystemNode,
  expandedPaths: Set<string>,
  sortOrder: FlattenTreeSortOrder
): TreeRow[] {
  if (root.type === 'file') return [];
  const dir = root as FileSystemDirectory;
  const entries = Object.values(dir.children);

  const compareFn = (a: FileSystemNode, b: FileSystemNode) => {
    const aName = a.path.split('/').pop() ?? '';
    const bName = b.path.split('/').pop() ?? '';
    if (sortOrder.sortBy === 'name') {
      const cmp = aName.localeCompare(bName);
      return sortOrder.order === 'asc' ? cmp : -cmp;
    }
    const sizeA = a.type === 'file' ? a.size : 0;
    const sizeB = b.type === 'file' ? b.size : 0;
    return sortOrder.order === 'asc' ? sizeA - sizeB : sizeB - sizeA;
  };

  entries.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
    return compareFn(a, b);
  });

  const result: TreeRow[] = [];
  function collect(node: FileSystemNode, depth: number): void {
    result.push({ node, depth });
    if (node.type === 'directory' && expandedPaths.has(node.path)) {
      const childEntries = Object.values((node as FileSystemDirectory).children);
      childEntries.sort((a, b) => {
        if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
        return compareFn(a, b);
      });
      for (const child of childEntries) {
        collect(child, depth + 1);
      }
    }
  }
  for (const entry of entries) {
    collect(entry, 0);
  }
  return result;
}

interface GetFilesTablesColumnsProps {
  datasetId?: string;
}
export const getFilesTablesColumns = ({
  datasetId,
}: GetFilesTablesColumnsProps): GridColDef<FileSystemNode>[] => {
  return [
    {
      field: 'path',
      headerName: 'Name',
      flex: 3,
      renderCell: ({ row: file }) => {
        const icon = file.type === 'directory' ? <FolderClosed /> : <IconDocument />;
        const filename = file.path.split('/').pop();
        return (
          <Flex gap="density-sm" align="center">
            {icon}
            <div>{filename}</div>
          </Flex>
        );
      },
    },
    {
      field: 'size',
      headerName: 'Size',
      flex: 1,
      align: 'right',
      headerAlign: 'right',
      renderCell: ({ row: file }) =>
        file.type === 'file' ? getHumanReadableFileSize(file.size) : null,
    },
    {
      field: 'actions',
      headerName: '',
      align: 'center',
      renderCell: ({ row: file }) => {
        if (file.type === 'file') {
          return <FileQuickActions file={file} datasetId={datasetId} />;
        } else if (file.type === 'directory') {
          return <DirectoryQuickActions directory={file} datasetId={datasetId} />;
        }
      },
      width: 52,
    },
  ];
};
