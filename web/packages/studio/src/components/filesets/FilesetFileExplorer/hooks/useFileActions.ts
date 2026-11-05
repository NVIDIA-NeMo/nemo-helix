// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FileContentsWithPath } from '@nemo/common/src/data-store/types';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import {
  type FileSystemNode,
  type FileSystemFile,
  flattenFileTree,
  isGitkeepPath,
  mapFileListToFileTree,
  toFileEntry,
  type TreeRow,
} from '@studio/components/FilesTable/utils';
import { useCallback, useMemo, useState } from 'react';

export interface SortOrder {
  sortBy: 'name' | 'size';
  order: 'asc' | 'desc';
}

export interface UseFileActionsOptions {
  /** Raw files list from API (for search across all files) */
  filesList: FilesetFileOutput[] | undefined;
  isUploading: boolean;
  isFilesFetching: boolean;
  pendingUploads?: (URL | File | FileContentsWithPath)[];
  pendingFileOid?: string;
}

export interface UseFileActionsResult {
  sortOrder: SortOrder;
  sortFiles: (sortBy: 'name' | 'size') => void;
  searchQuery: string;
  handleSearchQueryChange: (value: string, onClearSelection: () => void) => void;
  rowContents: FileSystemNode[];
  treeRows: TreeRow[];
  expandedFolders: Set<string>;
  toggleFolderExpand: (path: string) => void;
  /** Expands a folder and every folder beneath it, or collapses the whole
   *  subtree when it is already fully expanded. */
  toggleFolderSubtreeExpand: (path: string) => void;
  /** True when a folder and all folders beneath it are expanded. */
  isFolderSubtreeExpanded: (path: string) => boolean;
  /** True when a folder contains at least one subfolder — i.e. when the
   *  expand-subtree action does more than a plain toggle. */
  folderHasSubfolders: (path: string) => boolean;
}

/**
 * Manages file table actions: sorting, searching, and row content computation
 * - Sort by name or size with asc/desc order
 * - Search filtering across all files
 * - Includes pending uploads in row contents
 */
export function useFileActions(options: UseFileActionsOptions): UseFileActionsResult {
  const {
    filesList,
    isUploading,
    isFilesFetching,
    pendingUploads,
    pendingFileOid = '------PENDING------',
  } = options;

  const [sortOrder, setSortOrder] = useState<SortOrder>({
    sortBy: 'name',
    order: 'asc',
  });

  const sortFiles = useCallback(
    (sortBy: 'name' | 'size') =>
      setSortOrder((q) => ({
        sortBy: sortBy,
        order: q.sortBy !== sortBy ? 'desc' : q.order === 'desc' ? 'asc' : 'desc',
      })),
    []
  );

  const [searchQuery, setSearchQuery] = useState('');

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(() => new Set());

  const toggleFolderExpand = useCallback((path: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const allFolderPaths = useMemo(() => {
    const paths = new Set<string>();
    for (const file of filesList ?? []) {
      const segments = file.path.split('/').filter(Boolean);
      let acc = '';
      for (let i = 0; i < segments.length - 1; i++) {
        acc = acc ? `${acc}/${segments[i]}` : segments[i];
        paths.add(acc);
      }
    }
    return paths;
  }, [filesList]);

  const folderSubtrees = useMemo(() => {
    const subtrees = new Map<string, string[]>();
    for (const folder of allFolderPaths) {
      const segments = folder.split('/');
      let acc = '';
      for (const segment of segments) {
        acc = acc ? `${acc}/${segment}` : segment;
        const subtree = subtrees.get(acc);
        if (subtree) {
          subtree.push(folder);
        } else {
          subtrees.set(acc, [folder]);
        }
      }
    }
    return subtrees;
  }, [allFolderPaths]);

  const fullyExpandedFolders = useMemo(() => {
    const fullyExpanded = new Set<string>();
    for (const [folder, subtree] of folderSubtrees) {
      if (subtree.every((p) => expandedFolders.has(p))) {
        fullyExpanded.add(folder);
      }
    }
    return fullyExpanded;
  }, [folderSubtrees, expandedFolders]);

  const isFolderSubtreeExpanded = useCallback(
    (path: string) => !folderSubtrees.has(path) || fullyExpandedFolders.has(path),
    [folderSubtrees, fullyExpandedFolders]
  );

  const folderHasSubfolders = useCallback(
    (path: string) => (folderSubtrees.get(path)?.length ?? 0) > 1,
    [folderSubtrees]
  );

  const toggleFolderSubtreeExpand = useCallback(
    (path: string) => {
      const subtree = folderSubtrees.get(path) ?? [];
      setExpandedFolders((prev) => {
        const expand = !subtree.every((p) => prev.has(p));
        const next = new Set(prev);
        for (const p of subtree) {
          if (expand) {
            next.add(p);
          } else {
            next.delete(p);
          }
        }
        return next;
      });
    },
    [folderSubtrees]
  );

  const handleSearchQueryChange = useCallback((value: string, onClearSelection: () => void) => {
    onClearSelection();
    setSearchQuery(value);
  }, []);

  const { rowContents, treeRows } = useMemo(() => {
    // When searching, show flat filtered list (no tree)
    if (searchQuery && filesList) {
      const baseRows: FileSystemNode[] = filesList
        .filter((file) => !isGitkeepPath(file.path))
        .filter((file) => file.path.toLowerCase().includes(searchQuery.toLowerCase()))
        .map(
          (file): FileSystemFile => ({
            type: 'file',
            path: file.path,
            size: file.size,
            oid: file.file_ref,
          })
        );

      if ((isUploading || isFilesFetching) && pendingUploads?.length) {
        const pendingFiles: FileSystemFile[] = pendingUploads
          .filter((file): file is File => file instanceof File)
          .map((file) => ({
            type: 'file',
            size: file.size,
            path: file.name,
            oid: pendingFileOid,
          }));
        baseRows.push(...pendingFiles);
      }

      const sorted = [...baseRows].sort((a, b) => {
        if (sortOrder.sortBy === 'name') {
          return sortOrder.order === 'asc'
            ? a.path.localeCompare(b.path)
            : b.path.localeCompare(a.path);
        }
        return sortOrder.order === 'asc' ? a.size - b.size : b.size - a.size;
      });

      return {
        rowContents: sorted,
        treeRows: sorted.map((node) => ({ node, depth: 0 })),
      };
    }

    // Tree view: build from filesList and flatten with expanded state
    const fileEntries = (filesList ?? []).map(toFileEntry);
    if ((isUploading || isFilesFetching) && pendingUploads?.length) {
      const pendingFiles = pendingUploads
        .filter((file): file is File => file instanceof File)
        .map((file) => ({
          path: file.name,
          size: file.size,
          oid: pendingFileOid,
        }));
      fileEntries.push(...pendingFiles);
    }

    const fileTree = mapFileListToFileTree(fileEntries);
    const flattened = flattenFileTree(fileTree, expandedFolders, sortOrder);

    return {
      rowContents: flattened.map((t) => t.node),
      treeRows: flattened,
    };
  }, [
    filesList,
    searchQuery,
    sortOrder,
    isFilesFetching,
    isUploading,
    pendingUploads,
    pendingFileOid,
    expandedFolders,
  ]);

  return {
    sortOrder,
    sortFiles,
    searchQuery,
    handleSearchQueryChange,
    rowContents,
    treeRows,
    expandedFolders,
    toggleFolderExpand,
    toggleFolderSubtreeExpand,
    isFolderSubtreeExpanded,
    folderHasSubfolders,
  };
}
