// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import { useCallback, useEffect, useState } from 'react';

const getItemId = (item: FileSystemNode) => [item.type, item.path].join('::');

export interface UseFileSelectionResult {
  selectedItems: FileSystemNode[];
  addSelectedItem: (item: FileSystemNode) => void;
  removeSelectedItem: (item: FileSystemNode) => void;
  clearSelectedItems: () => void;
  selectAllItems: () => void;
}

/**
 * Manages file/folder selection state
 * - Multi-select with add/remove
 * - Select all functionality
 * - Auto-clear on folder navigation or dataset change
 */
export function useFileSelection(
  availableItems: FileSystemNode[],
  currentFolder: string | undefined,
  datasetId: string
): UseFileSelectionResult {
  const [selectedItems, setSelectedItems] = useState<FileSystemNode[]>([]);

  const addSelectedItem = useCallback((item: FileSystemNode) => {
    setSelectedItems((prev) =>
      prev.some((i) => getItemId(i) === getItemId(item)) ? prev : [...prev, item]
    );
  }, []);

  const removeSelectedItem = useCallback((item: FileSystemNode) => {
    setSelectedItems((prev) => prev.filter((i) => getItemId(i) !== getItemId(item)));
  }, []);

  const clearSelectedItems = useCallback(() => {
    setSelectedItems([]);
  }, []);

  const selectAllItems = useCallback(() => {
    setSelectedItems(availableItems);
  }, [availableItems]);

  // Clear selection when navigating to a different folder or switching datasets
  useEffect(() => {
    clearSelectedItems();
    // clearSelectedItems is stable (useCallback with empty deps) and doesn't need to be in the dependency array
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentFolder, datasetId]);

  return {
    selectedItems,
    addSelectedItem,
    removeSelectedItem,
    clearSelectedItems,
    selectAllItems,
  };
}
