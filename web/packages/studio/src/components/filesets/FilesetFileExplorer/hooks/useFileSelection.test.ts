// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFileSelection } from '@studio/components/filesets/FilesetFileExplorer/hooks/useFileSelection';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import { act, renderHook } from '@testing-library/react';

describe('useFileSelection', () => {
  const mockFile1: FileSystemNode = {
    type: 'file',
    path: 'file1.txt',
    size: 100,
    oid: 'oid1',
  };

  const mockFile2: FileSystemNode = {
    type: 'file',
    path: 'file2.txt',
    size: 200,
    oid: 'oid2',
  };

  const mockFolder1: FileSystemNode = {
    type: 'directory',
    path: 'folder1/',
    size: 0,
    oid: 'folder-oid-1',
    children: {},
  };

  const mockFolder2: FileSystemNode = {
    type: 'directory',
    path: 'folder2/',
    size: 0,
    oid: 'folder-oid-2',
    children: {},
  };

  const availableItems = [mockFile1, mockFile2, mockFolder1, mockFolder2];

  const defaultDatasetId = 'test-dataset';

  it('should initialize with empty selection', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    expect(result.current.selectedItems).toEqual([]);
  });

  it('should add item to selection', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);
  });

  it('should add multiple items to selection', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    act(() => {
      result.current.addSelectedItem(mockFile2);
    });

    expect(result.current.selectedItems).toEqual([mockFile1, mockFile2]);
  });

  it('should not add duplicate items to selection', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);
  });

  it('should remove item from selection', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
      result.current.addSelectedItem(mockFile2);
    });

    act(() => {
      result.current.removeSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems).toEqual([mockFile2]);
  });

  it('should handle removing item that is not in selection', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    act(() => {
      result.current.removeSelectedItem(mockFile2);
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);
  });

  it('should clear all selected items', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
      result.current.addSelectedItem(mockFile2);
      result.current.addSelectedItem(mockFolder1);
    });

    expect(result.current.selectedItems.length).toBe(3);

    act(() => {
      result.current.clearSelectedItems();
    });

    expect(result.current.selectedItems).toEqual([]);
  });

  it('should select all available items', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.selectAllItems();
    });

    expect(result.current.selectedItems).toEqual(availableItems);
  });

  it('should replace selection when selecting all', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);

    act(() => {
      result.current.selectAllItems();
    });

    expect(result.current.selectedItems).toEqual(availableItems);
  });

  it('should clear selection when folder changes', () => {
    const { result, rerender } = renderHook(
      ({ availableItems, currentFolder, datasetId }) =>
        useFileSelection(availableItems, currentFolder, datasetId),
      {
        initialProps: {
          availableItems,
          currentFolder: 'folder1/',
          datasetId: defaultDatasetId,
        },
      }
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
      result.current.addSelectedItem(mockFile2);
    });

    expect(result.current.selectedItems.length).toBe(2);

    rerender({
      availableItems,
      currentFolder: 'folder2/',
      datasetId: defaultDatasetId,
    });

    expect(result.current.selectedItems).toEqual([]);
  });

  it('should not clear selection when folder stays the same', () => {
    const { result, rerender } = renderHook(
      ({ availableItems, currentFolder, datasetId }) =>
        useFileSelection(availableItems, currentFolder, datasetId),
      {
        initialProps: {
          availableItems,
          currentFolder: 'folder1/',
          datasetId: defaultDatasetId,
        },
      }
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);

    rerender({
      availableItems,
      currentFolder: 'folder1/',
      datasetId: defaultDatasetId,
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);
  });

  it('should clear selection when navigating from undefined to a folder', () => {
    const { result, rerender } = renderHook(
      ({
        availableItems,
        currentFolder,
        datasetId,
      }: {
        availableItems: FileSystemNode[];
        currentFolder?: string;
        datasetId: string;
      }) => useFileSelection(availableItems, currentFolder, datasetId),
      {
        initialProps: {
          availableItems,
          currentFolder: undefined as string | undefined,
          datasetId: defaultDatasetId,
        },
      }
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems.length).toBe(1);

    rerender({
      availableItems,
      currentFolder: 'folder1/',
      datasetId: defaultDatasetId,
    });

    expect(result.current.selectedItems).toEqual([]);
  });

  it('should clear selection when navigating from a folder to undefined', () => {
    const { result, rerender } = renderHook(
      ({
        availableItems,
        currentFolder,
        datasetId,
      }: {
        availableItems: FileSystemNode[];
        currentFolder?: string;
        datasetId: string;
      }) => useFileSelection(availableItems, currentFolder, datasetId),
      {
        initialProps: {
          availableItems,
          currentFolder: 'folder1/' as string | undefined,
          datasetId: defaultDatasetId,
        },
      }
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems.length).toBe(1);

    rerender({
      availableItems,
      currentFolder: undefined,
      datasetId: defaultDatasetId,
    });

    expect(result.current.selectedItems).toEqual([]);
  });

  it('should clear selection when dataset changes', () => {
    const { result, rerender } = renderHook(
      ({
        availableItems,
        currentFolder,
        datasetId,
      }: {
        availableItems: FileSystemNode[];
        currentFolder?: string;
        datasetId: string;
      }) => useFileSelection(availableItems, currentFolder, datasetId),
      {
        initialProps: {
          availableItems,
          currentFolder: undefined as string | undefined,
          datasetId: 'dataset-1',
        },
      }
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
      result.current.addSelectedItem(mockFile2);
    });

    expect(result.current.selectedItems.length).toBe(2);

    // Switch to a different dataset
    rerender({
      availableItems,
      currentFolder: undefined,
      datasetId: 'dataset-2',
    });

    expect(result.current.selectedItems).toEqual([]);
  });

  it('should not clear selection when dataset stays the same', () => {
    const { result, rerender } = renderHook(
      ({
        availableItems,
        currentFolder,
        datasetId,
      }: {
        availableItems: FileSystemNode[];
        currentFolder?: string;
        datasetId: string;
      }) => useFileSelection(availableItems, currentFolder, datasetId),
      {
        initialProps: {
          availableItems,
          currentFolder: undefined as string | undefined,
          datasetId: defaultDatasetId,
        },
      }
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);

    // Re-render with same datasetId
    rerender({
      availableItems,
      currentFolder: undefined,
      datasetId: defaultDatasetId,
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);
  });

  it('should handle selection of both files and folders', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.addSelectedItem(mockFile1);
      result.current.addSelectedItem(mockFolder1);
    });

    expect(result.current.selectedItems).toEqual([mockFile1, mockFolder1]);
  });

  it('should distinguish between files and folders with same path prefix', () => {
    const fileNode: FileSystemNode = {
      type: 'file',
      path: 'test',
      size: 100,
      oid: 'oid1',
    };

    const folderNode: FileSystemNode = {
      type: 'directory',
      path: 'test',
      size: 0,
      oid: 'folder-oid-test',
      children: {},
    };

    const items = [fileNode, folderNode];
    const { result } = renderHook(() => useFileSelection(items, undefined, defaultDatasetId));

    act(() => {
      result.current.addSelectedItem(fileNode);
      result.current.addSelectedItem(folderNode);
    });

    // Both should be selected as they are different types
    expect(result.current.selectedItems).toEqual([fileNode, folderNode]);
  });

  it('should maintain stable function references', () => {
    const { result, rerender } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    const { addSelectedItem, removeSelectedItem, clearSelectedItems } = result.current;

    rerender();

    expect(result.current.addSelectedItem).toBe(addSelectedItem);
    expect(result.current.removeSelectedItem).toBe(removeSelectedItem);
    expect(result.current.clearSelectedItems).toBe(clearSelectedItems);
    // Note: selectAllItems depends on availableItems, so it may change
  });

  it('should update selectAllItems when availableItems change', () => {
    const { result, rerender } = renderHook(
      ({ items }) => useFileSelection(items, undefined, defaultDatasetId),
      {
        initialProps: { items: [mockFile1] },
      }
    );

    act(() => {
      result.current.selectAllItems();
    });

    expect(result.current.selectedItems).toEqual([mockFile1]);

    rerender({ items: [mockFile1, mockFile2] });

    act(() => {
      result.current.selectAllItems();
    });

    expect(result.current.selectedItems).toEqual([mockFile1, mockFile2]);
  });

  it('should handle empty available items', () => {
    const { result } = renderHook(() => useFileSelection([], undefined, defaultDatasetId));

    act(() => {
      result.current.selectAllItems();
    });

    expect(result.current.selectedItems).toEqual([]);
  });

  it('should handle selection operations on empty state', () => {
    const { result } = renderHook(() =>
      useFileSelection(availableItems, undefined, defaultDatasetId)
    );

    act(() => {
      result.current.removeSelectedItem(mockFile1);
    });

    expect(result.current.selectedItems).toEqual([]);

    act(() => {
      result.current.clearSelectedItems();
    });

    expect(result.current.selectedItems).toEqual([]);
  });
});
