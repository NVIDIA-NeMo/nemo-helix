// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { useFileActions } from '@studio/components/filesets/FilesetFileExplorer/hooks/useFileActions';
import { type FileSystemFile, GITKEEP_FILENAME } from '@studio/components/FilesTable/utils';
import { act, renderHook } from '@testing-library/react';

describe('useFileActions', () => {
  // Tree nodes (output of useDatasetNavigator)
  const mockFile1: FileSystemFile = {
    type: 'file',
    path: 'apple.txt',
    size: 100,
    oid: 'oid1',
  };

  const mockFile2: FileSystemFile = {
    type: 'file',
    path: 'banana.txt',
    size: 200,
    oid: 'oid2',
  };

  const mockFile3: FileSystemFile = {
    type: 'file',
    path: 'cherry.txt',
    size: 50,
    oid: 'oid3',
  };

  // Raw API response files (FilesetFileOutput[])
  const mockApiFile1: FilesetFileOutput = {
    path: 'apple.txt',
    size: 100,
    file_ref: 'oid1',
    file_url: 'https://example.com/apple.txt',
  };

  const mockApiFile2: FilesetFileOutput = {
    path: 'banana.txt',
    size: 200,
    file_ref: 'oid2',
    file_url: 'https://example.com/banana.txt',
  };

  const mockApiFile3: FilesetFileOutput = {
    path: 'cherry.txt',
    size: 50,
    file_ref: 'oid3',
    file_url: 'https://example.com/cherry.txt',
  };

  const allFilesList: FilesetFileOutput[] = [mockApiFile1, mockApiFile2, mockApiFile3];

  // Tree built from filesList produces nodes from paths - for root-level files, order is sorted by name
  const expectedTreeFromAllFiles = [mockFile1, mockFile2, mockFile3];

  it('should initialize with default sort order', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    expect(result.current.sortOrder).toEqual({
      sortBy: 'name',
      order: 'asc',
    });
  });

  it('should initialize with empty search query', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    expect(result.current.searchQuery).toBe('');
  });

  it('should display folder contents when no search query', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    // Tree is built from filesList; allFilesList has apple, banana, cherry at root
    expect(result.current.rowContents).toEqual(expectedTreeFromAllFiles);
  });

  it('should sort by name ascending by default', () => {
    const unsortedFilesList = [mockApiFile2, mockApiFile1, mockApiFile3];
    const { result } = renderHook(() =>
      useFileActions({
        filesList: unsortedFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    expect(result.current.rowContents).toEqual([mockFile1, mockFile2, mockFile3]);
  });

  it('should toggle sort order when sorting by same field', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    expect(result.current.sortOrder.order).toBe('asc');

    act(() => {
      result.current.sortFiles('name');
    });

    expect(result.current.sortOrder).toEqual({
      sortBy: 'name',
      order: 'desc',
    });

    act(() => {
      result.current.sortFiles('name');
    });

    expect(result.current.sortOrder).toEqual({
      sortBy: 'name',
      order: 'asc',
    });
  });

  it('should sort by name descending', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    act(() => {
      result.current.sortFiles('name');
    });

    expect(result.current.rowContents).toEqual([mockFile3, mockFile2, mockFile1]);
  });

  it('should change sort field and default to descending', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    act(() => {
      result.current.sortFiles('size');
    });

    expect(result.current.sortOrder).toEqual({
      sortBy: 'size',
      order: 'desc',
    });
  });

  it('should sort by size ascending', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    act(() => {
      result.current.sortFiles('size');
      result.current.sortFiles('size');
    });

    expect(result.current.sortOrder).toEqual({
      sortBy: 'size',
      order: 'asc',
    });
    expect(result.current.rowContents).toEqual([mockFile3, mockFile1, mockFile2]);
  });

  it('should sort by size descending', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    act(() => {
      result.current.sortFiles('size');
    });

    expect(result.current.sortOrder).toEqual({
      sortBy: 'size',
      order: 'desc',
    });
    expect(result.current.rowContents).toEqual([mockFile2, mockFile1, mockFile3]);
  });

  it('should filter files by search query', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const mockClearSelection = vi.fn();

    act(() => {
      result.current.handleSearchQueryChange('banana', mockClearSelection);
    });

    expect(mockClearSelection).toHaveBeenCalledTimes(1);
    expect(result.current.searchQuery).toBe('banana');
    expect(result.current.rowContents).toEqual([mockFile2]);
  });

  it('should search case-insensitively', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const mockClearSelection = vi.fn();

    act(() => {
      result.current.handleSearchQueryChange('BANANA', mockClearSelection);
    });

    expect(result.current.rowContents).toEqual([mockFile2]);
  });

  it('should search across all files, not just folder contents', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const mockClearSelection = vi.fn();

    act(() => {
      result.current.handleSearchQueryChange('cherry', mockClearSelection);
    });

    // cherry.txt is in allFilesList but not in folderContents
    expect(result.current.rowContents).toEqual([mockFile3]);
  });

  it('should return empty results for non-matching search', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const mockClearSelection = vi.fn();

    act(() => {
      result.current.handleSearchQueryChange('nonexistent', mockClearSelection);
    });

    expect(result.current.rowContents).toEqual([]);
  });

  it('should clear search query when setting to empty string', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const mockClearSelection = vi.fn();

    act(() => {
      result.current.handleSearchQueryChange('banana', mockClearSelection);
    });

    expect(result.current.rowContents).toEqual([mockFile2]);

    act(() => {
      result.current.handleSearchQueryChange('', mockClearSelection);
    });

    expect(result.current.searchQuery).toBe('');
    expect(result.current.rowContents).toEqual(expectedTreeFromAllFiles);
  });

  it('should add pending uploads to row contents when uploading', () => {
    const pendingFile = new File(['content'], 'pending.txt', { type: 'text/plain' });
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: [pendingFile],
      })
    );

    const rows = result.current.rowContents;
    const pendingRow = rows.find((r) => r.path === 'pending.txt');

    expect(pendingRow).toBeDefined();
    expect(pendingRow?.type).toBe('file');
    expect(pendingRow?.oid).toBe('------PENDING------');
    expect(pendingRow?.size).toBe(pendingFile.size);
  });

  it('should add pending uploads when fetching files', () => {
    const pendingFile = new File(['content'], 'fetching.txt', { type: 'text/plain' });
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: true,
        pendingUploads: [pendingFile],
      })
    );

    const rows = result.current.rowContents;
    const pendingRow = rows.find((r) => r.path === 'fetching.txt');

    expect(pendingRow).toBeDefined();
  });

  it('should use custom pendingFileOid', () => {
    const pendingFile = new File(['content'], 'custom.txt', { type: 'text/plain' });
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: [pendingFile],
        pendingFileOid: 'CUSTOM-PENDING',
      })
    );

    const rows = result.current.rowContents;
    const pendingRow = rows.find((r) => r.path === 'custom.txt');

    expect(pendingRow?.oid).toBe('CUSTOM-PENDING');
  });

  it('should filter out non-File instances from pending uploads', () => {
    const pendingFile = new File(['content'], 'file.txt', { type: 'text/plain' });
    const pendingUrl = new URL('https://example.com/file.txt');
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: [pendingFile, pendingUrl],
      })
    );

    const rows = result.current.rowContents;
    const pendingRows = rows.filter((r) => r.oid === '------PENDING------');

    // Only the File should be added, not the URL
    expect(pendingRows.length).toBe(1);
    expect(pendingRows[0].path).toBe('file.txt');
  });

  it('should not add pending uploads when not uploading or fetching', () => {
    const pendingFile = new File(['content'], 'pending.txt', { type: 'text/plain' });
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
        pendingUploads: [pendingFile],
      })
    );

    const rows = result.current.rowContents;
    const pendingRow = rows.find((r) => r.path === 'pending.txt');

    expect(pendingRow).toBeUndefined();
  });

  it('should handle empty pending uploads', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: [],
      })
    );

    expect(result.current.rowContents).toEqual(expectedTreeFromAllFiles);
  });

  it('should handle undefined pending uploads', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: undefined,
      })
    );

    expect(result.current.rowContents).toEqual(expectedTreeFromAllFiles);
  });

  it('should sort pending uploads along with other files', () => {
    const pendingFile = new File(['content'], 'zulu.txt', { type: 'text/plain' });
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: [pendingFile],
      })
    );

    const rows = result.current.rowContents;
    const paths = rows.map((r) => r.path);

    // Should be sorted: apple, banana, cherry, zulu
    expect(paths).toEqual(['apple.txt', 'banana.txt', 'cherry.txt', 'zulu.txt']);
  });

  it('should apply search filter before adding pending uploads', () => {
    const pendingFile = new File(['content'], 'pending.txt', { type: 'text/plain' });
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: [pendingFile],
      })
    );

    const mockClearSelection = vi.fn();

    act(() => {
      result.current.handleSearchQueryChange('banana', mockClearSelection);
    });

    // When searching, we use filesList (not folderContents),
    // and pending files are still added
    const rows = result.current.rowContents;
    expect(rows.some((r) => r.path === 'banana.txt')).toBe(true);
    expect(rows.some((r) => r.path === 'pending.txt')).toBe(true);
    expect(rows.some((r) => r.path === 'apple.txt')).toBe(false);
  });

  it('should handle multiple pending uploads', () => {
    const pending1 = new File(['content1'], 'pending1.txt', { type: 'text/plain' });
    const pending2 = new File(['content2'], 'pending2.txt', { type: 'text/plain' });
    const { result } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: true,
        isFilesFetching: false,
        pendingUploads: [pending1, pending2],
      })
    );

    const rows = result.current.rowContents;
    const pendingRows = rows.filter((r) => r.oid === '------PENDING------');

    expect(pendingRows.length).toBe(2);
  });

  it('should maintain stable sortFiles reference', () => {
    const { result, rerender } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const firstSortFiles = result.current.sortFiles;
    rerender();

    expect(result.current.sortFiles).toBe(firstSortFiles);
  });

  it('should maintain stable handleSearchQueryChange reference', () => {
    const { result, rerender } = renderHook(() =>
      useFileActions({
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const firstHandleSearchQueryChange = result.current.handleSearchQueryChange;
    rerender();

    expect(result.current.handleSearchQueryChange).toBe(firstHandleSearchQueryChange);
  });

  it('should recalculate rowContents when dependencies change', () => {
    const { result, rerender } = renderHook((props) => useFileActions(props), {
      initialProps: {
        filesList: allFilesList,
        isUploading: false,
        isFilesFetching: false,
      },
    });

    expect(result.current.rowContents).toEqual(expectedTreeFromAllFiles);

    const newFilesList: FilesetFileOutput[] = [mockApiFile3];
    rerender({
      filesList: newFilesList,
      isUploading: false,
      isFilesFetching: false,
    });

    expect(result.current.rowContents).toEqual([mockFile3]);
  });

  it('should handle undefined filesList', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: undefined,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    // Empty tree when filesList is undefined
    expect(result.current.rowContents).toEqual([]);
  });

  it('should handle search with undefined filesList', () => {
    const { result } = renderHook(() =>
      useFileActions({
        filesList: undefined,
        isUploading: false,
        isFilesFetching: false,
      })
    );

    const mockClearSelection = vi.fn();

    act(() => {
      result.current.handleSearchQueryChange('test', mockClearSelection);
    });

    // When filesList is undefined, search returns empty
    expect(result.current.rowContents).toEqual([]);
  });

  describe('.gitkeep filtering', () => {
    const filesetUrl = (path: string) =>
      `/apis/files/v2/workspaces/default/filesets/test-dataset/-/${path}`;

    const gitkeepRoot: FilesetFileOutput = {
      path: GITKEEP_FILENAME,
      size: 0,
      file_ref: 'gk-root',
      file_url: filesetUrl(GITKEEP_FILENAME),
    };

    const gitkeepInEmptyFolder: FilesetFileOutput = {
      path: `empty-folder/${GITKEEP_FILENAME}`,
      size: 0,
      file_ref: 'gk-empty',
      file_url: filesetUrl(`empty-folder/${GITKEEP_FILENAME}`),
    };

    const realFile: FilesetFileOutput = {
      path: 'populated/data.txt',
      size: 42,
      file_ref: 'data-1',
      file_url: filesetUrl('populated/data.txt'),
    };

    const gitkeepInPopulatedFolder: FilesetFileOutput = {
      path: `populated/${GITKEEP_FILENAME}`,
      size: 0,
      file_ref: 'gk-populated',
      file_url: filesetUrl(`populated/${GITKEEP_FILENAME}`),
    };

    it('omits a root-level .gitkeep from the rendered list', () => {
      const { result } = renderHook(() =>
        useFileActions({
          filesList: [mockApiFile1, gitkeepRoot],
          isUploading: false,
          isFilesFetching: false,
        })
      );

      expect(result.current.rowContents.map((r) => r.path)).toEqual(['apple.txt']);
    });

    it('renders a folder whose only file is .gitkeep as an empty folder when expanded', () => {
      const { result } = renderHook(() =>
        useFileActions({
          filesList: [gitkeepInEmptyFolder],
          isUploading: false,
          isFilesFetching: false,
        })
      );

      act(() => {
        result.current.toggleFolderExpand('empty-folder');
      });

      expect(result.current.treeRows).toEqual([
        {
          node: expect.objectContaining({ type: 'directory', path: 'empty-folder' }),
          depth: 0,
        },
      ]);
    });

    it('renders a folder with .gitkeep + real files showing only the real files', () => {
      const { result } = renderHook(() =>
        useFileActions({
          filesList: [realFile, gitkeepInPopulatedFolder],
          isUploading: false,
          isFilesFetching: false,
        })
      );

      act(() => {
        result.current.toggleFolderExpand('populated');
      });

      const childPaths = result.current.treeRows
        .filter((r) => r.depth === 1)
        .map((r) => r.node.path);

      expect(childPaths).toEqual(['populated/data.txt']);
    });

    it('omits .gitkeep from search results', () => {
      const { result } = renderHook(() =>
        useFileActions({
          filesList: [gitkeepRoot, gitkeepInEmptyFolder, realFile],
          isUploading: false,
          isFilesFetching: false,
        })
      );

      act(() => {
        result.current.handleSearchQueryChange('gitkeep', vi.fn());
      });

      expect(result.current.rowContents).toEqual([]);
    });
  });

  describe('expand folder subtree', () => {
    const filesetUrlFor = (path: string) => `https://example.com/${path}`;
    const nestedFile = (path: string, ref: string): FilesetFileOutput => ({
      path,
      size: 10,
      file_ref: ref,
      file_url: filesetUrlFor(path),
    });

    // docs/
    //   guides/deep/intro.md
    //   readme.md
    // data/train.jsonl
    const nestedFiles: FilesetFileOutput[] = [
      nestedFile('docs/guides/deep/intro.md', 'nested-1'),
      nestedFile('docs/readme.md', 'nested-2'),
      nestedFile('data/train.jsonl', 'nested-3'),
    ];

    const renderNested = () =>
      renderHook(() =>
        useFileActions({
          filesList: nestedFiles,
          isUploading: false,
          isFilesFetching: false,
        })
      );

    it('expands a folder and every folder beneath it', () => {
      const { result } = renderNested();

      act(() => {
        result.current.toggleFolderSubtreeExpand('docs');
      });

      expect(result.current.rowContents.map((r) => r.path)).toEqual([
        'data',
        'docs',
        'docs/guides',
        'docs/guides/deep',
        'docs/guides/deep/intro.md',
        'docs/readme.md',
      ]);
      expect(result.current.isFolderSubtreeExpanded('docs')).toBe(true);
    });

    it('leaves sibling subtrees untouched', () => {
      const { result } = renderNested();

      act(() => {
        result.current.toggleFolderSubtreeExpand('docs');
      });

      expect(result.current.expandedFolders.has('data')).toBe(false);
      expect(result.current.isFolderSubtreeExpanded('data')).toBe(false);
    });

    it('collapses the whole subtree when it is already fully expanded', () => {
      const { result } = renderNested();

      act(() => {
        result.current.toggleFolderSubtreeExpand('docs');
      });
      act(() => {
        result.current.toggleFolderSubtreeExpand('docs');
      });

      expect(result.current.expandedFolders.size).toBe(0);
      expect(result.current.rowContents.map((r) => r.path)).toEqual(['data', 'docs']);
    });

    it('returns to the original expansion state when toggled twice in one batch', () => {
      const { result } = renderNested();

      act(() => {
        result.current.toggleFolderSubtreeExpand('docs');
        result.current.toggleFolderSubtreeExpand('docs');
      });

      expect(result.current.expandedFolders.size).toBe(0);
      expect(result.current.rowContents.map((r) => r.path)).toEqual(['data', 'docs']);
    });

    it('expands the rest of the subtree when it is only partially expanded', () => {
      const { result } = renderNested();

      act(() => {
        result.current.toggleFolderExpand('docs');
      });

      expect(result.current.isFolderSubtreeExpanded('docs')).toBe(false);

      act(() => {
        result.current.toggleFolderSubtreeExpand('docs');
      });

      expect(result.current.expandedFolders.has('docs/guides/deep')).toBe(true);
    });

    it('reports which folders have subfolders', () => {
      const { result } = renderNested();

      expect(result.current.folderHasSubfolders('docs')).toBe(true);
      expect(result.current.folderHasSubfolders('docs/guides')).toBe(true);
      expect(result.current.folderHasSubfolders('docs/guides/deep')).toBe(false);
      expect(result.current.folderHasSubfolders('data')).toBe(false);
    });

    it('does not treat a name-prefixed sibling as a descendant', () => {
      const { result } = renderHook(() =>
        useFileActions({
          filesList: [
            nestedFile('docs/readme.md', 'a'),
            nestedFile('docs-archive/old/readme.md', 'b'),
          ],
          isUploading: false,
          isFilesFetching: false,
        })
      );

      act(() => {
        result.current.toggleFolderSubtreeExpand('docs');
      });

      expect(result.current.expandedFolders.has('docs-archive/old')).toBe(false);
      expect(result.current.folderHasSubfolders('docs')).toBe(false);
    });
  });
});
