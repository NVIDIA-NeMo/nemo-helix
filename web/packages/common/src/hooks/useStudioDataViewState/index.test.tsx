// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState/index';
import { renderHook, act } from '@testing-library/react';
import { MemoryRouter, useSearchParams } from 'react-router';

// Track mock state for DataView
const mockPaginationSet = vi.fn();
const mockSortingSet = vi.fn();
const mockSearchBarSet = vi.fn();
const mockColumnFilteringSet = vi.fn();
let mockPaginationState = { pageIndex: 0, pageSize: 50 };
let mockSortingState: Array<{ id: string; desc: boolean }> = [];
let mockSearchBarState = '';
let mockColumnFilteringState: Array<{ id: string; value: unknown }> = [];
// Track whether mock has been initialized (to avoid re-initializing on rerender)
let mockInitialized = false;
let lastDataViewOptions: Record<string, unknown> | undefined;

// Mock DataView module
vi.mock('@nemo/common/src/components/DataView/internal', () => ({
  useDataViewState: vi.fn((options) => {
    lastDataViewOptions = options;
    // Only initialize from options on first call (simulates real useDataViewState behavior)
    // Real useDataViewState uses options as initial values, then maintains internal state
    if (!mockInitialized) {
      mockInitialized = true;
      if (options?.pagination) {
        mockPaginationState = { ...options.pagination };
      }
      // Mirror the real useSortingState: the initial sort is always an ordered array.
      mockSortingState = options?.sorting ?? [];
      if (options?.searchBar) {
        mockSearchBarState = options.searchBar;
      }
      if (options?.columnFilters) {
        mockColumnFilteringState = options.columnFilters;
      }
    }
    return {
      pagination: {
        state: mockPaginationState,
        set: mockPaginationSet,
      },
      sorting: {
        state: mockSortingState,
        set: mockSortingSet,
      },
      columnPinning: {
        state: options?.columnPinning ?? {},
        set: vi.fn(),
      },
      rowSelection: {
        state: {},
        set: vi.fn(),
      },
      searchBar: {
        state: mockSearchBarState,
        set: mockSearchBarSet,
      },
      columnFiltering: {
        state: mockColumnFilteringState,
        set: mockColumnFilteringSet,
      },
    };
  }),
}));

// Mock use-debounce to return values immediately for testing
vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value],
}));

// Helper to capture the search params state in tests
let currentSearchParams: URLSearchParams;

// Create a wrapper component that captures search params
const createWrapper =
  (initialEntries: string[]) =>
  ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter initialEntries={initialEntries}>
      <SearchParamsCapture>{children}</SearchParamsCapture>
    </MemoryRouter>
  );

// Component to capture current search params
const SearchParamsCapture = ({ children }: { children: React.ReactNode }) => {
  const [searchParams] = useSearchParams();
  currentSearchParams = searchParams;
  return <>{children}</>;
};

describe('useStudioDataViewState', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPaginationState = { pageIndex: 0, pageSize: 50 };
    mockSortingState = [];
    mockSearchBarState = '';
    mockColumnFilteringState = [];
    mockInitialized = false;
    lastDataViewOptions = undefined;
  });

  describe('initial pagination state from URL', () => {
    it('should initialize DataView with pagination from URL', () => {
      const wrapper = createWrapper(['/?page=3&page_size=25']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      // Page 3 (1-based) should become pageIndex 2 (0-based)
      expect(mockPaginationState.pageIndex).toBe(2);
      expect(mockPaginationState.pageSize).toBe(25);
    });

    it('should use default values when URL has no pagination params', () => {
      const wrapper = createWrapper(['/']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      // Default: page 1 → pageIndex 0, pageSize 50
      expect(mockPaginationState.pageIndex).toBe(0);
      expect(mockPaginationState.pageSize).toBe(50);
    });

    it('should use custom default values when provided', () => {
      const wrapper = createWrapper(['/']);

      renderHook(
        () =>
          useStudioDataViewState({
            defaultPage: 2,
            defaultPageSize: 25,
          }),
        { wrapper }
      );

      // Custom defaults: page 2 → pageIndex 1, pageSize 25
      expect(mockPaginationState.pageIndex).toBe(1);
      expect(mockPaginationState.pageSize).toBe(25);
    });

    it('should prefer URL params over custom defaults', () => {
      const wrapper = createWrapper(['/?page=5&page_size=100']);

      renderHook(
        () =>
          useStudioDataViewState({
            defaultPage: 2,
            defaultPageSize: 25,
          }),
        { wrapper }
      );

      expect(mockPaginationState.pageIndex).toBe(4);
      expect(mockPaginationState.pageSize).toBe(100);
    });

    it('should handle invalid page param', () => {
      const wrapper = createWrapper(['/?page=invalid']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      // Should use default
      expect(mockPaginationState.pageIndex).toBe(0);
    });

    it('should handle negative page param', () => {
      const wrapper = createWrapper(['/?page=-5']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      // Should use default
      expect(mockPaginationState.pageIndex).toBe(0);
    });

    it('should handle zero page param', () => {
      const wrapper = createWrapper(['/?page=0']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      // Should use default
      expect(mockPaginationState.pageIndex).toBe(0);
    });
  });

  describe('initial sorting state from URL', () => {
    it('should initialize DataView with descending sort from URL', () => {
      const wrapper = createWrapper(['/?sort=-created_at']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.sorting.state).toEqual([{ id: 'created_at', desc: true }]);
    });

    it('should initialize DataView with ascending sort from URL', () => {
      const wrapper = createWrapper(['/?sort=name']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.sorting.state).toEqual([{ id: 'name', desc: false }]);
    });

    it('parses an ordered multi-field sort from the URL when multiSort is enabled', () => {
      const wrapper = createWrapper(['/?sort=-cost_usd.mean,name']);

      const { result } = renderHook(() => useStudioDataViewState({ multiSort: true }), { wrapper });

      expect(result.current.sorting.state).toEqual([
        { id: 'cost_usd.mean', desc: true },
        { id: 'name', desc: false },
      ]);
    });

    it('reads only the first field when multiSort is disabled (single-sort default)', () => {
      // A multi-field URL must be truncated to its first field in single-sort mode, not parsed as
      // one malformed comma-joined id.
      const wrapper = createWrapper(['/?sort=-cost_usd.mean,name']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.sorting.state).toEqual([{ id: 'cost_usd.mean', desc: true }]);
    });

    it('should use defaultSort when URL has no sort param', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(
        () =>
          useStudioDataViewState({
            defaultSort: [{ id: 'updated_at', desc: true }],
          }),
        { wrapper }
      );

      expect(result.current.sorting.state).toEqual([{ id: 'updated_at', desc: true }]);
    });

    it('should handle ascending defaultSort', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(
        () =>
          useStudioDataViewState({
            defaultSort: [{ id: 'name', desc: false }],
          }),
        { wrapper }
      );

      expect(result.current.sorting.state).toEqual([{ id: 'name', desc: false }]);
    });

    it('should prefer URL sort over defaultSort', () => {
      const wrapper = createWrapper(['/?sort=name']);

      const { result } = renderHook(
        () =>
          useStudioDataViewState({
            defaultSort: [{ id: 'created_at', desc: true }],
          }),
        { wrapper }
      );

      expect(result.current.sorting.state).toEqual([{ id: 'name', desc: false }]);
    });
  });

  describe('pagination sync to URL', () => {
    it('should update URL when DataView pagination changes', () => {
      const wrapper = createWrapper(['/']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate DataView pagination change
      act(() => {
        mockPaginationState = { pageIndex: 2, pageSize: 50 };
      });

      rerender();

      // URL should have page=3 (1-based)
      expect(currentSearchParams.get('page')).toBe('3');
    });

    it('should remove page param when returning to default', () => {
      const wrapper = createWrapper(['/?page=5']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate returning to default page
      act(() => {
        mockPaginationState = { pageIndex: 0, pageSize: 50 };
      });

      rerender();

      // page param should be removed (matches default)
      expect(currentSearchParams.get('page')).toBeNull();
    });

    it('should update page_size in URL when it changes', () => {
      const wrapper = createWrapper(['/']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate page size change
      act(() => {
        mockPaginationState = { pageIndex: 0, pageSize: 100 };
      });

      rerender();

      expect(currentSearchParams.get('page_size')).toBe('100');
    });

    it('should remove page_size param when returning to default', () => {
      const wrapper = createWrapper(['/?page_size=100']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate returning to default page size
      act(() => {
        mockPaginationState = { pageIndex: 0, pageSize: 50 };
      });

      rerender();

      // page_size param should be removed (matches default)
      expect(currentSearchParams.get('page_size')).toBeNull();
    });
  });

  describe('sorting sync to URL', () => {
    it('should update URL with descending sort when DataView sorting changes', () => {
      const wrapper = createWrapper(['/']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate DataView sorting change
      act(() => {
        mockSortingState = [{ id: 'name', desc: true }];
      });

      rerender();

      // URL should have sort=-name
      expect(currentSearchParams.get('sort')).toBe('-name');
    });

    it('should update URL with ascending sort when DataView sorting changes', () => {
      const wrapper = createWrapper(['/?sort=-created_at']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate DataView sorting change to ascending
      act(() => {
        mockSortingState = [{ id: 'updated_at', desc: false }];
      });

      rerender();

      // URL should have sort=updated_at
      expect(currentSearchParams.get('sort')).toBe('updated_at');
    });

    it('should remove sort param when sorting matches defaultSort', () => {
      const wrapper = createWrapper(['/?sort=name']);

      const { rerender } = renderHook(
        () =>
          useStudioDataViewState({
            defaultSort: [{ id: 'created_at', desc: true }],
          }),
        { wrapper }
      );

      // Simulate changing back to default sort
      act(() => {
        mockSortingState = [{ id: 'created_at', desc: true }];
      });

      rerender();

      // sort param should be removed (matches default)
      expect(currentSearchParams.get('sort')).toBeNull();
    });

    it('should remove sort param when DataView sorting is cleared', () => {
      const wrapper = createWrapper(['/?sort=-created_at']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate clearing sorting
      act(() => {
        mockSortingState = [];
      });

      rerender();

      // sort param should be removed
      expect(currentSearchParams.get('sort')).toBeNull();
    });
  });

  describe('return value', () => {
    it('should return a complete DataViewState object', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current).toHaveProperty('pagination');
      expect(result.current).toHaveProperty('sorting');
      expect(result.current).toHaveProperty('rowSelection');
      expect(result.current.pagination).toHaveProperty('state');
      expect(result.current.pagination).toHaveProperty('set');
      expect(result.current.sorting).toHaveProperty('state');
      expect(result.current.sorting).toHaveProperty('set');
    });

    it('should return pagination with correct initial state', () => {
      const wrapper = createWrapper(['/?page=5&page_size=100']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.pagination.state).toEqual({
        pageIndex: 4,
        pageSize: 100,
      });
    });

    it('should return sorting with correct initial state', () => {
      const wrapper = createWrapper(['/?sort=-updated_at']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.sorting.state).toEqual([{ id: 'updated_at', desc: true }]);
    });
  });

  describe('edge cases', () => {
    it('should handle page=1 correctly', () => {
      const wrapper = createWrapper(['/?page=1']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(mockPaginationState.pageIndex).toBe(0);
    });

    it('should handle very large page numbers', () => {
      const wrapper = createWrapper(['/?page=999999']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(mockPaginationState.pageIndex).toBe(999998);
    });

    it('should handle no sort in URL and no defaultSort', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.sorting.state).toEqual([]);
    });

    it('should preserve other URL params when updating pagination', () => {
      const wrapper = createWrapper(['/?filter=active&sort=-created_at']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate pagination change
      act(() => {
        mockPaginationState = { pageIndex: 2, pageSize: 50 };
      });

      rerender();

      // Other params should be preserved
      expect(currentSearchParams.get('filter')).toBe('active');
      expect(currentSearchParams.get('sort')).toBe('-created_at');
      expect(currentSearchParams.get('page')).toBe('3');
    });

    it('should preserve other URL params when updating sort', () => {
      const wrapper = createWrapper(['/?page=2&filter=active&sort=-created_at']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Simulate sorting change
      act(() => {
        mockSortingState = [{ id: 'name', desc: false }];
      });

      rerender();

      // Other params should be preserved
      expect(currentSearchParams.get('page')).toBe('2');
      expect(currentSearchParams.get('filter')).toBe('active');
      expect(currentSearchParams.get('sort')).toBe('name');
    });
  });

  describe('real-world scenarios', () => {
    it('should handle typical table initialization with sorting', () => {
      const wrapper = createWrapper(['/datasets']);

      const { result } = renderHook(
        () =>
          useStudioDataViewState({
            defaultSort: [{ id: 'created_at', desc: true }],
          }),
        { wrapper }
      );

      expect(result.current.pagination.state).toEqual({
        pageIndex: 0,
        pageSize: 50,
      });
      expect(result.current.sorting.state).toEqual([{ id: 'created_at', desc: true }]);
    });

    it('should handle bookmarked URL with pagination and sorting', () => {
      const wrapper = createWrapper(['/datasets?page=5&page_size=100&sort=name']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.pagination.state).toEqual({
        pageIndex: 4,
        pageSize: 100,
      });
      expect(result.current.sorting.state).toEqual([{ id: 'name', desc: false }]);
    });

    it('should handle complete URL with all params', () => {
      const wrapper = createWrapper(['/datasets?page=3&page_size=25&sort=-updated_at']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.pagination.state).toEqual({
        pageIndex: 2,
        pageSize: 25,
      });
      expect(result.current.sorting.state).toEqual([{ id: 'updated_at', desc: true }]);
    });
  });

  describe('URL to DataView sync (browser navigation)', () => {
    it('should update DataView pagination when URL changes externally', () => {
      // Start at page 3
      const wrapper = createWrapper(['/?page=3']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(mockPaginationState.pageIndex).toBe(2);

      // Simulate browser back to page 1 by updating the URL
      // In real usage, this would be triggered by browser history navigation
      // The hook should detect URL change and call set() to update DataView
      expect(mockPaginationSet).not.toHaveBeenCalled();
    });

    it('should update DataView sorting when URL changes externally', () => {
      // Start with descending sort
      const wrapper = createWrapper(['/?sort=-created_at']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(mockSortingState).toEqual([{ id: 'created_at', desc: true }]);

      // The hook should be ready to sync URL changes to DataView
      expect(mockSortingSet).not.toHaveBeenCalled();
    });

    it('should not cause infinite loops when DataView and URL sync', () => {
      const wrapper = createWrapper(['/']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Initial state
      expect(mockPaginationState.pageIndex).toBe(0);

      // Simulate DataView change
      act(() => {
        mockPaginationState = { pageIndex: 2, pageSize: 50 };
      });

      rerender();

      // URL should update
      expect(currentSearchParams.get('page')).toBe('3');

      // Should not call set() in a loop - only updateUrlParams should have been called
      // The URL→DataView sync should not trigger because DataView initiated the change
      expect(mockPaginationSet).not.toHaveBeenCalled();
    });

    it('should sync URL to DataView when DataView did not change', () => {
      // This test verifies the URL→DataView sync path
      // When URL differs from DataView but DataView didn't change (e.g., browser back),
      // the hook should call set() to sync DataView to URL

      const wrapper = createWrapper(['/?page=5']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      // Initial state from URL
      expect(mockPaginationState.pageIndex).toBe(4);

      // The set function should not have been called during normal initialization
      // (only during URL→DataView sync after URL changes externally)
      expect(mockPaginationSet).not.toHaveBeenCalled();
    });
  });

  describe('pagination reset on sort change', () => {
    it('should reset pagination to page 1 when sorting changes', () => {
      const wrapper = createWrapper(['/?page=5']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Start on page 5
      expect(mockPaginationState.pageIndex).toBe(4);

      // Simulate sorting change
      act(() => {
        mockSortingState = [{ id: 'name', desc: true }];
      });

      rerender();

      // Pagination should be reset - set() should have been called with pageIndex: 0
      expect(mockPaginationSet).toHaveBeenCalledWith(expect.any(Function));
    });

    it('should not reset pagination if already on page 1', () => {
      const wrapper = createWrapper(['/']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Already on page 1
      expect(mockPaginationState.pageIndex).toBe(0);

      // Simulate sorting change
      act(() => {
        mockSortingState = [{ id: 'name', desc: true }];
      });

      rerender();

      // Should not call pagination set since already on page 0
      expect(mockPaginationSet).not.toHaveBeenCalled();
    });
  });

  describe('column pinning defaults', () => {
    it('should pass default column pinning with row-selection left and row-actions right', () => {
      const wrapper = createWrapper(['/']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(lastDataViewOptions).toEqual(
        expect.objectContaining({
          columnPinning: {
            left: ['row-selection'],
            right: ['row-actions'],
          },
        })
      );
    });

    it('should allow consumers to override column pinning via options', () => {
      const wrapper = createWrapper(['/']);

      renderHook(
        () =>
          useStudioDataViewState({
            columnPinning: { left: ['custom-col'], right: [] },
          }),
        { wrapper }
      );

      expect(lastDataViewOptions).toEqual(
        expect.objectContaining({
          columnPinning: { left: ['custom-col'], right: [] },
        })
      );
    });
  });

  describe('debouncedSearchBar', () => {
    it('should return the current search bar value', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.debouncedSearchBar).toBe('');
    });

    it('should reflect search bar state changes', () => {
      const wrapper = createWrapper(['/']);

      const { result, rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      act(() => {
        mockSearchBarState = 'test query';
      });

      rerender();

      expect(result.current.debouncedSearchBar).toBe('test query');
    });
  });

  describe('debouncedColumnFilters', () => {
    it('should return empty array by default', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.debouncedColumnFilters).toEqual([]);
    });

    it('should reflect column filter state from URL', () => {
      const filters = JSON.stringify([{ id: 'storage_type', value: 's3' }]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.debouncedColumnFilters).toEqual([{ id: 'storage_type', value: 's3' }]);
    });
  });

  describe('apiFilter', () => {
    it('should return empty object when no searchText or filters', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter).toEqual({});
    });

    it('should expose debounced searchBar as searchText', () => {
      const wrapper = createWrapper(['/?s=test']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter.searchText).toBe('test');
    });

    it('should not include searchText when searchBar param is present but empty', () => {
      const wrapper = createWrapper(['/?s=']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter).toEqual({});
    });

    it('should map columnFilters to filter object', () => {
      const filters = JSON.stringify([
        { id: 'storage_type', value: 's3' },
        { id: 'status', value: 'active' },
      ]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter.filter).toEqual({
        storage_type: 's3',
        status: 'active',
      });
    });

    it('should exclude empty and undefined filter values', () => {
      const filters = JSON.stringify([
        { id: 'storage_type', value: 's3' },
        { id: 'empty', value: '' },
        { id: 'undef', value: undefined },
      ]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter.filter).toEqual({ storage_type: 's3' });
    });

    it('should exclude empty multi-select object filter values', () => {
      const filters = JSON.stringify([
        { id: 'storage_type', value: 's3' },
        { id: 'customization_type', value: {} },
      ]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter.filter).toEqual({ storage_type: 's3' });
    });

    it('should include populated multi-select object filter values', () => {
      const filters = JSON.stringify([
        { id: 'customization_type', value: { lora: true, full: true } },
      ]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter.filter).toEqual({
        customization_type: { lora: true, full: true },
      });
    });

    it('should include both searchText and filter when both are present', () => {
      const filters = JSON.stringify([{ id: 'type', value: 'local' }]);
      const wrapper = createWrapper([`/?s=query&filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.apiFilter).toEqual({
        searchText: 'query',
        filter: { type: 'local' },
      });
    });

    it('should remap a column filter id to its API key via filterFieldMap', () => {
      const filters = JSON.stringify([{ id: 'latency_ms', value: { $gte: 5, $lte: 10 } }]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(
        () => useStudioDataViewState({ filterFieldMap: { latency_ms: 'latency_ms.mean' } }),
        { wrapper }
      );

      expect(result.current.apiFilter.filter).toEqual({
        'latency_ms.mean': { $gte: 5, $lte: 10 },
      });
    });

    it('should leave unmapped column filter ids under their own id', () => {
      const filters = JSON.stringify([
        { id: 'latency_ms', value: { $gte: 5 } },
        { id: 'storage_type', value: 's3' },
      ]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(
        () => useStudioDataViewState({ filterFieldMap: { latency_ms: 'latency_ms.mean' } }),
        { wrapper }
      );

      expect(result.current.apiFilter.filter).toEqual({
        'latency_ms.mean': { $gte: 5 },
        storage_type: 's3',
      });
    });

    it('should remap column filter ids via a function-form filterFieldMap', () => {
      const filters = JSON.stringify([
        { id: 'evaluator-accuracy', value: { $gte: 0.8, $lte: 1 } },
        { id: 'cost_usd', value: { $lte: 0.5 } },
      ]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(
        () =>
          useStudioDataViewState({
            filterFieldMap: (id) => {
              if (id === 'cost_usd') return 'cost_usd.mean';
              const match = id.match(/^evaluator-(.+)$/);
              return match ? `evaluators.${match[1]}.mean` : undefined;
            },
          }),
        { wrapper }
      );

      expect(result.current.apiFilter.filter).toEqual({
        'evaluators.accuracy.mean': { $gte: 0.8, $lte: 1 },
        'cost_usd.mean': { $lte: 0.5 },
      });
    });

    it('should leave a column filter id under its own id when the function returns undefined', () => {
      const filters = JSON.stringify([{ id: 'storage_type', value: 's3' }]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(
        () => useStudioDataViewState({ filterFieldMap: () => undefined }),
        { wrapper }
      );

      expect(result.current.apiFilter.filter).toEqual({ storage_type: 's3' });
    });
  });

  describe('resetFilters', () => {
    it('should clear searchBar, columnFilters, and reset pagination', () => {
      const filters = JSON.stringify([{ id: 'type', value: 's3' }]);
      const wrapper = createWrapper([`/?page=3&s=test&filters=${encodeURIComponent(filters)}`]);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      act(() => {
        result.current.resetFilters();
      });

      expect(mockSearchBarSet).toHaveBeenCalledWith('');
      expect(mockColumnFilteringSet).toHaveBeenCalledWith([]);
    });

    it('should return resetFilters function', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(typeof result.current.resetFilters).toBe('function');
    });
  });

  describe('search bar URL sync', () => {
    it('should initialize searchBar from URL ?s= param', () => {
      const wrapper = createWrapper(['/?s=hello']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(lastDataViewOptions).toEqual(expect.objectContaining({ searchBar: 'hello' }));
    });

    it('should update URL when searchBar changes', () => {
      const wrapper = createWrapper(['/']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      act(() => {
        mockSearchBarState = 'search text';
      });

      rerender();

      expect(currentSearchParams.get('s')).toBe('search text');
    });

    it('should remove ?s= param when searchBar is cleared', () => {
      const wrapper = createWrapper(['/?s=hello']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      act(() => {
        mockSearchBarState = '';
      });

      rerender();

      expect(currentSearchParams.get('s')).toBeNull();
    });
  });

  describe('column filters URL sync', () => {
    it('should initialize columnFilters from URL ?filters= param', () => {
      const filters = JSON.stringify([{ id: 'type', value: 's3' }]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(lastDataViewOptions).toEqual(
        expect.objectContaining({
          columnFilters: [{ id: 'type', value: 's3' }],
        })
      );
    });

    it('should update URL when columnFilters change', () => {
      const wrapper = createWrapper(['/']);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      act(() => {
        mockColumnFilteringState = [{ id: 'storage_type', value: 's3' }];
      });

      rerender();

      const filtersParam = currentSearchParams.get('filters');
      expect(filtersParam).toBeTruthy();
      const decoded = JSON.parse(decodeURIComponent(filtersParam!));
      expect(decoded).toEqual([{ id: 'storage_type', value: 's3' }]);
    });

    it('should remove ?filters= param when columnFilters are cleared', () => {
      const filters = JSON.stringify([{ id: 'type', value: 's3' }]);
      const wrapper = createWrapper([`/?filters=${encodeURIComponent(filters)}`]);

      const { rerender } = renderHook(() => useStudioDataViewState(), { wrapper });

      act(() => {
        mockColumnFilteringState = [];
      });

      rerender();

      expect(currentSearchParams.get('filters')).toBeNull();
    });

    it('should handle invalid JSON in ?filters= param', () => {
      const wrapper = createWrapper(['/?filters=not-json']);

      renderHook(() => useStudioDataViewState(), { wrapper });

      expect(lastDataViewOptions).toEqual(expect.objectContaining({ columnFilters: [] }));
    });
  });

  describe('resetPagination helper', () => {
    it('should return resetPagination function', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      expect(result.current.resetPagination).toBeDefined();
      expect(typeof result.current.resetPagination).toBe('function');
    });

    it('should call pagination.set when resetPagination is called and not on page 0', () => {
      const wrapper = createWrapper(['/?page=5']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Start on page 5
      expect(mockPaginationState.pageIndex).toBe(4);

      act(() => {
        result.current.resetPagination();
      });

      // Should have called pagination.set
      expect(mockPaginationSet).toHaveBeenCalledWith(expect.any(Function));
    });

    it('should call rowSelection.set to clear selection when resetPagination is called', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      const mockRowSelectionSet = result.current.rowSelection.set as ReturnType<typeof vi.fn>;

      act(() => {
        result.current.resetPagination();
      });

      // Should have cleared row selection
      expect(mockRowSelectionSet).toHaveBeenCalledWith({});
    });

    it('should not call pagination.set if already on page 0', () => {
      const wrapper = createWrapper(['/']);

      const { result } = renderHook(() => useStudioDataViewState(), { wrapper });

      // Already on page 0
      expect(mockPaginationState.pageIndex).toBe(0);

      act(() => {
        result.current.resetPagination();
      });

      // Should not have called pagination.set
      expect(mockPaginationSet).not.toHaveBeenCalled();
    });
  });
});
