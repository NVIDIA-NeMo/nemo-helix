// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import {
  strictFilesetNameCandidate,
  useFilesetNameStatus,
} from '@studio/components/filesets/hooks/useFilesetNameStatus';
import { act, renderHook } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/files');

const mockListFilesets = vi.mocked(useFilesListFilesets);

const listResult = (rows: unknown[], overrides: { isFetching?: boolean; isError?: boolean } = {}) =>
  ({
    data: { data: rows },
    isFetching: false,
    isError: false,
    ...overrides,
  }) as unknown as ReturnType<typeof useFilesListFilesets>;

/** Renders the hook with `candidate` and lets the debounce settle on it. */
const settle = (candidate: string, options?: { ownName?: string }) => {
  const view = renderHook(({ value }) => useFilesetNameStatus('default', value, options), {
    initialProps: { value: candidate },
  });
  act(() => {
    vi.advanceTimersByTime(DEFAULT_DEBOUNCE_MS);
  });
  return view;
};

describe('strictFilesetNameCandidate', () => {
  it('trims a valid name', () => {
    expect(strictFilesetNameCandidate('  my-data  ')).toBe('my-data');
  });

  it.each([undefined, '', '   '])(
    'gives nothing to check for %j, which produces no salvageable name',
    (name) => {
      expect(strictFilesetNameCandidate(name)).toBe('');
    }
  );

  it('returns the sanitized name for salvageable inputs', () => {
    expect(strictFilesetNameCandidate('Bad Name')).toBe('bad-name');
    expect(strictFilesetNameCandidate('1starts-with-digit')).toBe('starts-with-digit');
  });
});

describe('useFilesetNameStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockListFilesets.mockReset();
    mockListFilesets.mockReturnValue(listResult([]));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('looks up the name in the given workspace only', () => {
    const { result } = settle('my-data');

    expect(result.current).toBe('available');
    expect(mockListFilesets).toHaveBeenLastCalledWith(
      'default',
      { page_size: 1, filter: { name: 'my-data' } },
      { query: { enabled: true, staleTime: 0, retry: false } }
    );
  });

  it('reports a conflict when the workspace already has the name', () => {
    mockListFilesets.mockReturnValue(listResult([{ name: 'my-data' }]));

    expect(settle('my-data').result.current).toBe('conflict');
  });

  it('reports a failed check rather than claiming the name is free', () => {
    mockListFilesets.mockReturnValue(listResult([], { isError: true }));

    expect(settle('my-data').result.current).toBe('failed');
  });

  it('reads as checking while the request is in flight', () => {
    mockListFilesets.mockReturnValue(listResult([], { isFetching: true }));

    expect(settle('my-data').result.current).toBe('checking');
  });

  it('reads as checking until the debounce catches up with a new name', () => {
    const { result, rerender } = settle('my-data');
    expect(result.current).toBe('available');

    rerender({ value: 'my-data-2' });
    expect(result.current).toBe('checking');

    act(() => {
      vi.advanceTimersByTime(DEFAULT_DEBOUNCE_MS);
    });
    expect(result.current).toBe('available');
  });

  it('does not check an empty candidate', () => {
    expect(settle('').result.current).toBeUndefined();
    expect(mockListFilesets).toHaveBeenLastCalledWith(
      'default',
      { page_size: 1, filter: { name: '' } },
      { query: { enabled: false, staleTime: 0, retry: false } }
    );
  });

  it('does not count a fileset the form created itself as a conflict', () => {
    mockListFilesets.mockReturnValue(listResult([{ name: 'my-data' }]));

    expect(settle('my-data', { ownName: 'my-data' }).result.current).toBeUndefined();
    expect(mockListFilesets).toHaveBeenLastCalledWith(
      'default',
      { page_size: 1, filter: { name: 'my-data' } },
      { query: { enabled: false, staleTime: 0, retry: false } }
    );
  });
});
