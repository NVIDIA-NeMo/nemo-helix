// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFallbackWorkspace } from '@studio/hooks/useFallbackWorkspace';
import { renderHook } from '@testing-library/react';

const { mockList, mockRecent, mockProfile } = vi.hoisted(() => ({
  mockList: vi.fn(),
  mockRecent: vi.fn(),
  mockProfile: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/entity-store', () => ({
  useEntitiesListWorkspaces: mockList,
}));
vi.mock('@studio/providers/auth/useAuthProfile', () => ({ useAuthProfile: mockProfile }));
vi.mock('@studio/providers/auth/useMe', () => ({ useMe: () => ({ isSuccess: true }) }));
vi.mock('@studio/providers/preferences/usePreference', () => ({
  usePreference: () => [mockRecent(), vi.fn(), vi.fn(), true],
}));

const listed = (...names: string[]) => ({
  data: { data: names.map((name) => ({ name })) },
  isSuccess: true,
  isError: false,
});

describe('useFallbackWorkspace', () => {
  beforeEach(() => mockProfile.mockReturnValue(undefined));

  it('prefers the most recent accessible workspace, never the excluded one', () => {
    mockRecent.mockReturnValue(['gone', 'forbidden', 'team']);
    mockList.mockReturnValue(listed('forbidden', 'team', 'other'));
    expect(renderHook(() => useFallbackWorkspace(['forbidden'])).result.current).toBe('team');
  });

  it('falls back to the first listed workspace, then default', () => {
    mockRecent.mockReturnValue([]);
    mockList.mockReturnValue(listed('alpha', 'beta'));
    expect(renderHook(() => useFallbackWorkspace(['alpha'])).result.current).toBe('beta');

    mockList.mockReturnValue(listed());
    expect(renderHook(() => useFallbackWorkspace()).result.current).toBe('default');
    expect(renderHook(() => useFallbackWorkspace(['default'])).result.current).toBeUndefined();
  });

  it('uses recent workspaces optimistically while the list loads', () => {
    mockRecent.mockReturnValue(['recent']);
    mockList.mockReturnValue({ data: undefined, isSuccess: false, isError: false });
    expect(renderHook(() => useFallbackWorkspace()).result.current).toBe('recent');
  });

  it('prefers the personal workspace over other accessible ones when there are no recents', () => {
    mockRecent.mockReturnValue([]);
    mockProfile.mockReturnValue({ workspace: 'jdoe' });
    mockList.mockReturnValue(listed('default', 'shared', 'jdoe'));
    expect(renderHook(() => useFallbackWorkspace()).result.current).toBe('jdoe');
  });

  it('never returns an excluded workspace when the list fails', () => {
    mockRecent.mockReturnValue(['a', 'b']);
    mockList.mockReturnValue({ data: undefined, isSuccess: false, isError: true });
    expect(renderHook(() => useFallbackWorkspace(['a'])).result.current).toBe('b');
    expect(renderHook(() => useFallbackWorkspace(['a', 'b'])).result.current).toBe('default');
    expect(
      renderHook(() => useFallbackWorkspace(['a', 'b', 'default'])).result.current
    ).toBeUndefined();
  });
});
