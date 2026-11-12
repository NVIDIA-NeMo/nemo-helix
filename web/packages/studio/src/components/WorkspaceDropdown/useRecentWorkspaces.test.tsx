// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRecentWorkspaces } from '@studio/components/WorkspaceDropdown/useRecentWorkspaces';
import { renderHook } from '@testing-library/react';

const { mockPreference, mockSet } = vi.hoisted(() => ({
  mockPreference: vi.fn(),
  mockSet: vi.fn(),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPathIfExists: () => 'current',
}));
vi.mock('@studio/providers/preferences/usePreference', () => ({ usePreference: mockPreference }));

describe('useRecentWorkspaces', () => {
  it('waits for the stored list before adding the current workspace', () => {
    mockPreference.mockReturnValue([[], mockSet, vi.fn(), true, false]);
    const { rerender } = renderHook(() => useRecentWorkspaces());
    expect(mockSet).not.toHaveBeenCalled();

    mockPreference.mockReturnValue([['a', 'b'], mockSet, vi.fn(), true, true]);
    rerender();
    rerender();
    expect(mockSet).toHaveBeenCalledTimes(1);
    expect(mockSet).toHaveBeenCalledWith(['current', 'a', 'b']);
  });
});
