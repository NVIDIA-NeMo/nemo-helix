// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceProvider } from '@studio/providers/workspace/WorkspaceProvider';
import { SELECTED_WORKSPACE_KEY } from '@studio/util/localStorage';
import { act, render, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';

vi.mock('@nemo/sdk/generated/platform/entity-store', () => ({
  useEntitiesGetWorkspace: () => ({
    error: undefined,
    isPending: false,
  }),
}));

vi.mock('@studio/providers/auth/useAuthProfile', () => ({
  useAuthProfile: () => ({ workspace: 'profile-workspace' }),
}));

vi.mock('@studio/providers/auth/useAuthTokenStatus', () => ({
  useAuthTokenStatus: () => ({ isTokenActive: true }),
}));

const renderWorkspaceProvider = (workspace: string) =>
  render(
    <MemoryRouter initialEntries={[`/workspaces/${workspace}`]}>
      <Routes>
        <Route
          path="/workspaces/:workspace"
          element={
            <WorkspaceProvider>
              <div>Workspace content</div>
            </WorkspaceProvider>
          }
        />
      </Routes>
    </MemoryRouter>
  );

describe('WorkspaceProvider', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('persists the route workspace', async () => {
    renderWorkspaceProvider('workspace-a');

    await waitFor(() => {
      expect(window.localStorage.getItem(SELECTED_WORKSPACE_KEY)).toBe(
        JSON.stringify('workspace-a')
      );
    });
  });

  it('does not counter-write when another tab changes the stored workspace', () => {
    window.localStorage.setItem(SELECTED_WORKSPACE_KEY, JSON.stringify('workspace-a'));
    renderWorkspaceProvider('workspace-a');
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');

    window.localStorage.setItem(SELECTED_WORKSPACE_KEY, JSON.stringify('workspace-b'));
    setItemSpy.mockClear();

    act(() => {
      window.dispatchEvent(new StorageEvent('storage', { key: SELECTED_WORKSPACE_KEY }));
    });

    expect(setItemSpy).not.toHaveBeenCalled();
    expect(window.localStorage.getItem(SELECTED_WORKSPACE_KEY)).toBe(JSON.stringify('workspace-b'));
  });
});
