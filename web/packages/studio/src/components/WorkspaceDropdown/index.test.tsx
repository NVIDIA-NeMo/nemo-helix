// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { WorkspaceDropdown } from '@studio/components/WorkspaceDropdown';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import { WORKSPACE_DROPDOWN_RECENT_KEY } from '@studio/util/localStorage';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const NEW_WORKSPACE = 'new-workspace';

const mockWorkspacesList = (names: string[]) => {
  const onRequest = vi.fn();
  server.use(
    http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces`, () => {
      onRequest();
      return HttpResponse.json({
        object: 'list',
        data: names.map((name) => ({
          id: `id-${name}`,
          name,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        })),
        pagination: {
          page: 1,
          page_size: 1000,
          current_page_size: names.length,
          total_pages: 1,
          total_results: names.length,
        },
      });
    })
  );
  return onRequest;
};

const renderDropdown = () =>
  renderRoute(undefined, {
    history: `/workspaces/${DEFAULT_WORKSPACE}`,
    routes: [{ path: '/workspaces/:workspace/*', element: <WorkspaceDropdown /> }],
  });

describe('WorkspaceDropdown', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('refetches workspaces when the dropdown is opened', async () => {
    const user = userEvent.setup();
    const initialListRequest = mockWorkspacesList([DEFAULT_WORKSPACE]);
    renderDropdown();
    await waitFor(() => expect(initialListRequest).toHaveBeenCalled());

    // Created outside the dropdown, e.g. from the CLI or another tab
    mockWorkspacesList([NEW_WORKSPACE, DEFAULT_WORKSPACE]);
    await user.click(await screen.findByRole('button', { name: 'Select workspace' }));

    expect(await screen.findByRole('menuitem', { name: NEW_WORKSPACE })).toBeInTheDocument();
  });

  it('adds a newly created workspace to recent workspaces', async () => {
    const user = userEvent.setup();
    mockWorkspacesList([DEFAULT_WORKSPACE]);
    renderDropdown();

    const trigger = await screen.findByRole('button', { name: 'Select workspace' });
    await user.click(trigger);
    await user.click(await screen.findByRole('menuitem', { name: 'New Workspace' }));

    const dialog = await screen.findByRole('dialog');
    await user.type(within(dialog).getByRole('textbox', { name: 'Name' }), NEW_WORKSPACE);
    mockWorkspacesList([NEW_WORKSPACE, DEFAULT_WORKSPACE]);
    await user.click(within(dialog).getByRole('button', { name: 'Create' }));

    expect(await screen.findByRole('button', { name: 'Select workspace' })).toHaveTextContent(
      NEW_WORKSPACE
    );
    await user.click(screen.getByRole('button', { name: 'Select workspace' }));

    // Listed under both "Recent Workspaces" and "Workspaces"
    expect(await screen.findAllByRole('menuitem', { name: NEW_WORKSPACE })).toHaveLength(2);
    expect(JSON.parse(window.localStorage.getItem(WORKSPACE_DROPDOWN_RECENT_KEY) ?? '[]')).toEqual([
      NEW_WORKSPACE,
      DEFAULT_WORKSPACE,
    ]);
  });
});
