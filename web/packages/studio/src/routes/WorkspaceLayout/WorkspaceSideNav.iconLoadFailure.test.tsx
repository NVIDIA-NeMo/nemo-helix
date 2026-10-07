// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { PluginContext } from '@studio/plugins/PluginContext';
import type { LoadedPlugin } from '@studio/plugins/types';
import { WorkspaceSideNav } from '@studio/routes/WorkspaceLayout/WorkspaceSideNav';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';

vi.mock('@studio/plugins/PluginNavIcon', () => {
  throw new Error('chunk failed to load');
});

const plugin: LoadedPlugin = {
  name: 'red-team',
  Root: () => null,
  navItems: () => [
    {
      group: 'Red Team',
      items: [
        {
          id: 'probes',
          iconName: 'shield',
          label: 'Probes',
          href: '/workspaces/test-workspace/probes',
        },
      ],
    },
  ],
};

describe('WorkspaceSideNav when the plugin icon chunk fails to load', () => {
  it('keeps the nav and renders the plugin item without an icon', async () => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    const element = (
      <PluginContext.Provider
        value={{
          plugins: [plugin],
          installedNames: new Set(['red-team']),
          isLoaded: true,
          isError: false,
        }}
      >
        <WorkspaceSideNav />
      </PluginContext.Provider>
    );

    renderRoute(element, {
      history: '/workspaces/test-workspace/dashboard',
      routes: [{ path: '/workspaces/:workspace/*', element }],
    });

    await waitFor(() =>
      expect(warn).toHaveBeenCalledWith(
        '[plugins] Failed to load plugin nav icons:',
        expect.anything()
      )
    );
    const link = screen.getByRole('link', { name: 'Probes' });
    expect(link).toBeInTheDocument();
    expect(link.closest('li')?.querySelector('svg')).toBeNull();

    warn.mockRestore();
  });
});
