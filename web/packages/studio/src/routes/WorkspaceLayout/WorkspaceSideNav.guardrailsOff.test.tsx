// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PluginContext } from '@studio/plugins/PluginContext';
import type { LoadedPlugin } from '@studio/plugins/types';
import { WorkspaceSideNav } from '@studio/routes/WorkspaceLayout/WorkspaceSideNav';
import { renderRoute, screen } from '@studio/tests/util/render';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'false');
  vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'true');
  vi.stubEnv('VITE_FF_GUARDRAILS_ENABLED', 'false');
  vi.stubEnv('VITE_FF_MONITOR_ENABLED', 'true');
  vi.stubEnv('VITE_FF_TASKSETS_ENABLED', 'true');
});

const hardener: LoadedPlugin = {
  name: 'agent-hardener',
  Root: () => null,
  navItems: () => [
    {
      group: 'Safety',
      items: [
        {
          id: 'agent-hardening',
          iconName: 'shield',
          label: 'Agent Hardening',
          href: '/workspaces/test-workspace/agent-hardening',
        },
      ],
    },
  ],
};

const render = (plugins: LoadedPlugin[]) => {
  const element = (
    <PluginContext.Provider
      value={{
        plugins,
        installedNames: new Set(plugins.map((p) => p.name)),
        isLoaded: true,
        isError: false,
      }}
    >
      <WorkspaceSideNav />
    </PluginContext.Provider>
  );
  return renderRoute(element, {
    history: '/workspaces/test-workspace/dashboard',
    routes: [{ path: '/workspaces/:workspace/*', element }],
  });
};

describe('WorkspaceSideNav with Guardrails disabled', () => {
  beforeEach(() => window.localStorage.clear());

  it('places a plugin Safety item between Resources and Data Preparation', () => {
    render([hardener]);

    expect(screen.getAllByText('Safety')).toHaveLength(1);
    expect(screen.queryByRole('link', { name: 'Guardrails' })).not.toBeInTheDocument();

    const ordered = [
      screen.getByRole('link', { name: 'Datasets' }),
      screen.getByText('Safety'),
      screen.getByRole('link', { name: 'Agent Hardening' }),
      screen.getByRole('link', { name: 'Data Designer' }),
    ];
    ordered.slice(1).forEach((node, i) => {
      expect(
        ordered[i].compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING
      ).toBeTruthy();
    });
  });

  it('omits the Safety heading when nothing contributes to it', () => {
    render([]);

    expect(screen.queryByText('Safety')).not.toBeInTheDocument();
  });
});
