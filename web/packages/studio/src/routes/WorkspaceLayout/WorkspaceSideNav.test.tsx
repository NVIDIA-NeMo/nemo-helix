// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PluginContext } from '@studio/plugins/PluginContext';
import type { LoadedPlugin, PluginNavGroup } from '@studio/plugins/types';
import { WorkspaceSideNav } from '@studio/routes/WorkspaceLayout/WorkspaceSideNav';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import type { ReactElement } from 'react';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'false');
  vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'true');
  vi.stubEnv('VITE_FF_GUARDRAILS_ENABLED', 'true');
  vi.stubEnv('VITE_FF_MONITOR_ENABLED', 'true');
  vi.stubEnv('VITE_FF_TASKSETS_ENABLED', 'true');
});

const renderSideNav = (history = '/workspaces/test-workspace/dashboard') =>
  renderRoute(<WorkspaceSideNav />, {
    history,
    routes: [
      {
        path: '/workspaces/:workspace/*',
        element: <WorkspaceSideNav />,
      },
    ],
  });

const makePlugin = (name: string, groups: PluginNavGroup[]): LoadedPlugin => ({
  name,
  Root: () => null,
  navItems: () => groups,
});

const renderWithPlugins = (plugins: LoadedPlugin[]) => {
  const element: ReactElement = (
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

const findPluginLinkWithIcon = async (name: string) => {
  const link = screen.getByRole('link', { name });

  await waitFor(() => expect(link.closest('li')?.querySelector('svg')).not.toBeNull());
  return link;
};

describe('WorkspaceSideNav', () => {
  beforeEach(() => window.localStorage.clear());

  it('links to the traces view by default', () => {
    renderSideNav();

    expect(screen.getByRole('link', { name: 'Traces' })).toHaveAttribute(
      'href',
      '/workspaces/test-workspace/intake/traces'
    );
    expect(screen.queryByRole('link', { name: /annotations?/i })).not.toBeInTheDocument();
  });

  it('omits Optimizer navigation when Optimizer is disabled', () => {
    renderSideNav();

    expect(screen.queryByRole('link', { name: 'Insights' })).not.toBeInTheDocument();
  });

  it('lays the nav out as flat, task-named sections in order', () => {
    renderSideNav();

    expect(
      screen.queryByRole('button', { name: /^(expand|collapse) (agents|models)$/i })
    ).toBeNull();
    for (const old of ['Observability', 'Components', 'Governance', 'System', 'Evaluations']) {
      expect(screen.queryByText(old)).not.toBeInTheDocument();
    }

    const sections: [string | null, string[]][] = [
      [null, ['Dashboard', 'Playground']],
      ['Agent Development', ['Agents', 'Monitor', 'Experiments', 'Tasksets', 'Tasks', 'Traces']],
      [
        'Models',
        ['Model Catalog', 'Model Evaluations', 'Fine-tuning', 'Deployments', 'Virtual Models'],
      ],
      ['Resources', ['Datasets']],
      ['Safety', ['Guardrails']],
      ['Data Preparation', ['Data Designer', 'Safe Synthesizer', 'Anonymizer']],
      ['Workspace', ['Inference Providers', 'Filesets', 'Jobs', 'Settings']],
    ];
    const nodes = sections.flatMap(([heading, rows]) => [
      ...(heading ? [screen.getByText(heading)] : []),
      ...rows.map((row) => screen.getByRole('link', { name: row })),
    ]);
    nodes.slice(1).forEach((node, i) => {
      expect(
        nodes[i].compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING
      ).toBeTruthy();
    });
    // No row is a disclosure parent any more (section headings are not rows).
    expect(screen.queryByRole('button', { name: 'Agents' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Datasets' })).toBeNull();
  });

  it('links the renamed and new rows to their pages', () => {
    renderSideNav();

    const hrefs: Record<string, string> = {
      Agents: '/workspaces/test-workspace/agents',
      Monitor: '/workspaces/test-workspace/agents/monitor',
      'Model Evaluations': '/workspaces/test-workspace/evaluation/results',
      Datasets: '/workspaces/test-workspace/datasets',
      'Inference Providers': '/workspaces/test-workspace/inference-providers',
      'Model Catalog': '/workspaces/test-workspace/base-models',
      'Fine-tuning': '/workspaces/test-workspace/fine-tune',
    };
    for (const [name, href] of Object.entries(hrefs)) {
      expect(screen.getByRole('link', { name })).toHaveAttribute('href', href);
    }
    expect(screen.queryByRole('link', { name: 'Evaluations' })).not.toBeInTheDocument();
    expect(screen.queryByText('Custom Models')).not.toBeInTheDocument();
  });

  it('exposes the same links on the collapsed rail', () => {
    renderRoute(<WorkspaceSideNav collapsed />, {
      history: '/workspaces/test-workspace/dashboard',
      routes: [{ path: '/workspaces/:workspace/*', element: <WorkspaceSideNav collapsed /> }],
    });

    for (const name of ['Playground', 'Model Evaluations', 'Datasets', 'Inference Providers']) {
      expect(screen.getByRole('link', { name })).toBeInTheDocument();
    }
  });

  it('folds a plugin group into the core group of the same name', async () => {
    renderWithPlugins([
      makePlugin('red-team', [
        {
          group: 'Safety',
          items: [
            {
              id: 'red-team',
              iconName: 'shield',
              label: 'Red Team',
              href: '/workspaces/test-workspace/red-team',
            },
          ],
        },
      ]),
    ]);

    expect(screen.getAllByText('Safety')).toHaveLength(1);
    expect(await findPluginLinkWithIcon('Red Team')).toBeInTheDocument();
  });

  it('appends a plugin group that matches no core group', async () => {
    renderWithPlugins([
      makePlugin('red-team', [
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
      ]),
    ]);

    expect(screen.getAllByText('Red Team')).toHaveLength(1);
    expect(await findPluginLinkWithIcon('Probes')).toBeInTheDocument();
  });

  it('keeps a plugin item whose id matches a core item in the merged group', async () => {
    const duplicateKeyWarning = vi.spyOn(console, 'error').mockImplementation(() => {});

    renderWithPlugins([
      makePlugin('red-team', [
        {
          group: 'Safety',
          items: [
            {
              id: 'guardrails',
              iconName: 'shield',
              label: 'Red Team Models',
              href: '/workspaces/test-workspace/red-team-models',
            },
          ],
        },
      ]),
    ]);

    expect(screen.getByRole('link', { name: 'Guardrails' })).toBeInTheDocument();
    expect(await findPluginLinkWithIcon('Red Team Models')).toBeInTheDocument();
    const keyWarnings = duplicateKeyWarning.mock.calls
      .map((args) => args.map(String).join(' '))
      .filter((message) => message.includes('same key'));
    expect(keyWarnings).toEqual([]);

    duplicateKeyWarning.mockRestore();
  });

  it('merges groups of the same name across two plugins', async () => {
    renderWithPlugins([
      makePlugin('one', [
        {
          group: 'Safety',
          items: [
            { id: 'one', iconName: 'shield', label: 'One', href: '/workspaces/test-workspace/one' },
          ],
        },
      ]),
      makePlugin('two', [
        {
          group: 'Safety',
          items: [
            { id: 'two', iconName: 'shield', label: 'Two', href: '/workspaces/test-workspace/two' },
          ],
        },
      ]),
    ]);

    expect(screen.getAllByText('Safety')).toHaveLength(1);
    expect(await findPluginLinkWithIcon('One')).toBeInTheDocument();
    expect(await findPluginLinkWithIcon('Two')).toBeInTheDocument();
  });
});
