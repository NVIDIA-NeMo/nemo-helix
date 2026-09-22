// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PluginContext } from '@studio/plugins/PluginContext';
import type { LoadedPlugin, PluginNavGroup } from '@studio/plugins/types';
import { WorkspaceSideNav } from '@studio/routes/WorkspaceLayout/WorkspaceSideNav';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import type { ReactElement } from 'react';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'false');
  vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'true');
  vi.stubEnv('VITE_FF_GUARDRAILS_ENABLED', 'true');
  vi.stubEnv('VITE_FF_MONITOR_ENABLED', 'true');
});

/** A parent row's label is a link; its chevron is a separate disclosure button. */
const disclosure = (label: string) => screen.getByRole('button', { name: new RegExp(label, 'i') });

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

  it('renders the six RFC sections', () => {
    renderSideNav();

    for (const section of ['Observability', 'Components', 'Data', 'Governance', 'System']) {
      expect(screen.getByText(section)).toBeInTheDocument();
    }
    // Superseded group headings
    expect(screen.queryByText('Safety')).not.toBeInTheDocument();
    expect(screen.queryByText('Evaluate')).not.toBeInTheDocument();
  });

  it('links Agents to its entity list page and exposes Model Catalog under Models', () => {
    renderSideNav('/workspaces/test-workspace/base-models');

    expect(screen.getByRole('link', { name: 'Agents' })).toHaveAttribute(
      'href',
      '/workspaces/test-workspace/agents'
    );
    // Models is a pure disclosure, not a link to a page of its own.
    expect(screen.queryByRole('link', { name: 'Models' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Model Catalog' })).toHaveAttribute(
      'href',
      '/workspaces/test-workspace/base-models'
    );
  });

  it('expands and collapses a parent from the chevron alone', async () => {
    const user = userEvent.setup();
    renderSideNav('/workspaces/test-workspace/agents/monitor');

    expect(disclosure('Agents')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('link', { name: 'Monitor' })).toHaveAttribute(
      'href',
      '/workspaces/test-workspace/agents/monitor'
    );

    await user.click(disclosure('Agents'));
    expect(disclosure('Agents')).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('link', { name: 'Monitor' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Agents' })).toBeInTheDocument();
  });

  it('renders the customization screen as Fine-tune under Models', () => {
    renderSideNav('/workspaces/test-workspace/fine-tune');

    expect(screen.getByRole('link', { name: 'Fine-tune' })).toHaveAttribute(
      'href',
      '/workspaces/test-workspace/fine-tune'
    );
    expect(screen.queryByText('Custom Models')).not.toBeInTheDocument();
  });

  it('expands only the parent owning the current nested route', () => {
    renderSideNav('/workspaces/test-workspace/agents/monitor');

    expect(disclosure('Agents')).toHaveAttribute('aria-expanded', 'true');
    expect(disclosure('Models')).toHaveAttribute('aria-expanded', 'false');
  });

  it('keeps a manually opened Datasets open after the route moves elsewhere', async () => {
    const user = userEvent.setup();
    renderSideNav();

    // Datasets has no landing page of its own, so its whole row is the disclosure control and a
    // plain click on the label opens it.
    expect(disclosure('Datasets')).toHaveAttribute('aria-expanded', 'false');
    await user.click(disclosure('Datasets'));
    expect(disclosure('Datasets')).toHaveAttribute('aria-expanded', 'true');

    // The manual open is the user's preference now; navigating elsewhere does not undo it.
    await user.click(screen.getByRole('link', { name: 'Agents' }));
    expect(disclosure('Agents')).toHaveAttribute('aria-expanded', 'true');
    expect(disclosure('Datasets')).toHaveAttribute('aria-expanded', 'true');
  });

  it('keeps a chevron-collapsed parent collapsed when the route comes back to it', async () => {
    const user = userEvent.setup();
    renderSideNav('/workspaces/test-workspace/agents');

    await user.click(disclosure('Agents'));
    expect(disclosure('Agents')).toHaveAttribute('aria-expanded', 'false');

    await user.click(disclosure('Models'));
    await user.click(screen.getByRole('link', { name: 'Model Catalog' }));
    await user.click(screen.getByRole('link', { name: 'Agents' }));
    // The collapse is a saved preference; returning to the section must not reopen it.
    expect(disclosure('Agents')).toHaveAttribute('aria-expanded', 'false');
  });

  it('folds a plugin group into the core group of the same name', () => {
    renderWithPlugins([
      makePlugin('red-team', [
        {
          group: 'Governance',
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

    expect(screen.getAllByText('Governance')).toHaveLength(1);
    expect(screen.getByRole('link', { name: 'Red Team' })).toBeInTheDocument();
  });

  it('appends a plugin group that matches no core group', () => {
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
    expect(screen.getByRole('link', { name: 'Probes' })).toBeInTheDocument();
  });

  it('keeps a plugin item whose id matches a core item in the merged group', () => {
    const duplicateKeyWarning = vi.spyOn(console, 'error').mockImplementation(() => {});

    renderWithPlugins([
      makePlugin('red-team', [
        {
          group: 'Governance',
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
    expect(screen.getByRole('link', { name: 'Red Team Models' })).toBeInTheDocument();
    const keyWarnings = duplicateKeyWarning.mock.calls
      .map((args) => args.map(String).join(' '))
      .filter((message) => message.includes('same key'));
    expect(keyWarnings).toEqual([]);

    duplicateKeyWarning.mockRestore();
  });

  it('merges groups of the same name across two plugins', () => {
    renderWithPlugins([
      makePlugin('one', [
        {
          group: 'Governance',
          items: [
            { id: 'one', iconName: 'shield', label: 'One', href: '/workspaces/test-workspace/one' },
          ],
        },
      ]),
      makePlugin('two', [
        {
          group: 'Governance',
          items: [
            { id: 'two', iconName: 'shield', label: 'Two', href: '/workspaces/test-workspace/two' },
          ],
        },
      ]),
    ]);

    expect(screen.getAllByText('Governance')).toHaveLength(1);
    expect(screen.getByRole('link', { name: 'One' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Two' })).toBeInTheDocument();
  });
});
