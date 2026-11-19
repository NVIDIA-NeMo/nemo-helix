// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  usePluginLoadFailure,
  usePluginLoadFailures,
  usePlugins,
  usePluginsLoaded,
} from '@studio/plugins/PluginContext';
import { PluginRenderer } from '@studio/plugins/PluginRenderer';
import type { LoadedPlugin, PluginLoadFailure, PluginRootProps } from '@studio/plugins/types';
import { BreadcrumbsProvider } from '@studio/providers/breadcrumbs/BreadcrumbsProvider';
import {
  useBreadcrumbs,
  type BreadcrumbsItemProps,
} from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { act, render, screen, within } from '@testing-library/react';
import { useEffect } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router';

const authState = vi.hoisted(() => ({ accessToken: 'test-token' }));

let capturedProps: PluginRootProps | undefined;
const mountSpy = vi.fn();
const mockNavItems = vi.fn(() => []);

function MockRoot(props: PluginRootProps) {
  capturedProps = props;
  useEffect(() => {
    mountSpy();
  }, []);
  return (
    <div data-testid="plugin-root">
      ws:{props.host.workspaceId} token:{props.host.auth.accessToken}
    </div>
  );
}

function makePlugin(name: string): LoadedPlugin {
  return { name, Root: MockRoot, navItems: mockNavItems };
}

vi.mock('@studio/plugins/PluginContext', () => ({
  usePlugins: vi.fn(),
  usePluginsLoaded: vi.fn(),
  usePluginLoadFailures: vi.fn(),
  usePluginLoadFailure: vi.fn(),
}));
vi.mock('@nemo/common/src/providers/toast/useToast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  }),
}));
vi.mock('react-oidc-context', () => ({
  useAuth: vi.fn(() => ({ user: { access_token: authState.accessToken } })),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: vi.fn(() => 'my-workspace'),
}));

function renderPlugin(pluginName = 'test-plugin') {
  return render(
    <BreadcrumbsProvider>
      <MemoryRouter initialEntries={[`/workspaces/ws1/plugin/${pluginName}/`]}>
        <Routes>
          <Route path="/workspaces/:workspace/plugin/:pluginName/*" element={<PluginRenderer />} />
        </Routes>
      </MemoryRouter>
    </BreadcrumbsProvider>
  );
}

beforeEach(() => {
  authState.accessToken = 'test-token';
  capturedProps = undefined;
  mountSpy.mockReset();
  vi.mocked(usePluginsLoaded).mockReturnValue(true);
  vi.mocked(usePluginLoadFailures).mockReturnValue([]);
  vi.mocked(usePluginLoadFailure).mockReturnValue(undefined);
});

const failure: PluginLoadFailure = {
  name: 'pr-review',
  issues: [
    {
      code: 'PLUGIN_SINGLETON_BUNDLED',
      subject: 'react-router',
      message: 'The singleton "react-router" is bundled into the plugin.',
      steps: ['Import react-router only by its contract specifier.', 'Rebuild the plugin.'],
    },
    {
      code: 'PLUGIN_SPECIFIER_UNMAPPED',
      subject: 'lodash',
      message: 'Studio does not provide "lodash".',
      steps: ['Bundle lodash as a normal dependency.'],
    },
  ],
};

function mockFailedPlugin() {
  vi.mocked(usePlugins).mockReturnValue([]);
  vi.mocked(usePluginLoadFailures).mockReturnValue([failure]);
  vi.mocked(usePluginLoadFailure).mockImplementation((name) =>
    name === failure.name ? failure : undefined
  );
}

describe('PluginRenderer', () => {
  it('shows loading state while plugins have not finished loading', () => {
    vi.mocked(usePluginsLoaded).mockReturnValue(false);
    vi.mocked(usePlugins).mockReturnValue([]);

    render(
      <BreadcrumbsProvider>
        <MemoryRouter initialEntries={['/workspaces/ws1/plugin/test-plugin/page1']}>
          <Routes>
            <Route
              path="/workspaces/:workspace/plugin/:pluginName/*"
              element={<PluginRenderer />}
            />
          </Routes>
        </MemoryRouter>
      </BreadcrumbsProvider>
    );

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('renders the plugin Root with workspaceId and accessToken when loaded', () => {
    vi.mocked(usePlugins).mockReturnValue([makePlugin('test-plugin')]);

    renderPlugin();

    expect(screen.getByTestId('plugin-root')).toBeInTheDocument();
    expect(capturedProps?.host.workspaceId).toBe('my-workspace');
    expect(capturedProps?.host.auth.accessToken).toBe('test-token');
    expect(capturedProps?.host.auth.getAccessToken()).toBe('test-token');
    expect(typeof capturedProps?.host.sdk.platform.useEntitiesListWorkspaces).toBe('function');
    expect(typeof capturedProps?.host.sdk.agents.agentsListAgents).toBe('function');
    expect(typeof capturedProps?.host.navigation.navigate).toBe('function');
    expect(typeof capturedProps?.host.notifications.notify).toBe('function');
    expect(typeof capturedProps?.host.telemetry.event).toBe('function');
    // A plugin calling its own service prefixes requests with this; it must be a
    // string even when unset, so `${apiBaseUrl}${path}` never yields "undefined/...".
    expect(typeof capturedProps?.host.apiBaseUrl).toBe('string');
  });

  it('does not remount on token renewal and getAccessToken returns the new token', () => {
    vi.mocked(usePlugins).mockReturnValue([makePlugin('test-plugin')]);

    const { rerender } = renderPlugin();
    expect(mountSpy).toHaveBeenCalledTimes(1);

    authState.accessToken = 'renewed-token';
    rerender(
      <BreadcrumbsProvider>
        <MemoryRouter initialEntries={['/workspaces/ws1/plugin/test-plugin/']}>
          <Routes>
            <Route
              path="/workspaces/:workspace/plugin/:pluginName/*"
              element={<PluginRenderer />}
            />
          </Routes>
        </MemoryRouter>
      </BreadcrumbsProvider>
    );

    expect(mountSpy).toHaveBeenCalledTimes(1);
    expect(capturedProps?.host.auth.getAccessToken()).toBe('renewed-token');
  });

  it('shows not found when plugin name does not match any loaded plugin', () => {
    vi.mocked(usePlugins).mockReturnValue([makePlugin('other-plugin')]);

    renderPlugin('missing-plugin');

    expect(screen.getByText(/not found/i)).toBeInTheDocument();
  });

  it('contains a plugin render error in the plugin panel instead of unwinding', () => {
    function ThrowingRoot(): never {
      throw new Error('boom from plugin');
    }
    vi.mocked(usePlugins).mockReturnValue([
      { name: 'test-plugin', Root: ThrowingRoot, navItems: mockNavItems },
    ]);
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    renderPlugin();

    expect(screen.getByText(/this plugin ran into a problem/i)).toBeInTheDocument();
    expect(screen.getByText(/boom from plugin/)).toBeInTheDocument();
    expect(screen.queryByTestId('plugin-root')).not.toBeInTheDocument();

    consoleError.mockRestore();
  });

  it('writes a plugin breadcrumb trail into Studio and clears it on unmount', () => {
    const seen: BreadcrumbsItemProps[][] = [];
    function BreadcrumbSpy() {
      seen.push(useBreadcrumbs().breadcrumbs);
      return null;
    }
    vi.mocked(usePlugins).mockReturnValue([makePlugin('test-plugin')]);

    // The provider and spy outlive the plugin subtree, so the cleared trail is
    // still observable after the plugin itself unmounts.
    const tree = (mounted: boolean) => (
      <BreadcrumbsProvider>
        <BreadcrumbSpy />
        {mounted ? (
          <MemoryRouter initialEntries={['/workspaces/ws1/plugin/test-plugin/']}>
            <Routes>
              <Route
                path="/workspaces/:workspace/plugin/:pluginName/*"
                element={<PluginRenderer />}
              />
            </Routes>
          </MemoryRouter>
        ) : null}
      </BreadcrumbsProvider>
    );
    const { rerender } = render(tree(true));

    act(() =>
      capturedProps?.host.breadcrumbs.set([
        { label: 'Runs', href: '/workspaces/ws1/plugin/test-plugin/runs' },
        { label: 'Run 7' },
      ])
    );
    expect(seen.at(-1)).toEqual([
      { slotLabel: 'Runs', href: '/workspaces/ws1/plugin/test-plugin/runs' },
      { slotLabel: 'Run 7', href: undefined },
    ]);

    rerender(tree(false));
    // Studio clears the trail itself; a plugin that forgets cannot leave one behind.
    expect(seen.at(-1)).toEqual([]);
  });

  it('clears the trail when the router swaps one plugin for another', () => {
    const seen: BreadcrumbsItemProps[][] = [];
    function BreadcrumbSpy() {
      seen.push(useBreadcrumbs().breadcrumbs);
      return null;
    }
    vi.mocked(usePlugins).mockReturnValue([makePlugin('plugin-a'), makePlugin('plugin-b')]);

    render(
      <BreadcrumbsProvider>
        <BreadcrumbSpy />
        <MemoryRouter initialEntries={['/workspaces/ws1/plugin/plugin-a/']}>
          <Routes>
            <Route
              path="/workspaces/:workspace/plugin/:pluginName/*"
              element={<PluginRenderer />}
            />
          </Routes>
        </MemoryRouter>
      </BreadcrumbsProvider>
    );

    act(() => capturedProps?.host.breadcrumbs.set([{ label: 'From plugin A' }]));
    expect(seen.at(-1)).toEqual([{ slotLabel: 'From plugin A', href: undefined }]);

    // Navigate for real: MemoryRouter only reads initialEntries once, and the
    // point is that PluginRenderer stays mounted while :pluginName changes, so
    // only the effect's deps can force the reset.
    act(() => capturedProps?.host.navigation.navigate('/workspaces/ws1/plugin/plugin-b/'));
    expect(seen.at(-1)).toEqual([]);
  });
  describe('when the plugin failed to load', () => {
    it('shows a failure panel naming the plugin, not "not found"', () => {
      mockFailedPlugin();

      renderPlugin('pr-review');

      expect(screen.getAllByText(/pr-review/).length).toBeGreaterThan(0);
      expect(screen.queryByText(/not found/i)).not.toBeInTheDocument();
      expect(screen.queryByTestId('plugin-root')).not.toBeInTheDocument();
    });

    it('lists each issue code and message with its steps as a numbered list', () => {
      mockFailedPlugin();

      renderPlugin('pr-review');

      for (const issue of failure.issues) {
        expect(screen.getByText(new RegExp(issue.code))).toBeInTheDocument();
        expect(screen.getByText(issue.message)).toBeInTheDocument();
      }
      const lists = screen.getAllByRole('list');
      expect(lists).toHaveLength(failure.issues.length);
      lists.forEach((list, i) => {
        expect(list).toBeInstanceOf(HTMLOListElement);
        expect(
          within(list)
            .getAllByRole('listitem')
            .map((li) => li.textContent)
        ).toEqual(failure.issues[i].steps);
      });
    });

    it('still shows not found for a plugin that is neither loaded nor failed', () => {
      mockFailedPlugin();

      renderPlugin('unknown-plugin');

      expect(screen.getByText(/not found/i)).toBeInTheDocument();
      expect(screen.queryByText(/PLUGIN_SINGLETON_BUNDLED/)).not.toBeInTheDocument();
    });
  });
});
