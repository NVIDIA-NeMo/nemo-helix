// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExternalPlugin, PreviewResponse } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { server } from '@studio/mocks/node';
import { CommunityPluginsRoute } from '@studio/routes/CommunityPluginsRoute';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { menuFor } from '@studio/tests/util/menuFor';
import { render, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@studio/hooks/useWorkspaceFromPath');

const API = `${PLATFORM_BASE_URL}/apis/plugins`;
const MANIFEST_URL = 'https://publisher.example/phoenix.yaml';
const SHA = 'a'.repeat(64);

const phoenix: ExternalPlugin = {
  name: 'phoenix',
  title: 'Phoenix Traces',
  enabled: true,
  endpoints: [
    {
      method: 'GET',
      path: '/v2/workspaces/{workspace}/projects',
      access: 'read',
      upstream_url: 'https://adapter.example/projects',
    },
  ],
};

const preview: PreviewResponse = {
  manifest_sha256: SHA,
  warning: 'Registering this plugin lets Studio run code you did not write.',
  manifest: {
    schema_version: 1,
    plugin: {
      ...phoenix,
      credential: { secret_name: 'phoenix-key', scheme: 'bearer' },
      bundle: {
        url: 'https://publisher.example/index.js',
        sha256: 'b'.repeat(64),
        sidecar: { contractVersion: '1.0.0', specifiers: [], builtAgainst: {} },
      },
      navigation: [{ label: 'Traces', path: '/workspaces/{workspace}/plugin/phoenix' }],
    },
  },
};

interface Backend {
  installed: ExternalPlugin[];
  installBodies: unknown[];
  patches: unknown[];
  removed: string[];
}

/** A tiny in-memory registry behind the management API, so the page re-reads real state. */
const mockRegistry = (
  overrides: {
    install?: () => Response;
    catalog?: { url: string; name: string; title: string }[];
  } = {}
): Backend => {
  const state: Backend = { installed: [], installBodies: [], patches: [], removed: [] };
  server.use(
    http.get(`${API}/-/installed`, () =>
      HttpResponse.json({ schema_version: 1, plugins: state.installed })
    ),
    http.get(`${API}/-/catalog`, () => HttpResponse.json({ entries: overrides.catalog ?? [] })),
    http.post(`${API}/-/preview`, () => HttpResponse.json(preview)),
    http.post(`${API}/-/installed`, async ({ request }) => {
      state.installBodies.push(await request.json());
      if (overrides.install) return overrides.install();
      state.installed.push(preview.manifest.plugin);
      return HttpResponse.json({
        name: 'phoenix',
        manifest_sha256: SHA,
        bundleUrl: '/plugin-ui/phoenix/x/index.js',
      });
    }),
    http.patch(`${API}/-/installed/:name`, async ({ request }) => {
      const body = (await request.json()) as { enabled: boolean };
      state.patches.push(body);
      state.installed = state.installed.map((p) => ({ ...p, enabled: body.enabled }));
      return HttpResponse.json({});
    }),
    http.delete(`${API}/-/installed/:name`, ({ params }) => {
      state.removed.push(String(params.name));
      state.installed = state.installed.filter((p) => p.name !== params.name);
      return new HttpResponse(null, { status: 204 });
    })
  );
  return state;
};

describe('CommunityPluginsRoute', () => {
  beforeEach(() => {
    vi.mocked(useWorkspaceFromPath).mockReturnValue('default');
  });

  it('registers a plugin only after the previewed manifest is reviewed and consented to', async () => {
    const user = userEvent.setup();
    const backend = mockRegistry();
    render(<CommunityPluginsRoute />);

    await screen.findByText('No community plugins', undefined, { timeout: XL_SELECTOR_TIMEOUT });

    await user.click(screen.getAllByRole('button', { name: 'Add Plugin' })[0]);
    const dialog = await screen.findByRole('dialog');
    const previewButton = within(dialog).getByRole('button', { name: 'Preview' });
    expect(previewButton).toBeDisabled();

    await user.type(within(dialog).getByLabelText('Manifest URL'), MANIFEST_URL);
    await user.click(previewButton);

    // Review step: everything the admin consents to is on screen.
    expect(await screen.findByText('Review: Phoenix Traces')).toBeInTheDocument();
    expect(screen.getByTestId('plugin-trust-warning')).toHaveTextContent('code you did not write');
    const endpoints = screen.getByTestId('plugin-endpoints');
    expect(endpoints).toHaveTextContent('/apis/plugins/phoenix/v2/workspaces/{workspace}/projects');
    expect(endpoints).toHaveTextContent('https://adapter.example/projects');
    expect(screen.getByText('phoenix-key')).toBeInTheDocument();
    expect(screen.getByText(/not sandboxed/)).toBeInTheDocument();
    expect(screen.getByText('b'.repeat(64))).toBeInTheDocument();

    const register = screen.getByRole('button', { name: 'Register' });
    expect(register).toBeDisabled();
    expect(backend.installBodies).toHaveLength(0);

    await user.click(screen.getByRole('checkbox'));
    await user.click(register);

    // The request carries the previewed URL and digest plus consent, and the list refreshes.
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(backend.installBodies).toEqual([
      { url: MANIFEST_URL, manifest_sha256: SHA, consent: true },
    ]);
    expect(await screen.findByText('Phoenix Traces')).toBeInTheDocument();
    expect(screen.getByText('Enabled')).toBeInTheDocument();
  });

  it('sends the admin back to the URL step when the manifest changed since the preview', async () => {
    const user = userEvent.setup();
    mockRegistry({
      install: () =>
        HttpResponse.json(
          { detail: 'Manifest changed since preview; preview it again.' },
          { status: 409 }
        ),
    });
    render(<CommunityPluginsRoute />);

    await user.click((await screen.findAllByRole('button', { name: 'Add Plugin' }))[0]);
    const dialog = await screen.findByRole('dialog');
    await user.type(within(dialog).getByLabelText('Manifest URL'), MANIFEST_URL);
    await user.click(within(dialog).getByRole('button', { name: 'Preview' }));
    await screen.findByText('Review: Phoenix Traces');
    await user.click(screen.getByRole('checkbox'));
    await user.click(screen.getByRole('button', { name: 'Register' }));

    expect(
      await screen.findByText('Manifest changed since preview; preview it again.')
    ).toBeInTheDocument();
    // Back on the URL step: consent is gone, so the stale review cannot be re-submitted.
    expect(screen.getByLabelText('Manifest URL')).toBeInTheDocument();
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('prefills the dialog from a catalog suggestion, hiding ones already registered', async () => {
    const user = userEvent.setup();
    const backend = mockRegistry({
      catalog: [
        { url: MANIFEST_URL, name: 'phoenix', title: 'Phoenix Traces' },
        { url: 'https://publisher.example/tavily.yaml', name: 'tavily', title: 'Tavily' },
      ],
    });
    backend.installed = [phoenix];
    render(<CommunityPluginsRoute />);

    // Phoenix is registered, so only Tavily is suggested.
    await screen.findByText('Suggested plugins');
    expect(screen.queryByText(MANIFEST_URL)).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Review' }));

    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByLabelText('Manifest URL')).toHaveValue(
      'https://publisher.example/tavily.yaml'
    );
  });

  it('disables, re-enables and removes a registered plugin', async () => {
    const user = userEvent.setup();
    const backend = mockRegistry();
    backend.installed = [phoenix];
    render(<CommunityPluginsRoute />);

    await screen.findByText('Phoenix Traces', undefined, { timeout: XL_SELECTOR_TIMEOUT });
    expect(screen.getByText('Enabled')).toBeInTheDocument();

    // The row remounts after each refetch, so look the trigger up again every time.
    const choose = async (item: string) => {
      const trigger = screen.getByRole('button', { name: 'Open quick actions menu' });
      await user.click(trigger);
      await user.click(within(menuFor(trigger)).getByRole('menuitem', { name: item }));
    };

    await choose('Disable');
    expect(await screen.findByText('Disabled')).toBeInTheDocument();
    expect(backend.patches).toEqual([{ enabled: false }]);

    await choose('Enable');
    expect(await screen.findByText('Enabled')).toBeInTheDocument();
    expect(backend.patches).toEqual([{ enabled: false }, { enabled: true }]);

    await choose('Remove');
    const confirm = await screen.findByRole('dialog');
    expect(confirm).toHaveTextContent('Remove: Phoenix Traces');
    await user.click(within(confirm).getByRole('button', { name: 'Delete' }));

    await screen.findByText('No community plugins');
    expect(backend.removed).toEqual(['phoenix']);
  });
});
