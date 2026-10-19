// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { getQuickstartDismissedKey } from '@studio/routes/WorkspaceDashboardHomeRoute/quickstartDismissedStorage';
import { QuickstartSection } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSection';
import { renderRoute } from '@studio/tests/util/render';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { ComponentProps } from 'react';

const renderQuickstartSection = (props: Partial<ComponentProps<typeof QuickstartSection>> = {}) =>
  renderRoute(<QuickstartSection workspace="my-workspace" {...props} />);

describe('QuickstartSection', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('renders the heading and all three panels', () => {
    renderQuickstartSection();
    expect(screen.getByText('Quickstart')).toBeInTheDocument();
    expect(screen.getByText('Connect an Agent')).toBeInTheDocument();
    expect(screen.getByText('Observability')).toBeInTheDocument();
    expect(screen.getByText('Customize Models')).toBeInTheDocument();
  });

  it('dismisses and persists the dismissal to localStorage, scoped to the workspace', () => {
    renderQuickstartSection();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss Quickstart' }));

    expect(screen.queryByText('Quickstart')).not.toBeInTheDocument();
    expect(window.localStorage.getItem(getQuickstartDismissedKey('my-workspace'))).toBe('true');
  });

  it('calls onDismiss before the section unmounts, so a caller can move focus first', () => {
    const onDismiss = vi.fn();
    renderQuickstartSection({ onDismiss });

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss Quickstart' }));

    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it('does not render when already dismissed for this workspace', () => {
    window.localStorage.setItem(getQuickstartDismissedKey('my-workspace'), 'true');
    renderQuickstartSection();

    expect(screen.queryByText('Quickstart')).not.toBeInTheDocument();
  });

  it('dismissing one workspace does not dismiss Quickstart for another workspace', () => {
    window.localStorage.setItem(getQuickstartDismissedKey('other-workspace'), 'true');
    renderQuickstartSection();

    expect(screen.getByText('Quickstart')).toBeInTheDocument();
  });

  it('hides a panel whose backing feature flag is disabled', () => {
    renderQuickstartSection({ agentsEnabled: false });

    expect(screen.queryByText('Connect an Agent')).not.toBeInTheDocument();
    expect(screen.getByText('Observability')).toBeInTheDocument();
  });

  it('hides a single action whose backing feature flag is disabled, keeping the panel', () => {
    renderQuickstartSection({ optimizerEnabled: false });

    expect(screen.getByText('Connect an Agent')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Optimize' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Upload an Agent' })).toBeInTheDocument();
  });

  it('renders nothing when every panel flag is disabled', () => {
    renderQuickstartSection({
      agentsEnabled: false,
      intakeEnabled: false,
      customizerEnabled: false,
    });

    // TestProviders adds wrapper elements, so the container is never empty.
    expect(screen.queryByText('Quickstart')).not.toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('sizes panels to fill the row width regardless of how many are visible', () => {
    renderQuickstartSection();

    const grid = screen.getByTestId('nv-grid');
    expect(grid.style.getPropertyValue('--nv-grid-template-columns')).toBe(
      'repeat(auto-fit, minmax(320px, 1fr))'
    );
  });

  it('opens the sample workspace modal from the header', async () => {
    renderQuickstartSection();

    fireEvent.click(await screen.findByRole('button', { name: 'Try a Sample Agent' }));

    expect(await screen.findByText('Create Sample Workspace')).toBeInTheDocument();
  });

  // Each case renders an eligible section beside the gated one: the eligible button appearing
  // proves the shared workspace list resolved, so the gated section had its chance to render one.
  it('hides the sample button when agents are disabled', async () => {
    renderRoute(
      <>
        <QuickstartSection workspace="my-workspace" />
        <QuickstartSection workspace="other-workspace" agentsEnabled={false} />
      </>
    );

    expect(await screen.findAllByRole('button', { name: 'Try a Sample Agent' })).toHaveLength(1);
  });

  it('hides the sample button inside a sample workspace', async () => {
    renderRoute(
      <>
        <QuickstartSection workspace="my-workspace" />
        <QuickstartSection workspace="sample-1a2b3c4d" />
      </>
    );

    expect(await screen.findAllByRole('button', { name: 'Try a Sample Agent' })).toHaveLength(1);
  });

  it('hides the sample button when a sample workspace is visible', async () => {
    const listWorkspaces = vi.fn(() =>
      HttpResponse.json({
        object: 'list',
        data: [{ name: 'my-workspace' }, { name: 'sample-1a2b3c4d' }],
        pagination: { page: 1, page_size: 1000, total_pages: 1, total_results: 2 },
      })
    );
    server.use(http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces`, listWorkspaces));

    renderQuickstartSection();

    await waitFor(() => expect(listWorkspaces).toHaveBeenCalled());
    expect(await screen.findByText('Connect an Agent')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Try a Sample Agent' })).not.toBeInTheDocument();
  });
});
