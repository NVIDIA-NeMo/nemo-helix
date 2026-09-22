// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getQuickstartDismissedKey } from '@studio/routes/WorkspaceDashboardHomeRoute/quickstartDismissedStorage';
import { QuickstartSection } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSection';
import { fireEvent, render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';
import { MemoryRouter } from 'react-router';

const renderQuickstartSection = (props: Partial<ComponentProps<typeof QuickstartSection>> = {}) =>
  render(
    <MemoryRouter>
      <QuickstartSection workspace="my-workspace" {...props} />
    </MemoryRouter>
  );

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
    const { container } = renderQuickstartSection({
      agentsEnabled: false,
      intakeEnabled: false,
      customizerEnabled: false,
    });

    expect(container).toBeEmptyDOMElement();
  });

  it('sizes panels to fill the row width regardless of how many are visible', () => {
    renderQuickstartSection();

    const grid = screen.getByTestId('nv-grid');
    expect(grid.style.getPropertyValue('--nv-grid-template-columns')).toBe(
      'repeat(auto-fit, minmax(320px, 1fr))'
    );
  });
});
