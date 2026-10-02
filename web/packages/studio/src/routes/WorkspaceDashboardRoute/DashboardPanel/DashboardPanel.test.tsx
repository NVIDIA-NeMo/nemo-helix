// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DashboardPanel } from '@studio/routes/WorkspaceDashboardRoute/DashboardPanel';
import { render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';
import { MemoryRouter } from 'react-router';

const renderPanel = (props: Partial<ComponentProps<typeof DashboardPanel>> = {}) =>
  render(
    <MemoryRouter>
      <DashboardPanel
        icon={<span>icon</span>}
        title="Connect an Agent"
        description="Register an agent you already run."
        actions={[
          { icon: <span>upload-icon</span>, label: 'Upload an Agent', href: '/upload' },
          { icon: <span>optimize-icon</span>, label: 'Optimize', href: '/optimize' },
        ]}
        {...props}
      />
    </MemoryRouter>
  );

describe('DashboardPanel', () => {
  it('renders title and description', () => {
    renderPanel();
    expect(screen.getByText('Connect an Agent')).toBeInTheDocument();
    expect(screen.getByText('Register an agent you already run.')).toBeInTheDocument();
  });

  it('renders a link for every action pointing at its href', () => {
    renderPanel();
    expect(screen.getByRole('link', { name: /Upload an Agent/ })).toHaveAttribute(
      'href',
      '/upload'
    );
    expect(screen.getByRole('link', { name: /Optimize/ })).toHaveAttribute('href', '/optimize');
  });

  it('renders no action links when actions is empty', () => {
    renderPanel({ actions: [] });
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('renders internal actions as in-app links without target="_blank"', () => {
    renderPanel();
    expect(screen.getByRole('link', { name: /Optimize/ })).not.toHaveAttribute('target');
  });

  it('renders external actions as new-tab links with rel="noopener noreferrer"', () => {
    renderPanel({
      actions: [
        { icon: <span>icon</span>, label: 'Docs', href: 'https://docs.example.com/agents' },
      ],
    });
    const link = screen.getByRole('link', { name: /Docs/ });
    expect(link).toHaveAttribute('href', 'https://docs.example.com/agents');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('renders distinct links for actions that share a label but have different hrefs', () => {
    renderPanel({
      actions: [
        { icon: <span>icon</span>, label: 'View', href: '/one' },
        { icon: <span>icon</span>, label: 'View', href: '/two' },
      ],
    });
    const links = screen.getAllByRole('link', { name: /View/ });
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveAttribute('href', '/one');
    expect(links[1]).toHaveAttribute('href', '/two');
  });

  it('renders distinct links for actions that share an href but have different labels', () => {
    renderPanel({
      actions: [
        { icon: <span>icon</span>, label: 'Primary', href: '/same' },
        { icon: <span>icon</span>, label: 'Secondary', href: '/same' },
      ],
    });
    expect(screen.getByRole('link', { name: /Primary/ })).toHaveAttribute('href', '/same');
    expect(screen.getByRole('link', { name: /Secondary/ })).toHaveAttribute('href', '/same');
  });

  it.each(['mailto:support@nvidia.com', 'tel:+15555551234'])(
    'treats a scheme-only href (%s) as external and opens it in a new tab',
    (href) => {
      renderPanel({ actions: [{ icon: <span>icon</span>, label: 'Contact', href }] });
      const link = screen.getByRole('link', { name: /Contact/ });
      expect(link).toHaveAttribute('href', href);
      expect(link).toHaveAttribute('target', '_blank');
      expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    }
  );
});
