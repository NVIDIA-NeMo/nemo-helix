// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AssistantStudioLink } from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLink';
import {
  getAssistantDocsLinkTarget,
  getStudioInternalLinkTarget,
} from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLinkTarget';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

describe('AssistantStudioLink', () => {
  it('accepts same-origin Studio workspace paths', () => {
    expect(getStudioInternalLinkTarget('/workspaces/default/agents', 'https://studio.test')).toBe(
      '/workspaces/default/agents'
    );
    expect(
      getStudioInternalLinkTarget(
        'https://studio.test/workspaces/default/agents?status=ready#agent',
        'https://studio.test'
      )
    ).toBe('/workspaces/default/agents?status=ready#agent');
  });

  it('rejects external links', () => {
    expect(
      getStudioInternalLinkTarget('https://example.com/settings', 'https://studio.test')
    ).toBeUndefined();
  });

  it('internalizes absolute Studio links from other local origins', () => {
    expect(
      getStudioInternalLinkTarget(
        'http://localhost:8080/workspaces/danielleali/customizations',
        'http://ns.local.aire.nvidia.com:5173',
        'default'
      )
    ).toBe('/workspaces/default/customizations');
  });

  it('maps legacy Studio entry URLs to the current workspace dashboard', () => {
    expect(
      getStudioInternalLinkTarget(
        'http://localhost:8080/studio',
        'http://ns.local.aire.nvidia.com:5173',
        'default'
      )
    ).toBe('/workspaces/default/dashboard');
  });

  it('strips legacy Studio prefixes and rewrites stale workspaces', () => {
    expect(
      getStudioInternalLinkTarget(
        'http://localhost:8080/studio/workspaces/danielleali/agents/spanish-translator?tab=chat-playground',
        'http://ns.local.aire.nvidia.com:5173',
        'default'
      )
    ).toBe('/workspaces/default/agents/spanish-translator?tab=chat-playground');
  });

  it('canonicalizes generated evaluation results links to the registered route', () => {
    expect(
      getStudioInternalLinkTarget(
        '/workspaces/default/dashboard/evaluations/results',
        'https://studio.test'
      )
    ).toBe('/workspaces/default/evaluation/results');
    expect(
      getStudioInternalLinkTarget(
        'http://localhost:8080/workspaces/danielleali/dashboard/evaluation/results?status=complete',
        'http://ns.local.aire.nvidia.com:5173',
        'default'
      )
    ).toBe('/workspaces/default/evaluation/results?status=complete');
    expect(
      getStudioInternalLinkTarget(
        '/workspaces/default/evaluations/results#latest',
        'https://studio.test'
      )
    ).toBe('/workspaces/default/evaluation/results#latest');
  });

  it('renders accepted Studio paths as router links', () => {
    render(
      <MemoryRouter>
        <AssistantStudioLink href="/workspaces/default/agents">Agents</AssistantStudioLink>
      </MemoryRouter>
    );

    const link = screen.getByRole('link', { name: 'Agents' });

    expect(link).toHaveAttribute('href', '/workspaces/default/agents');
    expect(link).toHaveClass('inline-flex', 'rounded');
    expect(link.className).toContain('bg-[linear-gradient');
  });

  it('renders rejected paths as inert text', () => {
    render(
      <MemoryRouter>
        <AssistantStudioLink href="https://example.com">External</AssistantStudioLink>
      </MemoryRouter>
    );

    expect(screen.queryByRole('link', { name: 'External' })).not.toBeInTheDocument();
    expect(screen.getByText('External')).toBeInTheDocument();
  });

  it('renders a docs-site link as a small external link that opens in a new tab', () => {
    render(
      <MemoryRouter>
        <AssistantStudioLink href="https://docs.nvidia.com/nemo-helix">
          NeMo docs
        </AssistantStudioLink>
      </MemoryRouter>
    );

    const link = screen.getByRole('link', { name: 'NeMo docs' });

    expect(link).toHaveAttribute('href', 'https://docs.nvidia.com/nemo-helix');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link.getAttribute('rel')).toContain('noopener');
    // Subtle inline link, not the large Studio page button.
    expect(link.className).not.toContain('bg-[linear-gradient');
  });

  it('keeps links to other external sites inert even when they look like docs links', () => {
    render(
      <MemoryRouter>
        <AssistantStudioLink href="https://docs.nvidia.com.evil.example/nemo-helix">
          Fake docs
        </AssistantStudioLink>
      </MemoryRouter>
    );

    expect(screen.queryByRole('link', { name: 'Fake docs' })).not.toBeInTheDocument();
    expect(screen.getByText('Fake docs')).toBeInTheDocument();
  });
});

describe('getAssistantDocsLinkTarget', () => {
  it.each([
    ['https://docs.nvidia.com/nemo-helix', 'https://docs.nvidia.com/nemo-helix'],
    [
      'https://docs.nvidia.com/nemo-helix/latest/documentation/studio/agents',
      'https://docs.nvidia.com/nemo-helix/latest/documentation/studio/agents',
    ],
    [
      'https://docs.nvidia.com/nemo-helix/latest/documentation/studio#overview',
      'https://docs.nvidia.com/nemo-helix/latest/documentation/studio#overview',
    ],
  ])('accepts docs-site link %s', (href, expected) => {
    expect(getAssistantDocsLinkTarget(href)).toBe(expected);
  });

  it.each([
    ['undefined', undefined],
    ['empty', ''],
    ['relative path', '/nemo-helix'],
    ['plain http', 'http://docs.nvidia.com/nemo-helix'],
    ['another NVIDIA site', 'https://developer.nvidia.com/nemo-helix'],
    ['a lookalike host', 'https://docs.nvidia.com.evil.example/nemo-helix'],
    ['embedded credentials', 'https://docs.nvidia.com@evil.example/nemo-helix'],
    ['a credentials userinfo on the docs host', 'https://user:pw@docs.nvidia.com/nemo-helix'],
    ['a lookalike path prefix', 'https://docs.nvidia.com/nemo-helix-evil'],
    ['a different docs product', 'https://docs.nvidia.com/other-product/latest'],
    ['path traversal out of the docs root', 'https://docs.nvidia.com/nemo-helix/../other-product'],
    ['a javascript URL', 'javascript:alert(1)'],
  ])('rejects %s', (_name, href) => {
    expect(getAssistantDocsLinkTarget(href)).toBeUndefined();
  });
});
