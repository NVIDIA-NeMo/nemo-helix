// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Breadcrumbs } from '@studio/components/Breadcrumbs';
import { BreadcrumbsProvider } from '@studio/providers/breadcrumbs/BreadcrumbsProvider';
import {
  useBreadcrumbs,
  type BreadcrumbsItemProps,
} from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { render, screen } from '@studio/tests/util/render';
import type { FC } from 'react';
import { MemoryRouter } from 'react-router';

const SetCrumbs: FC<{ items: BreadcrumbsItemProps[] }> = ({ items }) => {
  useBreadcrumbs({ items });
  return null;
};

const renderCrumbs = (items: BreadcrumbsItemProps[]) =>
  render(
    <MemoryRouter initialEntries={['/workspaces/default/agents/optimizations/sweep-3']}>
      <BreadcrumbsProvider>
        <SetCrumbs items={items} />
        <Breadcrumbs />
      </BreadcrumbsProvider>
    </MemoryRouter>
  );

describe('Breadcrumbs', () => {
  it('strips the query string by default, so an "up" link does not leak detail state', () => {
    renderCrumbs([{ slotLabel: 'Agents', href: '/workspaces/default/agents?tab=optimizations' }]);

    expect(screen.getByRole('link', { name: 'Agents' })).toHaveAttribute(
      'href',
      '/workspaces/default/agents'
    );
  });

  it('keeps the query string when the item opts in with preserveQuery', () => {
    renderCrumbs([
      {
        slotLabel: 'Optimizations',
        href: '/workspaces/default/agents/email-analyzer?tab=optimizations',
        preserveQuery: true,
      },
    ]);

    // Without this the parent route falls back to its default tab — the bug this flag exists for.
    expect(screen.getByRole('link', { name: 'Optimizations' })).toHaveAttribute(
      'href',
      '/workspaces/default/agents/email-analyzer?tab=optimizations'
    );
  });
});
