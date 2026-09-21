// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StartPage } from '@studio/components/StartOptions/StartPage';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import { Box, Plus } from 'lucide-react';

const OPTIONS = [
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Start with an empty form.',
    icon: Plus,
    enabled: true,
  },
];

const group = (over: Partial<StartTemplateGroup> = {}): StartTemplateGroup => ({
  id: 'g1',
  title: 'Group One',
  templates: [{ id: 't1', name: 'Template One', description: 'Does a thing.', icon: Box }],
  ...over,
});

const renderPage = (groups: StartTemplateGroup[], value: string | null = null) =>
  render(
    <TestProviders>
      <StartPage
        heading="Page Title"
        headingDescription="Page Description"
        options={OPTIONS}
        templateGroups={groups}
        value={value}
        onChange={() => undefined}
        canContinue={value !== null}
        onContinue={() => undefined}
      />
    </TestProviders>
  );

describe('StartPage template groups', () => {
  it('renders every group it is given', () => {
    renderPage([
      group(),
      group({
        id: 'g2',
        title: 'Group Two',
        templates: [{ id: 't2', name: 'Template Two', description: 'Does another.', icon: Box }],
      }),
    ]);

    expect(screen.getByText('Group One')).toBeInTheDocument();
    expect(screen.getByText('Group Two')).toBeInTheDocument();
  });

  /** The section must hold its place, or the page reflows when the fetch lands. */
  it('keeps a loading group on the page instead of dropping it', () => {
    renderPage([group({ templates: [], loading: true })]);

    expect(screen.getByText('Group One')).toBeInTheDocument();
    expect(screen.getAllByLabelText('Loading Group One').length).toBeGreaterThan(0);
  });

  it('drops a group that finished loading with nothing in it', () => {
    renderPage([group({ templates: [] })]);

    expect(screen.queryByText('Group One')).not.toBeInTheDocument();
  });

  it('does not name a placeholder in the footer', () => {
    // A loading group carries no selectable template, so nothing can resolve to its name.
    renderPage([group({ templates: [], loading: true })], 't1');

    expect(screen.getByText('Select an option above to continue')).toBeInTheDocument();
  });
});
