// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StartPage } from '@studio/components/StartOptions/StartPage';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Box, Plus } from 'lucide-react';

const OPTIONS = [
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Start with an empty form.',
    icon: Plus,
    tag: { label: 'Advanced', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
];

const group = (over: Partial<StartTemplateGroup> = {}): StartTemplateGroup => ({
  id: 'g1',
  title: 'Group One',
  templates: [{ id: 't1', name: 'Template One', description: 'Does a thing.', icon: Box }],
  ...over,
});

const renderPage = (groups: StartTemplateGroup[], onSelect: (id: string) => void = () => undefined) =>
  render(
    <TestProviders>
      <StartPage
        heading="Page Title"
        headingDescription="Page Description"
        options={OPTIONS}
        templateGroups={groups}
        onSelect={onSelect}
      />
    </TestProviders>
  );

describe('StartPage accessibility', () => {
  it('names the group holding the options and templates', () => {
    renderPage([group()]);

    // The tiles carry no visible prompt, so without this the set is announced unnamed.
    expect(screen.getByRole('group', { name: 'How do you want to start?' })).toBeInTheDocument();
  });
});

describe('StartPage badges', () => {
  it('names each template section from its own heading', () => {
    renderPage([group()]);

    expect(screen.getByRole('group', { name: 'Group One' })).toBeInTheDocument();
  });

  it('leaves the divider label and tag readable', () => {
    render(
      <TestProviders>
        <StartPage
          heading="Page Title"
          headingDescription="Page Description"
          options={OPTIONS}
          templateGroups={[group()]}
          templatesTag={{ label: 'Intermediate', color: 'gray', kind: 'solid' }}
          onSelect={() => undefined}
        />
      </TestProviders>
    );

    // Only the rules either side are decorative; the label and tag carry meaning.
    // `getByText` reads hidden nodes too, so skip anything under aria-hidden — otherwise
    // this passes whether or not the row is hidden.
    const readable = { ignore: '[aria-hidden="true"], [aria-hidden="true"] *' } as const;
    expect(screen.getByText('OR START FROM A TEMPLATE', readable)).toBeInTheDocument();
    expect(screen.getByText('Intermediate', readable)).toBeInTheDocument();
  });

  it('shows an option tag on its tile', () => {
    renderPage([group()]);

    expect(screen.getByText('Advanced')).toBeInTheDocument();
  });

  it('shows the templates tag on the divider', () => {
    render(
      <TestProviders>
        <StartPage
          heading="Page Title"
          headingDescription="Page Description"
          options={OPTIONS}
          templateGroups={[group()]}
          templatesTag={{ label: 'Intermediate', color: 'gray', kind: 'solid' }}
          onSelect={() => undefined}
        />
      </TestProviders>
    );

    expect(screen.getByText('Intermediate')).toBeInTheDocument();
  });

  it('leaves the divider bare when no templates tag is given', () => {
    renderPage([group()]);

    expect(screen.queryByText('Intermediate')).not.toBeInTheDocument();
  });
});

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

  it('offers no tile for a group that is still loading', () => {
    renderPage([group({ templates: [], loading: true })]);

    expect(screen.queryByRole('button', { name: /Template One/ })).not.toBeInTheDocument();
  });
});

describe('StartPage selection', () => {
  it('runs the flow on the first click, with no confirm step', async () => {
    const onSelect = vi.fn();
    renderPage([group()], onSelect);

    await userEvent.click(screen.getByRole('button', { name: /Build from scratch/ }));

    expect(onSelect).toHaveBeenCalledExactlyOnceWith('scratch');
    expect(screen.queryByRole('button', { name: 'Continue' })).not.toBeInTheDocument();
  });

  it('starts a template the same way', async () => {
    const onSelect = vi.fn();
    renderPage([group()], onSelect);

    await userEvent.click(screen.getByRole('button', { name: /Template One/ }));

    expect(onSelect).toHaveBeenCalledExactlyOnceWith('t1');
  });

  it('ignores clicks on a tile that is not wired up', async () => {
    const onSelect = vi.fn();
    render(
      <TestProviders>
        <StartPage
          heading="Page Title"
          headingDescription="Page Description"
          options={[{ ...OPTIONS[0], enabled: false }]}
          onSelect={onSelect}
        />
      </TestProviders>
    );

    await userEvent.click(screen.getByRole('button', { name: /Build from scratch/ }));

    expect(onSelect).not.toHaveBeenCalled();
  });

  it('locks every tile while a pick is being acted on', async () => {
    const onSelect = vi.fn();
    render(
      <TestProviders>
        <StartPage
          heading="Page Title"
          headingDescription="Page Description"
          options={OPTIONS}
          templateGroups={[group()]}
          onSelect={onSelect}
          disabled
          slotBanner={<div>Registering model</div>}
        />
      </TestProviders>
    );

    // Without this a second setup can start over the first one mid-flight.
    await userEvent.click(screen.getByRole('button', { name: /Build from scratch/ }));

    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.getByText('Registering model')).toBeInTheDocument();
  });
});
