// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StartPage } from '@studio/components/StartOptions/StartPage';
import { StartSubPage } from '@studio/components/StartOptions/StartSubPage';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Plus, Sparkles } from 'lucide-react';

const OPTIONS = [
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Start with an empty form.',
    icon: Plus,
    tag: { label: 'Advanced', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
  {
    id: 'ai',
    title: 'Describe with AI',
    description: 'Say what you need.',
    icon: Sparkles,
    tag: { label: 'Beginner', color: 'gray', kind: 'solid' } as const,
    enabled: true,
  },
];

const renderPage = (over: Partial<React.ComponentProps<typeof StartPage>> = {}) =>
  render(
    <TestProviders>
      <StartPage
        heading="Page Title"
        headingDescription="Page Description"
        options={OPTIONS}
        value="scratch"
        onChange={() => undefined}
        {...over}
      />
    </TestProviders>
  );

const renderSubPage = (over: Partial<React.ComponentProps<typeof StartSubPage>> = {}) =>
  render(
    <TestProviders>
      <StartSubPage
        heading="Describe with AI"
        headingDescription="Say what you need."
        onBack={() => undefined}
        canContinue
        onContinue={() => undefined}
        {...over}
      >
        <div>Option content</div>
      </StartSubPage>
    </TestProviders>
  );

const GROUPS = [
  {
    id: 'recipes',
    title: 'Recipes',
    templates: [
      { id: 'sft', name: 'SFT recipe', description: 'Supervised.', icon: Plus },
      {
        id: 'mine',
        name: 'My recipe',
        description: 'Saved.',
        icon: Plus,
        action: <button type="button">Delete My recipe</button>,
      },
    ],
  },
];

describe('StartPage', () => {
  it('names the group holding the options', () => {
    renderPage();

    expect(
      screen.getByRole('radiogroup', { name: 'How do you want to start?' })
    ).toBeInTheDocument();
  });

  it('ties an option tag to the choice it qualifies', () => {
    renderPage();

    // The tag is not part of the radio's name, so without this it is never read out
    // against the option it belongs to.
    expect(screen.getByRole('radio', { name: 'Build from scratch' })).toHaveAttribute(
      'aria-describedby',
      'scratch-tag'
    );
  });

  it('shows the content the caller gives it', () => {
    renderPage({ slotDetail: <div>Recipe list</div> });

    expect(screen.getByText('Recipe list')).toBeInTheDocument();
  });

  it('reports a card click to the caller, with no Continue step', async () => {
    const onChange = vi.fn();
    renderPage({ onChange });

    await userEvent.click(screen.getByRole('radio', { name: 'Describe with AI' }));

    expect(onChange).toHaveBeenCalledWith('ai');
    expect(screen.queryByRole('button', { name: 'Continue' })).not.toBeInTheDocument();
  });

  it('does not act while locked', async () => {
    const onChange = vi.fn();
    renderPage({ onChange, disabled: true });

    await userEvent.click(screen.getByRole('radio', { name: 'Describe with AI' }));

    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('StartSubPage', () => {
  it('says what is missing while Continue is disabled', () => {
    renderSubPage({ canContinue: false, blockedHint: 'Draft settings to continue.' });

    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled();
    expect(screen.getByText('Draft settings to continue.')).toBeInTheDocument();
  });

  it('drops the hint once Continue is available', async () => {
    const onContinue = vi.fn();
    renderSubPage({ onContinue, blockedHint: 'Draft settings to continue.' });

    expect(screen.queryByText('Draft settings to continue.')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }));
    expect(onContinue).toHaveBeenCalled();
  });

  it('goes back to the options', async () => {
    const onBack = vi.fn();
    renderSubPage({ onBack });

    await userEvent.click(screen.getByRole('button', { name: 'Back' }));

    expect(onBack).toHaveBeenCalled();
    expect(screen.getByText('Option content')).toBeInTheDocument();
  });
});

describe('TemplateGroups', () => {
  const renderGroups = (over: Partial<React.ComponentProps<typeof TemplateGroups>> = {}) =>
    render(
      <TestProviders>
        <TemplateGroups groups={GROUPS} onSelect={() => undefined} {...over} />
      </TestProviders>
    );

  it('starts from a template on click', async () => {
    const onSelect = vi.fn();
    renderGroups({ onSelect });

    await userEvent.click(screen.getByRole('radio', { name: 'SFT recipe' }));

    expect(onSelect).toHaveBeenCalledWith('sft');
  });

  it('keeps a tile action apart from the tile', async () => {
    const onSelect = vi.fn();
    renderGroups({ onSelect });

    await userEvent.click(screen.getByRole('button', { name: 'Delete My recipe' }));

    expect(onSelect).not.toHaveBeenCalled();
  });

  it('says what is running on the template being set up', () => {
    renderGroups({ pendingId: 'sft', pendingLabel: 'Registering model…', disabled: true });

    expect(screen.getByRole('status')).toHaveTextContent('Registering model…');
    expect(screen.getByRole('radio', { name: 'My recipe' })).toBeDisabled();
  });
});
