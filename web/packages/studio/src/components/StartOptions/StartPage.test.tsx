// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StartPage } from '@studio/components/StartOptions/StartPage';
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
        canContinue
        onContinue={() => undefined}
        {...over}
      />
    </TestProviders>
  );

describe('StartPage', () => {
  it('names the group holding the options', () => {
    renderPage();

    // KUI sets the role but takes its name from a wrapping FormField, which this page has
    // no visible prompt for — without a name the group is announced as an unnamed one.
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

  it('shows the detail panel the caller gives it', () => {
    renderPage({ slotDetail: <div>Recipe list</div> });

    expect(screen.getByText('Recipe list')).toBeInTheDocument();
  });

  it('reports the selection rather than acting on it', async () => {
    const onChange = vi.fn();
    const onContinue = vi.fn();
    renderPage({ onChange, onContinue });

    await userEvent.click(screen.getByRole('radio', { name: 'Describe with AI' }));

    // Choosing a way in reveals its panel; nothing starts until Continue.
    expect(onChange).toHaveBeenCalledWith('ai');
    expect(onContinue).not.toHaveBeenCalled();
  });

  it('says what is missing while Continue is disabled', () => {
    renderPage({ canContinue: false, blockedHint: 'Pick a recipe to continue.' });

    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled();
    expect(screen.getByText('Pick a recipe to continue.')).toBeInTheDocument();
  });

  it('drops the hint once Continue is available', () => {
    renderPage({ canContinue: true, blockedHint: 'Pick a recipe to continue.' });

    expect(screen.queryByText('Pick a recipe to continue.')).not.toBeInTheDocument();
  });
});
