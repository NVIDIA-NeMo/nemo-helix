// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { validateDraft } from '@studio/components/CreateCustomizationStart/aiDraft';
import { DraftResult } from '@studio/components/CreateCustomizationStart/DraftResult';
import { automodelDraft, INPUTS } from '@studio/components/CreateCustomizationStart/testFixtures';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const summaryOf = () => {
  const result = validateDraft(
    JSON.stringify(automodelDraft({}, { needs_from_user: ['Register a teacher model.'] })),
    INPUTS
  );
  if (result.status !== 'valid') throw new Error('fixture draft should validate');
  return result.summary;
};

describe('DraftResult', () => {
  it('lays out what will be trained beside why', () => {
    render(
      <TestProviders>
        <DraftResult summary={summaryOf()} config="{}" onEdit={vi.fn()} />
      </TestProviders>
    );

    expect(screen.getByText('llama-8b-ticket-router')).toBeInTheDocument();
    expect(screen.getByText('Training setup')).toBeInTheDocument();
    expect(screen.getByText('default/llama-8b')).toBeInTheDocument();
    expect(screen.getByText('Why these settings')).toBeInTheDocument();
    expect(
      screen.getByText('SFT, because your dataset holds chat transcripts.')
    ).toBeInTheDocument();
    expect(screen.getByText('Register a teacher model.')).toBeInTheDocument();
  });

  it('hands control back to the form on Edit', async () => {
    const onEdit = vi.fn();
    render(
      <TestProviders>
        <DraftResult summary={summaryOf()} config="{}" onEdit={onEdit} />
      </TestProviders>
    );

    await userEvent.setup().click(screen.getByRole('button', { name: /edit/i }));
    expect(onEdit).toHaveBeenCalledTimes(1);
  });

  it('shows only the headline settings, with readable labels, and counts the rest', () => {
    render(
      <TestProviders>
        <DraftResult summary={summaryOf()} config="{}" onEdit={vi.fn()} />
      </TestProviders>
    );

    expect(screen.getByText('Learning rate')).toBeInTheDocument();
    expect(screen.getByText('LoRA rank')).toBeInTheDocument();
    expect(screen.getByText('LoRA alpha')).toBeInTheDocument();
    // Training and fine-tuning type are in the header, so nothing is left uncounted.
    expect(screen.getByText(/Everything else keeps its default/)).toBeInTheDocument();
  });

  it('opens the full request from View config', async () => {
    render(
      <TestProviders>
        <DraftResult
          summary={summaryOf()}
          config={'{"spec": {"model": "default/llama-8b"}}'}
          onEdit={vi.fn()}
        />
      </TestProviders>
    );

    await userEvent.setup().click(screen.getByRole('button', { name: /view config/i }));
    expect(await screen.findByText(/The request the form will submit/)).toBeInTheDocument();
  });
});
