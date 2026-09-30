// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CreateFilesetStart } from '@studio/components/CreateFilesetStart';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const renderStart = () => {
  const onContinue = vi.fn();
  render(<CreateFilesetStart workspace="default" onContinue={onContinue} />);
  return { onContinue };
};

describe('CreateFilesetStart', () => {
  it('renders all start options', () => {
    renderStart();

    expect(screen.getByText('Describe with AI')).toBeInTheDocument();
    expect(screen.getByText('Start from a template')).toBeInTheDocument();
    expect(screen.getByText('Build from scratch')).toBeInTheDocument();
  });

  it('opens on the template rung with its recipes already showing', () => {
    renderStart();

    // The likeliest way in, so the page starts there rather than on an empty panel.
    expect(screen.getByRole('radio', { name: 'Start from a template' })).toBeChecked();
    expect(screen.getByText('Instruction fine-tuning (SFT)')).toBeInTheDocument();
  });

  it('waits for a recipe before enabling Continue', () => {
    renderStart();

    // The option is chosen, but the option alone is not a job.
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
    expect(screen.getByText('Pick a recipe to continue.')).toBeInTheDocument();
  });

  it('selecting Build from scratch reveals Continue and invokes onContinue with "scratch"', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByText('Build from scratch'));

    const continueButton = screen.getByRole('button', { name: /continue/i });
    expect(continueButton).toBeEnabled();

    await user.click(continueButton);
    expect(onContinue).toHaveBeenCalledTimes(1);
    expect(onContinue).toHaveBeenCalledWith({ optionId: 'scratch' });
  });

  it('reveals template cards but keeps Continue disabled until a template is chosen', async () => {
    const user = userEvent.setup();
    renderStart();

    await user.click(screen.getByText('Start from a template'));

    expect(screen.getByText('Instruction fine-tuning (SFT)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
    expect(screen.getByText('Pick a recipe to continue.')).toBeInTheDocument();
  });

  it('choosing a template enables Continue and invokes onContinue with the template id', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByText('Start from a template'));
    await user.click(screen.getByText('Instruction fine-tuning (SFT)'));

    await user.click(screen.getByRole('button', { name: /continue/i }));

    expect(onContinue).toHaveBeenCalledTimes(1);
    expect(onContinue).toHaveBeenCalledWith({
      optionId: 'template',
      templateId: 'sft-instruction',
    });
  });

  it('switching options clears a prior template selection', async () => {
    const user = userEvent.setup();
    renderStart();

    await user.click(screen.getByText('Start from a template'));
    await user.click(screen.getByText('Instruction fine-tuning (SFT)'));
    expect(screen.getByRole('button', { name: /continue/i })).toBeEnabled();

    await user.click(screen.getByText('Build from scratch'));
    await user.click(screen.getByText('Start from a template'));

    // Template selection was reset when the option changed, so Continue is blocked again.
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
  });

  it('selecting Describe with AI keeps Continue disabled until a config is generated', async () => {
    const user = userEvent.setup();
    renderStart();

    await user.click(screen.getByText('Describe with AI'));

    expect(
      screen.getByRole('textbox', { name: /what do you want to generate/i })
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
    expect(screen.getByText('Generate a valid config to continue.')).toBeInTheDocument();
  });

  it('blocks an empty generate with field errors instead of calling the model', async () => {
    const user = userEvent.setup();
    renderStart();

    await user.click(screen.getByText('Describe with AI'));
    await user.click(screen.getByRole('button', { name: /generate/i }));

    expect(await screen.findByText('Choose a model to draft the config.')).toBeInTheDocument();
    expect(screen.getByText('Describe the fileset you want.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
  });
});
