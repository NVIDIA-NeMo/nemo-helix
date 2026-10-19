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
  it('renders all start options, with no Continue footer', () => {
    renderStart();

    expect(screen.getByText('Describe with AI')).toBeInTheDocument();
    expect(screen.getByText('Start from a template')).toBeInTheDocument();
    expect(screen.getByText('Build from scratch')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /continue/i })).not.toBeInTheDocument();
  });

  it('opens with the recipes already showing', () => {
    renderStart();

    // The likeliest way in, so the page starts there rather than on an empty panel.
    expect(screen.getByText('Instruction fine-tuning (SFT)')).toBeInTheDocument();
  });

  it('goes straight to the builder from "Build from scratch"', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByRole('radio', { name: /Build from scratch/ }));

    expect(onContinue).toHaveBeenCalledTimes(1);
    expect(onContinue).toHaveBeenCalledWith({ optionId: 'scratch' });
  });

  it('goes straight to the builder from a template', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByRole('radio', { name: /Instruction fine-tuning \(SFT\)/ }));

    expect(onContinue).toHaveBeenCalledTimes(1);
    expect(onContinue).toHaveBeenCalledWith({
      optionId: 'template',
      templateId: 'sft-instruction',
    });
  });

  it('opens Describe with AI on its own page, with Continue held until a config validates', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByRole('radio', { name: /Describe with AI/ }));

    expect(
      screen.getByRole('textbox', { name: /what do you want to generate/i })
    ).toBeInTheDocument();
    expect(screen.queryByText('Instruction fine-tuning (SFT)')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
    expect(screen.getByText('Generate a valid config to continue.')).toBeInTheDocument();
    expect(onContinue).not.toHaveBeenCalled();
  });

  it('goes back from Describe with AI to the options', async () => {
    const user = userEvent.setup();
    renderStart();

    await user.click(screen.getByRole('radio', { name: /Describe with AI/ }));
    await user.click(screen.getByRole('button', { name: 'Back' }));

    expect(screen.getByText('Instruction fine-tuning (SFT)')).toBeInTheDocument();
  });

  it('blocks an empty generate with field errors instead of calling the model', async () => {
    const user = userEvent.setup();
    renderStart();

    await user.click(screen.getByRole('radio', { name: /Describe with AI/ }));
    await user.click(screen.getByRole('button', { name: /generate/i }));

    expect(await screen.findByText('Choose a model to draft the config.')).toBeInTheDocument();
    expect(screen.getByText('Describe the fileset you want.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
  });
});
