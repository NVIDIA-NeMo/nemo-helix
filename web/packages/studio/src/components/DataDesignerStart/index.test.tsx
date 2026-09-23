// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DataDesignerStart } from '@studio/components/DataDesignerStart';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const renderStart = () => {
  const onContinue = vi.fn();
  render(<DataDesignerStart workspace="default" onContinue={onContinue} />);
  return { onContinue };
};

const continueButton = () => screen.getByRole('button', { name: /continue/i });

describe('DataDesignerStart', () => {
  it('offers every way in at once', () => {
    renderStart();

    expect(screen.getByText('Describe with AI')).toBeInTheDocument();
    expect(screen.getByText('Build from scratch')).toBeInTheDocument();
    // Templates are picked outright, so they are on the page rather than behind an option.
    expect(screen.getByText('Instruction fine-tuning (SFT)')).toBeInTheDocument();
  });

  it('groups the templates under the sections their tags name', () => {
    renderStart();

    expect(screen.getByText('Evaluation')).toBeInTheDocument();
    expect(screen.getByText('Fine-tuning')).toBeInTheDocument();
    // Tags that name no section of their own collect here rather than each getting a heading.
    expect(screen.getByText('Other')).toBeInTheDocument();
  });

  it('badges each way in with how much it asks of you', () => {
    renderStart();

    expect(screen.getByText('Beginner')).toBeInTheDocument();
    expect(screen.getByText('Advanced')).toBeInTheDocument();
    expect(screen.getByText('Intermediate')).toBeInTheDocument();
  });

  it('keeps Continue disabled until something is picked', async () => {
    const user = userEvent.setup();
    renderStart();
    expect(continueButton()).toBeDisabled();

    await user.click(screen.getByText('Build from scratch'));
    expect(continueButton()).toBeEnabled();
  });

  it('hands "from scratch" straight over', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByText('Build from scratch'));
    await user.click(continueButton());

    expect(onContinue).toHaveBeenCalledWith({ optionId: 'scratch' });
  });

  it('hands a template over by id', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByText('Instruction fine-tuning (SFT)'));
    await user.click(continueButton());

    expect(onContinue).toHaveBeenCalledWith({
      optionId: 'template',
      templateId: 'sft-instruction',
    });
  });

  it('replaces a template selection when an option is picked instead', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByText('Instruction fine-tuning (SFT)'));
    await user.click(screen.getByText('Build from scratch'));
    await user.click(continueButton());

    // One radio group across both, so the last pick wins rather than both being held.
    expect(onContinue).toHaveBeenCalledWith({ optionId: 'scratch' });
  });

  describe('describe with AI', () => {
    /** The only option that asks for something else before the canvas. */
    it('goes to the drafting screen rather than straight to the canvas', async () => {
      const user = userEvent.setup();
      const { onContinue } = renderStart();

      await user.click(screen.getByText('Describe with AI'));
      await user.click(continueButton());

      expect(
        screen.getByRole('textbox', { name: /what do you want to generate/i })
      ).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    it('will not continue until a config is generated', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Describe with AI'));
      await user.click(continueButton());

      expect(continueButton()).toBeDisabled();
    });

    it('blocks an empty generate with field errors instead of calling the model', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Describe with AI'));
      await user.click(continueButton());
      await user.click(screen.getByRole('button', { name: /generate/i }));

      expect(await screen.findByText('Choose a model to draft the config.')).toBeInTheDocument();
      expect(screen.getByText('Describe the fileset you want.')).toBeInTheDocument();
      expect(continueButton()).toBeDisabled();
    });

    it('goes back to the options without having started anything', async () => {
      const user = userEvent.setup();
      const { onContinue } = renderStart();

      await user.click(screen.getByText('Describe with AI'));
      await user.click(continueButton());
      await user.click(screen.getByRole('button', { name: /back/i }));

      expect(screen.getByText('Build from scratch')).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });
  });
});
