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

/** The drafting step's own Continue; the landing page no longer has one. */
const continueButton = () => screen.getByRole('button', { name: /continue/i });

describe('CreateFilesetStart', () => {
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

  it('hands "from scratch" over on the click itself', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByText('Build from scratch'));

    // No confirm step: the pick is the action.
    expect(onContinue).toHaveBeenCalledExactlyOnceWith({ optionId: 'scratch' });
    expect(screen.queryByRole('button', { name: /continue/i })).not.toBeInTheDocument();
  });

  it('hands a template over by id on the click itself', async () => {
    const user = userEvent.setup();
    const { onContinue } = renderStart();

    await user.click(screen.getByText('Instruction fine-tuning (SFT)'));

    expect(onContinue).toHaveBeenCalledExactlyOnceWith({
      optionId: 'template',
      templateId: 'sft-instruction',
    });
  });

  describe('describe with AI', () => {
    /** The only option that asks for something else before the canvas. */
    it('goes to the drafting screen rather than straight to the canvas', async () => {
      const user = userEvent.setup();
      const { onContinue } = renderStart();

      await user.click(screen.getByText('Describe with AI'));

      expect(
        screen.getByRole('textbox', { name: /what do you want to generate/i })
      ).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    it('will not continue until a config is generated', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Describe with AI'));

      expect(continueButton()).toBeDisabled();
    });

    it('blocks an empty generate with field errors instead of calling the model', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Describe with AI'));
      await user.click(screen.getByRole('button', { name: /generate/i }));

      expect(await screen.findByText('Choose a model to draft the config.')).toBeInTheDocument();
      expect(screen.getByText('Describe the fileset you want.')).toBeInTheDocument();
      expect(continueButton()).toBeDisabled();
    });

    it('goes back to the options without having started anything', async () => {
      const user = userEvent.setup();
      const { onContinue } = renderStart();

      await user.click(screen.getByText('Describe with AI'));
      await user.click(screen.getByRole('button', { name: /back/i }));

      expect(screen.getByText('Build from scratch')).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });
  });
});
