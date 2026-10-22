// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Sparkle } from '@nvidia/react-gui-icons/icons';
import type { Meta, StoryObj } from '@storybook/react';
import { EvalCard } from '@studio/components/evaluation/EvalCard';

const meta = {
  component: EvalCard,
  title: 'Components/EvalCard',
  decorators: [
    (Story) => (
      <div className="w-[400px]">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof EvalCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    name: 'nemotron3-prompt-ideal',
  },
};

export const WithDescription: Story = {
  args: {
    name: 'nemotron3-prompt-ideal',
    description: 'Evaluates whether the model response matches the ideal prompt format.',
  },
};

export const LlmJudgeType: Story = {
  args: {
    name: 'response-quality',
    type: 'llm-judge',
    description: 'Uses an LLM judge to score response quality.',
  },
};

export const CustomType: Story = {
  args: {
    name: 'bleu-score',
    type: 'string-match',
    description: 'Measures n-gram overlap between prediction and reference.',
  },
};

export const FullCard: Story = {
  args: {
    name: 'nemotron3-prompt-ideal',
    type: 'llm-judge',
    description: 'Evaluates whether the model response follows the ideal prompt format.',
  },
};

export const CustomIcon: Story = {
  args: {
    name: 'custom-metric',
    type: 'llm-judge',
    description: 'A metric with a custom icon.',
    icon: <Sparkle size="12" />,
  },
};

export const LongContent: Story = {
  args: {
    name: 'extremely-long-metric-name-that-should-be-truncated-within-the-card-header',
    type: 'llm-judge',
    description:
      'This is a very long description that tests how the card handles overflow. It should be truncated with an ellipsis to keep the layout consistent.',
  },
};
