// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FinetuningType } from '@nemo/sdk/generated/platform/schema/FinetuningType';
import { Flex, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { KindTag, TrainingType } from '@studio/components/dataViews/CustomModelsDataView/KindTag';
import { fn } from 'storybook/test';

const meta = {
  component: KindTag,
  title: 'Components/KindTag',
  decorators: [
    (Story) => (
      <Flex gap="density-lg" align="center">
        <Story />
      </Flex>
    ),
  ],
  argTypes: {
    finetuningType: {
      control: 'select',
      options: Object.values(FinetuningType),
    },
    trainingType: {
      control: 'select',
      options: [undefined, ...Object.values(TrainingType)],
    },
  },
  args: {
    onClick: fn(),
  },
} satisfies Meta<typeof KindTag>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    finetuningType: FinetuningType.lora,
    trainingType: TrainingType.sft,
  },
};

export const AllFinetuningTypes: Story = {
  name: 'All Finetuning Types',
  args: { finetuningType: FinetuningType.lora },
  render: (args) => (
    <Flex gap="density-lg" align="center" wrap="wrap">
      {Object.values(FinetuningType).map((ft) => (
        <KindTag key={ft} finetuningType={ft} onClick={args.onClick} />
      ))}
    </Flex>
  ),
};

export const FinetuningTrainingMatrix: Story = {
  name: 'Finetuning × Training Matrix',
  args: { finetuningType: FinetuningType.lora },
  render: (args) => (
    <Flex direction="col" gap="density-lg">
      {Object.values(FinetuningType).map((ft) => (
        <Flex key={ft} gap="density-lg" align="center" wrap="wrap">
          <Text kind="body/regular/sm" className="w-40 shrink-0">
            {ft}
          </Text>
          <KindTag finetuningType={ft} onClick={args.onClick} />
          {Object.values(TrainingType).map((tt) => (
            <KindTag
              key={`${ft}-${tt}`}
              finetuningType={ft}
              trainingType={tt}
              onClick={args.onClick}
            />
          ))}
        </Flex>
      ))}
    </Flex>
  ),
};
