/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import { RadioCard } from '@nemo/common/src/components/RadioCard/index';
import { Flex, RadioGroupRoot, Stack } from '@nvidia/foundations-react-core';
import { CubeStack } from '@nvidia/react-gui-icons/icons';
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';

const meta: Meta<typeof RadioCard> = {
  component: RadioCard,
  title: 'Studio Common/RadioCard',
  decorators: [
    (Story) => (
      <div className="p-4 max-w-md">
        <Story />
      </div>
    ),
  ],
};

export default meta;

type Story = StoryObj<typeof RadioCard>;

export const Default: Story = {
  render: function DefaultStory() {
    const [value, setValue] = useState<string>('option-1');
    return (
      <RadioGroupRoot name="radio-card-default" value={value} onValueChange={setValue}>
        <Stack gap="density-md">
          <RadioCard
            value="option-1"
            label="Radio Label"
            description="Lorem ipsum dolor sit amet."
            icon={<CubeStack width={24} height={24} />}
          />
          <RadioCard
            value="option-2"
            label="Second option"
            description="Another choice with a short description."
            icon={<CubeStack width={24} height={24} />}
          />
        </Stack>
      </RadioGroupRoot>
    );
  },
};

export const WithoutIcon: Story = {
  render: function WithoutIconStory() {
    const [value, setValue] = useState<string>('a');
    return (
      <RadioGroupRoot name="radio-card-no-icon" value={value} onValueChange={setValue}>
        <Stack gap="density-md">
          <RadioCard value="a" label="Option A" description="No icon on this card." />
          <RadioCard value="b" label="Option B" description="Also no icon." />
        </Stack>
      </RadioGroupRoot>
    );
  },
};

export const LabelOnly: Story = {
  render: function LabelOnlyStory() {
    const [value, setValue] = useState<string>('simple');
    return (
      <RadioGroupRoot name="radio-card-label-only" value={value} onValueChange={setValue}>
        <Stack gap="density-md">
          <RadioCard value="simple" label="Simple option" />
          <RadioCard value="minimal" label="Minimal card" />
        </Stack>
      </RadioGroupRoot>
    );
  },
};

export const LabelSideLeft: Story = {
  render: function LabelSideLeftStory() {
    const [value, setValue] = useState<string>('option-1');
    return (
      <RadioGroupRoot name="radio-card-right" value={value} onValueChange={setValue}>
        <Stack gap="density-md">
          <RadioCard
            value="option-1"
            label="Radio on the right"
            description="The radio control appears on the right when labelSide is left."
            icon={<CubeStack width={24} height={24} />}
            labelSide="left"
          />
          <RadioCard
            value="option-2"
            label="Second option"
            description="Also with radio on the right."
            labelSide="left"
          />
        </Stack>
      </RadioGroupRoot>
    );
  },
};

export const HiddenIndicator: Story = {
  render: function HiddenIndicatorStory() {
    const [value, setValue] = useState<string>('dataset');
    return (
      <RadioGroupRoot
        name="radio-card-hidden-indicator"
        orientation="horizontal"
        className="w-full"
        value={value}
        onValueChange={setValue}
      >
        <Flex gap="density-xl" className="w-full *:flex-1">
          <RadioCard
            value="dataset"
            label="Dataset-driven"
            description="Score a fixed dataset of prompts and expected answers."
            showIndicator={false}
          />
          <RadioCard
            value="task"
            label="Task-driven"
            description="Score the agent against per-task metrics."
            showIndicator={false}
          />
        </Flex>
      </RadioGroupRoot>
    );
  },
};

export const RichDescription: Story = {
  render: function RichDescriptionStory() {
    const [value, setValue] = useState<string>('option-1');
    return (
      <RadioGroupRoot name="radio-card-rich" value={value} onValueChange={setValue}>
        <Stack gap="density-md">
          <RadioCard
            value="option-1"
            label="Option with link"
            description={
              <>
                This description includes a{' '}
                <a
                  href="https://example.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline text-primary-foreground hover:no-underline"
                >
                  link to documentation
                </a>{' '}
                and <strong>bold text</strong> for emphasis.
              </>
            }
            icon={<CubeStack width={24} height={24} />}
          />
          <RadioCard
            value="option-2"
            label="Option with list"
            description={
              <span>
                Supports multiple elements:
                <ul className="list-disc list-inside mt-1 space-y-0.5">
                  <li>Bullet one</li>
                  <li>Bullet two</li>
                  <li>Bullet three</li>
                </ul>
              </span>
            }
          />
        </Stack>
      </RadioGroupRoot>
    );
  },
};
