// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DeltaText,
  type DeltaTextSize,
  formatSignedDelta,
} from '@nemo/common/src/components/DeltaText/index';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';

const meta: Meta<typeof DeltaText> = {
  component: DeltaText,
  title: 'Studio Common/DeltaText',
  args: { size: 'md' },
  argTypes: {
    size: { control: 'inline-radio', options: ['xs', 'sm', 'md', 'lg', 'xl'] },
  },
};

export default meta;

type Story = StoryObj<typeof DeltaText>;

export const Improved: Story = {
  args: { value: 0.07 },
};

export const Regressed: Story = {
  args: { value: -0.07 },
};

export const Unchanged: Story = {
  args: { value: 0 },
};

/** Latency fell, which is the good news — the triangle points down but the text stays green. */
export const LowerIsBetter: Story = {
  args: { value: -120, higherIsBetter: false, format: (value) => `${value.toFixed(0)} ms` },
};

/**
 * A qualifier after the value — what the delta is measured against — rides along in `format`, so it
 * picks up the same tint and sits on the same line as the number.
 *
 * `kind` rather than `size` here because the design's 12px falls between the `xs` and `sm` steps.
 */
export const WithQualifier: Story = {
  args: {
    value: -18,
    higherIsBetter: false,
    format: (value) => `${formatSignedDelta(value, 0)}% vs baseline`,
    kind: 'body/semibold/sm',
  },
};

/** The qualifier is context, not part of the magnitude, so keep it out of the spoken name. */
export const WithQualifierCustomLabel: Story = {
  args: {
    ...WithQualifier.args,
    'aria-label': 'Improved by 18 percent versus baseline',
  },
};

const SIZES: [DeltaTextSize, string][] = [
  ['xs', '10px'],
  ['sm', '14px'],
  ['md', '18px'],
  ['lg', '24px'],
  ['xl', '32px'],
];

/** The glyph is sized in `em`, so it holds the text's cap height at every step. */
export const Sizes: Story = {
  render: () => (
    <Stack gap="density-md">
      {SIZES.map(([size, px]) => (
        <Flex key={size} align="center" gap="density-lg">
          <Text kind="label/regular/sm" className="w-20 text-secondary">
            {size} · {px}
          </Text>
          <DeltaText value={0.07} size={size} />
          <DeltaText value={-0.07} size={size} />
          <DeltaText value={0} size={size} />
        </Flex>
      ))}
    </Stack>
  ),
};

/** The compact `xs` treatment the design uses, sitting under the value it qualifies. */
export const InContext: Story = {
  render: () => (
    <Stack gap="density-sm">
      <Text kind="label/bold/2xl" className="tabular-nums">
        0.84
      </Text>
      <DeltaText value={0.07} />
    </Stack>
  ),
};
