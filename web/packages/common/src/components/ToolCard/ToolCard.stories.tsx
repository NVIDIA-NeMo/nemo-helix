// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToolCard } from '@nemo/common/src/components/ToolCard';
import type { Meta, StoryObj } from '@storybook/react';

const meta: Meta<typeof ToolCard> = {
  component: ToolCard,
  title: 'Studio Common/ToolCard',
  decorators: [(Story) => <div className="w-[440px]">{Story()}</div>],
};

export default meta;

type Story = StoryObj<typeof ToolCard>;

export const Default: Story = {
  args: {
    tool: {
      type: 'function',
      function: {
        name: 'get_weather',
        description:
          'Returns the current weather conditions and a short forecast for a given location.',
        strict: true,
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string', description: 'City name or "lat,long" coordinates.' },
            units: { type: 'string', description: 'metric or imperial. Defaults to metric.' },
            days: { type: 'integer', description: 'Forecast length, 1–7 days. Defaults to 1.' },
          },
          required: ['location'],
        },
      },
    },
  },
};

export const NoParameters: Story = {
  args: {
    tool: { type: 'function', function: { name: 'get_current_time' } },
  },
};
