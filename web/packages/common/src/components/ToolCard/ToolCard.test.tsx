// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getToolParameters, ToolCard } from '@nemo/common/src/components/ToolCard';
import type { ChatCompletionTool } from '@nemo/sdk/generated/platform/schema';
import { render, screen } from '@testing-library/react';

const tool: ChatCompletionTool = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Returns the current weather for a location.',
    strict: true,
    parameters: {
      type: 'object',
      properties: {
        location: { type: 'string', description: 'City name.' },
        days: { type: ['integer', 'null'] },
      },
      required: ['location'],
    },
  },
};

describe('getToolParameters', () => {
  it('flattens properties with type, description and required flag', () => {
    expect(getToolParameters(tool.function.parameters)).toEqual([
      { name: 'location', type: 'string', description: 'City name.', required: true },
      { name: 'days', type: 'integer | null', description: undefined, required: false },
    ]);
  });

  it('returns no parameters for a missing or malformed schema', () => {
    expect(getToolParameters(undefined)).toEqual([]);
    expect(getToolParameters({ properties: 'nope' })).toEqual([]);
  });
});

describe('ToolCard', () => {
  it('renders the tool summary, parameters and metadata', () => {
    render(<ToolCard tool={tool} />);

    expect(screen.getByText('get_weather')).toBeInTheDocument();
    expect(screen.getByText('Function')).toBeInTheDocument();
    expect(screen.getByText('Returns the current weather for a location.')).toBeInTheDocument();
    expect(screen.getByText('location')).toBeInTheDocument();
    expect(screen.getByText('City name.')).toBeInTheDocument();
    expect(screen.getAllByText('required')).toHaveLength(1);
    expect(screen.getByText('true')).toBeInTheDocument();
    expect(screen.getByText('2 (1 required)')).toBeInTheDocument();
  });

  it('shows an empty state when the tool takes no parameters', () => {
    render(<ToolCard tool={{ type: 'function', function: { name: 'ping' } }} />);

    expect(screen.getByText('No parameters')).toBeInTheDocument();
    expect(screen.getByText('0 (0 required)')).toBeInTheDocument();
  });
});
