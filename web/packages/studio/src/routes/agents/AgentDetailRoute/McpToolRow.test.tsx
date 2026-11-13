// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { McpToolRow } from '@studio/routes/agents/AgentDetailRoute/McpToolRow';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('McpToolRow', () => {
  it('expands a long description on demand', async () => {
    const description = `Get the current weather for a city. ${'More detail. '.repeat(10)}The end.`;
    render(<McpToolRow tool={{ name: 'get_current_weather', description, exposed: true }} />);

    expect(screen.getByText('get_current_weather')).toBeInTheDocument();
    expect(screen.queryByText(/The end\./)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show more' }));
    expect(screen.getByText(/The end\./)).toBeInTheDocument();
    expect(screen.queryByText('Blocked')).not.toBeInTheDocument();
  });

  it('marks blocked tools and renders a leading control', () => {
    render(
      <McpToolRow
        tool={{ name: 'wipe', exposed: false }}
        slotControl={<input type="checkbox" aria-label="Enable wipe" />}
      />
    );

    expect(screen.getByText('Blocked')).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Enable wipe' })).toBeInTheDocument();
  });
});
