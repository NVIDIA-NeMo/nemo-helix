// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { AgentConfigPanels } from '@studio/routes/agents/AgentDetailRoute/AgentConfigPanels';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const baseAgent: Agent = {
  name: 'react-agent',
  workspace: 'default',
  config: {},
  id: 'react-agent',
  created_at: '2026-04-20T10:00:00Z',
  created_by: null,
  updated_at: '2026-04-20T10:00:00Z',
  updated_by: null,
  entity_id: 'react-agent',
  parent: '',
  db_version: 1,
};

describe('AgentConfigPanels', () => {
  it('collapses Additional configuration until its header is clicked', async () => {
    const user = userEvent.setup();
    render(
      <AgentConfigPanels agent={{ ...baseAgent, config: { general: { telemetry: 'enabled' } } }} />
    );

    expect(screen.getByText('enabled')).not.toBeVisible();

    await user.click(screen.getByText('Additional configuration'));

    expect(screen.getByText('enabled')).toBeVisible();
  });

  it('says when the agent has no stored configuration', () => {
    render(<AgentConfigPanels agent={baseAgent} />);

    expect(screen.getByText('This agent has no stored configuration.')).toBeInTheDocument();
  });
});
