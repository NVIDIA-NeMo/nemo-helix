// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deliberately without the flag: this file covers the default, where the platform
// refuses container deployments and the form must not offer them.
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { CreateDeploymentModal } from '@studio/routes/agents/AgentDeploymentsListRoute/CreateDeploymentModal';
import { renderRoute, screen, within } from '@studio/tests/util/render';

const workspace = workspace1.workspace;

describe('CreateDeploymentModal without container deployments', () => {
  it('offers no runtime the platform would reject', async () => {
    renderRoute(
      <CreateDeploymentModal open onClose={vi.fn()} workspace={workspace} agent="an-agent" />
    );

    const dialog = await screen.findByRole('dialog', { name: 'Deploy Agent' });

    // Subprocess is the only possibility, so there is nothing to choose and nothing
    // advanced to disclose.
    expect(within(dialog).queryByRole('combobox', { name: 'Runtime' })).not.toBeInTheDocument();
    expect(within(dialog).queryByText(/Advanced/)).not.toBeInTheDocument();
  });

  it('does not ask for a container image', async () => {
    renderRoute(
      <CreateDeploymentModal open onClose={vi.fn()} workspace={workspace} agent="an-agent" />
    );

    await screen.findByRole('dialog', { name: 'Deploy Agent' });

    expect(screen.queryByRole('textbox', { name: 'Container Image' })).not.toBeInTheDocument();
  });
});
