// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deliberately without the flag: this file covers the default, where the platform
// refuses container deployments and the form must not offer them.
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { CreateDeploymentModal } from '@studio/routes/agents/AgentDeploymentsListRoute/CreateDeploymentModal';
import { renderRoute, screen, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const workspace = workspace1.workspace;

describe('CreateDeploymentModal without container deployments', () => {
  it('offers no runtime the platform would reject', async () => {
    const user = userEvent.setup();
    renderRoute(
      <CreateDeploymentModal open onClose={vi.fn()} workspace={workspace} agent="an-agent" />
    );

    const dialog = await screen.findByRole('dialog', { name: 'Deploy Agent' });
    await user.click(within(dialog).getByRole('combobox', { name: 'Runtime' }));

    expect(await screen.findByRole('option', { name: 'Subprocess' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Docker' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Kubernetes' })).not.toBeInTheDocument();
  });

  it('does not ask for a container image', async () => {
    renderRoute(
      <CreateDeploymentModal open onClose={vi.fn()} workspace={workspace} agent="an-agent" />
    );

    await screen.findByRole('dialog', { name: 'Deploy Agent' });

    expect(screen.queryByRole('textbox', { name: 'Container Image' })).not.toBeInTheDocument();
  });
});
