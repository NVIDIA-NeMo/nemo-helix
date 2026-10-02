// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { workspace1 } from '@studio/mocks/entity-store/projects';
import { PackageAgentControl } from '@studio/routes/agents/AgentDetailRoute/PackageAgentControl';
import {
  mockEnabledModes,
  mockExecutionProfiles,
} from '@studio/tests/util/mockAgentDeploymentCapabilities';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

describe('PackageAgentControl on a Kubernetes platform', () => {
  it('reads the platform capabilities and offers the local Kubernetes build', async () => {
    mockExecutionProfiles([{ profile: 'default', backend: 'kubernetes_job' }]);
    mockEnabledModes('subprocess', 'k8s');
    renderRoute(
      <PackageAgentControl workspace={workspace1.workspace} agentName="my-agent" canPackage />
    );

    await userEvent.setup().click(screen.getByRole('button', { name: 'Build image' }));
    const dialog = screen.getByRole('dialog');

    await waitFor(() => expect(dialog).toHaveTextContent(/--mode k8s --image IMAGE/));
    expect(within(dialog).getByRole('button', { name: 'Build image' })).toBeDisabled();
    expect(dialog).toHaveTextContent(/to deploy it with Kubernetes\./);
  });
});
