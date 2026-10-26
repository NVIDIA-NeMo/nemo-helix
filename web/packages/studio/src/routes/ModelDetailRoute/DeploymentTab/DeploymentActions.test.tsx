// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import {
  type ModelDeployment,
  ModelDeploymentStatus,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import { useDeleteDeploymentAndConfig } from '@studio/routes/DeploymentsListRoute/useDeleteDeploymentAndConfig';
import { DeploymentActions } from '@studio/routes/ModelDetailRoute/DeploymentTab/DeploymentActions';
import { render, screen, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

vi.mock('@nemo/sdk/generated/platform/model-deployments');
vi.mock('@studio/routes/DeploymentsListRoute/useDeleteDeploymentAndConfig');

const model = { name: 'my-model', workspace: 'ws' } as ModelEntity;
const deployment = {
  name: 'my-deployment',
  workspace: 'ws',
  status: ModelDeploymentStatus.READY,
} as ModelDeployment;

const renderActions = (state: DeploymentIndicatorState) =>
  render(<DeploymentActions model={model} adapter={null} deployment={state} />);

describe('DeploymentActions', () => {
  const deleteDeploymentAndConfig = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.mocked(useDeleteDeploymentAndConfig).mockReturnValue({
      deleteDeploymentAndConfig,
      isDeleting: false,
    });
    vi.mocked(useModelsGetLatestDeployment).mockReturnValue({
      data: deployment,
    } as unknown as ReturnType<typeof useModelsGetLatestDeployment>);
  });

  it('offers chat and delete for a served model', async () => {
    renderActions({ kind: 'served', hasDeployment: true, deploymentRef: 'ws/my-deployment' });

    expect(screen.getByRole('link', { name: 'Chat' })).toHaveAttribute(
      'href',
      expect.stringContaining('tab=chat')
    );

    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    const dialog = await screen.findByRole('dialog');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Delete' }));

    expect(deleteDeploymentAndConfig).toHaveBeenCalledWith(deployment);
  });

  it('only offers chat when an external provider serves the model', () => {
    vi.mocked(useModelsGetLatestDeployment).mockReturnValue({
      data: undefined,
    } as unknown as ReturnType<typeof useModelsGetLatestDeployment>);
    renderActions({ kind: 'served', hasDeployment: false, providerRef: 'default/build' });

    expect(screen.getByRole('link', { name: 'Chat' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Delete' })).not.toBeInTheDocument();
  });

  it('offers to delete a deployment that is still starting', () => {
    renderActions({
      kind: 'not-serving',
      deployment: { ...deployment, status: ModelDeploymentStatus.PENDING },
    });

    expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Redeploy…' })).not.toBeInTheDocument();
  });

  it('offers to delete or redeploy a failed deployment', () => {
    renderActions({
      kind: 'not-serving',
      deployment: { ...deployment, status: ModelDeploymentStatus.ERROR },
    });

    expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Redeploy…' })).toBeInTheDocument();
  });

  it('offers to deploy a model nothing serves', () => {
    renderActions({ kind: 'not-deployed' });

    expect(screen.getByRole('button', { name: 'Deploy' })).toBeInTheDocument();
  });
});
