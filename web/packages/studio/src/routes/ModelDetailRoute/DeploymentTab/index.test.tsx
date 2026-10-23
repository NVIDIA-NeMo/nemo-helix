// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type Adapter,
  type ModelDeployment,
  ModelDeploymentStatus,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import { DeploymentTab } from '@studio/routes/ModelDetailRoute/DeploymentTab';
import { render, screen } from '@studio/tests/util/render';

vi.mock('@studio/routes/ModelDetailRoute/DeploymentTab/ServedView', () => ({
  ServedView: ({ deploymentRef }: { deploymentRef?: string }) => (
    <div>served by {deploymentRef ?? 'external'}</div>
  ),
}));

const model = { name: 'my-model', workspace: 'ws' } as ModelEntity;
const deployment = {
  name: 'my-deployment',
  workspace: 'ws',
  status: ModelDeploymentStatus.ERROR,
  status_message: 'CUDA out of memory',
} as ModelDeployment;

const renderTab = (state: DeploymentIndicatorState, adapter: Adapter | null = null) =>
  render(<DeploymentTab model={model} adapter={adapter} deployment={state} />);

describe('DeploymentTab', () => {
  it('says when nothing serves the model, leaving the action to the page header', () => {
    renderTab({ kind: 'not-deployed' });

    expect(screen.getByText('Not deployed')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('points an adapter at its base model when nothing serves it', () => {
    renderTab({ kind: 'not-deployed' }, { name: 'my-adapter', workspace: 'ws' } as Adapter);

    expect(screen.getByText('Base model not deployed')).toBeInTheDocument();
    expect(screen.getByText('Deploy my-model to serve this adapter.')).toBeInTheDocument();
  });

  it('shows the serving deployment', () => {
    renderTab({ kind: 'served', hasDeployment: true, deploymentRef: 'ws/my-deployment' });

    expect(screen.getByText('served by ws/my-deployment')).toBeInTheDocument();
  });

  it('shows a deployment that is still starting', () => {
    renderTab({
      kind: 'not-serving',
      deployment: {
        ...deployment,
        status: ModelDeploymentStatus.PENDING,
        status_message: 'Pulling weights',
      },
    });

    expect(screen.getByRole('link', { name: /my-deployment/ })).toBeInTheDocument();
    expect(screen.getByText('Pulling weights')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('explains a failed deployment', () => {
    renderTab({ kind: 'not-serving', deployment });

    expect(screen.getByText('CUDA out of memory')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /my-deployment/ })).toBeInTheDocument();
  });
});
