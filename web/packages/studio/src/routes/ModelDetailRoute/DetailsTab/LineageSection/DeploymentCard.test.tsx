// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DeploymentCard,
  type DeploymentCardProps,
} from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection/DeploymentCard';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

const renderCard = (props: Partial<DeploymentCardProps> = {}) =>
  render(
    <MemoryRouter>
      <DeploymentCard deployment={{ kind: 'not-deployed' }} isAdapter={false} {...props} />
    </MemoryRouter>
  );

describe('DeploymentCard', () => {
  it('links a served model to its deployment', () => {
    renderCard({
      deployment: { kind: 'served', hasDeployment: true, deploymentRef: 'ws/my-deployment' },
    });

    expect(screen.getByRole('link', { name: /my-deployment/ })).toHaveAttribute(
      'href',
      expect.stringContaining('/deployments/my-deployment')
    );
  });

  it('links the deployment card to the given route, even when nothing serves the model', () => {
    renderCard({ to: '/deployment-tab' });

    expect(screen.getByRole('link', { name: /Deployment/ })).toHaveAttribute(
      'href',
      '/deployment-tab'
    );
  });

  it('points an adapter at its base model when nothing serves it', () => {
    renderCard({ deployment: { kind: 'not-deployed' }, isAdapter: true });

    expect(screen.getByText('Deploy the base model to serve this adapter')).toBeInTheDocument();
  });

  it('explains when an adapter is not loaded', () => {
    renderCard({ deployment: { kind: 'adapter-not-loaded' }, isAdapter: true });

    expect(screen.getByText('Not loaded')).toBeInTheDocument();
  });

  it('describes an unknown deployment without an Unknown badge', () => {
    renderCard({ deployment: { kind: 'unknown' } });

    expect(screen.getByText("Couldn't check where this is served")).toBeInTheDocument();
    expect(screen.getAllByText('Unknown')).toHaveLength(1);
  });
});
