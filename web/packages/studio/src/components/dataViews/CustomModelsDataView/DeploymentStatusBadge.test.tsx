// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import { render, screen } from '@testing-library/react';

describe('DeploymentStatusBadge', () => {
  it('reads "Deployed" for a ready base model', () => {
    render(
      <DeploymentStatusBadge state={{ kind: 'served', hasDeployment: true, status: 'READY' }} />
    );
    expect(screen.getByText('Deployed')).toBeVisible();
  });

  it('reads "Served" for a loaded adapter, not "Deployed"', () => {
    render(
      <DeploymentStatusBadge
        state={{ kind: 'served', hasDeployment: true, status: 'READY' }}
        isAdapter
      />
    );
    expect(screen.getByText('Served')).toBeVisible();
    expect(screen.queryByText('Deployed')).not.toBeInTheDocument();
  });

  it('reads "Not served" when the base is up but the adapter is not loaded', () => {
    render(<DeploymentStatusBadge state={{ kind: 'adapter-not-loaded' }} isAdapter />);
    expect(screen.getByText('Not served')).toBeVisible();
  });

  it('reads "Not deployed" when nothing serves it', () => {
    render(<DeploymentStatusBadge state={{ kind: 'not-deployed' }} />);
    expect(screen.getByText('Not deployed')).toBeVisible();
  });

  it('reads "Deploying" while a deployment is coming up', () => {
    render(
      <DeploymentStatusBadge state={{ kind: 'served', hasDeployment: true, status: 'PENDING' }} />
    );
    expect(screen.getByText('Deploying')).toBeVisible();
  });

  it('reads "Failed" for a deployment in error', () => {
    render(
      <DeploymentStatusBadge state={{ kind: 'served', hasDeployment: true, status: 'ERROR' }} />
    );
    expect(screen.getByText('Failed')).toBeVisible();
  });

  it('reads "Unavailable" for a deployment that is gone', () => {
    render(
      <DeploymentStatusBadge state={{ kind: 'served', hasDeployment: true, status: 'LOST' }} />
    );
    expect(screen.getByText('Unavailable')).toBeVisible();
  });

  it('reads "Available" when served by a provider with no deployment', () => {
    render(
      <DeploymentStatusBadge
        state={{ kind: 'served', hasDeployment: false, providerRef: 'ws/build' }}
      />
    );
    expect(screen.getByText('Available')).toBeVisible();
  });

  it('reads "Unknown", not "Available", when a deployment is expected but unreadable', () => {
    render(
      <DeploymentStatusBadge state={{ kind: 'served', hasDeployment: true, providerRef: 'ws/p' }} />
    );
    expect(screen.getByText('Unknown')).toBeVisible();
    expect(screen.queryByText('Available')).not.toBeInTheDocument();
  });

  it('reads "Unknown", not "Not deployed", when providers could not be fetched', () => {
    render(<DeploymentStatusBadge state={{ kind: 'unknown' }} />);
    expect(screen.getByText('Unknown')).toBeVisible();
    expect(screen.queryByText('Not deployed')).not.toBeInTheDocument();
  });

  it('shows no status text while the row is still resolving', () => {
    render(<DeploymentStatusBadge state={undefined} />);
    for (const label of ['Deployed', 'Served', 'Not deployed', 'Not served', 'Unknown']) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
  });
});
