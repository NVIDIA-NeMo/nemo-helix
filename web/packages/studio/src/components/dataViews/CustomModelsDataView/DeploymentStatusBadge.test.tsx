// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import { useModelDeploymentIndicator } from '@studio/hooks/useModelDeploymentIndicator';
import { render, screen } from '@testing-library/react';

vi.mock('@studio/hooks/useModelDeploymentIndicator', () => ({
  useModelDeploymentIndicator: vi.fn(),
}));

const mockedIndicator = vi.mocked(useModelDeploymentIndicator);
const model = { id: 'm', name: 'base-model', workspace: 'ws' } as ModelEntity;
const adapter = { name: 'my-adapter', workspace: 'ws' } as Adapter;

describe('DeploymentStatusBadge', () => {
  beforeEach(() => vi.clearAllMocks());

  it('reads "Deployed" for a ready base model', () => {
    mockedIndicator.mockReturnValue({ kind: 'served', hasDeployment: true, status: 'READY' });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Deployed')).toBeVisible();
  });

  it('reads "Served" for a loaded adapter, not "Deployed"', () => {
    mockedIndicator.mockReturnValue({ kind: 'served', hasDeployment: true, status: 'READY' });
    render(<DeploymentStatusBadge model={model} adapter={adapter} />);
    expect(screen.getByText('Served')).toBeVisible();
    expect(screen.queryByText('Deployed')).not.toBeInTheDocument();
  });

  it('reads "Not served" when the base is up but the adapter is not loaded', () => {
    mockedIndicator.mockReturnValue({ kind: 'adapter-not-loaded' });
    render(<DeploymentStatusBadge model={model} adapter={adapter} />);
    expect(screen.getByText('Not served')).toBeVisible();
  });

  it('reads "Not deployed" when nothing serves it', () => {
    mockedIndicator.mockReturnValue({ kind: 'not-deployed' });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Not deployed')).toBeVisible();
  });

  it('reads "Deploying" while a deployment is coming up', () => {
    mockedIndicator.mockReturnValue({ kind: 'served', hasDeployment: true, status: 'PENDING' });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Deploying')).toBeVisible();
  });

  it('reads "Failed" for a deployment in error', () => {
    mockedIndicator.mockReturnValue({ kind: 'served', hasDeployment: true, status: 'ERROR' });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Failed')).toBeVisible();
  });

  it('reads "Unavailable" for a deployment that is gone', () => {
    mockedIndicator.mockReturnValue({ kind: 'served', hasDeployment: true, status: 'LOST' });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Unavailable')).toBeVisible();
  });

  it('reads "Available" when served by a provider with no deployment', () => {
    mockedIndicator.mockReturnValue({
      kind: 'served',
      hasDeployment: false,
      providerRef: 'ws/build',
    });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Available')).toBeVisible();
  });

  it('reads "Unknown", not "Available", when a deployment is expected but unreadable', () => {
    mockedIndicator.mockReturnValue({ kind: 'served', hasDeployment: true, providerRef: 'ws/p' });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Unknown')).toBeVisible();
    expect(screen.queryByText('Available')).not.toBeInTheDocument();
  });

  it('reads "Unknown", not "Not deployed", when providers could not be fetched', () => {
    mockedIndicator.mockReturnValue({ kind: 'unknown' });
    render(<DeploymentStatusBadge model={model} />);
    expect(screen.getByText('Unknown')).toBeVisible();
    expect(screen.queryByText('Not deployed')).not.toBeInTheDocument();
  });

  it('shows no status text while resolving', () => {
    mockedIndicator.mockReturnValue({ kind: 'loading' });
    render(<DeploymentStatusBadge model={model} />);
    for (const label of ['Deployed', 'Served', 'Not deployed', 'Not served', 'Unknown']) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
  });
});
