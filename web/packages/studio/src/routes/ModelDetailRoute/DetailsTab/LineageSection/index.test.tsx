// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  LineageSection,
  type LineageSectionProps,
} from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

const renderLineage = (props: Partial<LineageSectionProps> = {}) =>
  render(
    <MemoryRouter>
      <LineageSection
        base={{ name: 'base-model', note: 'Checkpoint it was fine-tuned from' }}
        job={{ jobName: 'job-1', isLoading: false, to: '/training' }}
        fileset={{ workspace: 'ws', name: 'weights' }}
        deployment={{ kind: 'not-deployed' }}
        isAdapter={false}
        {...props}
      />
    </MemoryRouter>
  );

describe('LineageSection', () => {
  it('links the job card to the training tab', () => {
    renderLineage();

    expect(screen.getByRole('link', { name: /Training Job/ })).toHaveAttribute('href', '/training');
  });

  it('hides the base model and job cards when they are omitted', () => {
    renderLineage({ base: undefined, job: undefined });

    expect(screen.queryByText('Base Model')).not.toBeInTheDocument();
    expect(screen.queryByText('Training Job')).not.toBeInTheDocument();
    expect(screen.getByText('Files')).toBeInTheDocument();
  });

  it('links a served model to its deployment', () => {
    renderLineage({
      deployment: { kind: 'served', hasDeployment: true, deploymentRef: 'ws/my-deployment' },
    });

    expect(screen.getByRole('link', { name: /my-deployment/ })).toHaveAttribute(
      'href',
      expect.stringContaining('/deployments/my-deployment')
    );
  });

  it('links the deployment card to the given route, even when nothing serves the model', () => {
    renderLineage({ deploymentTo: '/deployment-tab' });

    expect(screen.getByRole('link', { name: /Deployment/ })).toHaveAttribute(
      'href',
      '/deployment-tab'
    );
  });

  it('points an adapter at its base model when nothing serves it', () => {
    renderLineage({ deployment: { kind: 'not-deployed' }, isAdapter: true });

    expect(screen.getByText('Deploy the base model to serve this adapter')).toBeInTheDocument();
  });

  it('explains when an adapter is not loaded', () => {
    renderLineage({ deployment: { kind: 'adapter-not-loaded' }, isAdapter: true });

    expect(screen.getByText('Not loaded')).toBeInTheDocument();
  });

  it('shows when no job produced the model', () => {
    renderLineage({ job: { jobName: undefined, isLoading: false, to: '/t' } });

    expect(screen.getByText('Not found')).toBeInTheDocument();
  });
});
