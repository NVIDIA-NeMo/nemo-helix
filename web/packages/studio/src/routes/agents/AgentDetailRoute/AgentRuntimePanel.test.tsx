// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema';
import { AgentRuntimePanel } from '@studio/routes/agents/AgentDetailRoute/AgentRuntimePanel';
import { render, screen } from '@testing-library/react';

describe('AgentRuntimePanel', () => {
  it('summarizes the selected deployment and its models', () => {
    render(
      <AgentRuntimePanel
        deployment={
          {
            name: 'weather-dep',
            status: 'failed',
            deployment_mode: 'docker',
            spec_revision: '0123456789abcdef0123456789abcdef01234567',
            error: 'Health check timed out after 120s.',
          } as AgentDeployment
        }
        modelNames={['nvidia/nemotron']}
      />
    );

    expect(screen.getByRole('complementary', { name: 'Runtime' })).toBeInTheDocument();
    expect(screen.getByText('weather-dep')).toBeInTheDocument();
    expect(screen.getByText('docker')).toBeInTheDocument();
    expect(screen.getByText('0123456')).toBeInTheDocument();
    expect(screen.getByText('Health check timed out after 120s.')).toBeInTheDocument();
    expect(screen.getByText('nvidia/nemotron')).toBeInTheDocument();
  });

  it('says when no deployment is selected', () => {
    render(<AgentRuntimePanel modelNames={[]} />);

    expect(screen.getByText('No deployment selected.')).toBeInTheDocument();
  });
});
