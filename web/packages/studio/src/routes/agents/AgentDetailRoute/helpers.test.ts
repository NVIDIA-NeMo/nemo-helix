// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { deploymentUrl } from '@studio/routes/agents/AgentDetailRoute/helpers';

const dep = (over: Partial<AgentDeployment>) =>
  ({ workspace: 'default', ...over }) as AgentDeployment;

describe('deploymentUrl', () => {
  it('shows the scalar endpoint for subprocess deployments', () => {
    expect(deploymentUrl(dep({ endpoint: 'http://127.0.0.1:49152', endpoints: [] }))).toBe(
      'http://127.0.0.1:49152'
    );
  });

  it('falls back to the first http(s) endpoint, skipping other protocols and empty urls', () => {
    expect(
      deploymentUrl(
        dep({
          endpoint: '',
          endpoints: [
            { name: 'grpc', url: 'grpc://x:1', protocol: 'grpc' },
            { name: 'blank', url: '', protocol: 'http' },
            { name: 'a', url: 'http://localhost:49154', protocol: 'http' },
            { name: 'b', url: 'https://other', protocol: 'https' },
          ],
        })
      )
    ).toBe('http://localhost:49154');
  });

  it('prefers a non-empty endpoint over endpoints', () => {
    expect(
      deploymentUrl(
        dep({
          endpoint: 'http://a',
          endpoints: [{ name: 'p', url: 'http://b', protocol: 'http' }],
        })
      )
    ).toBe('http://a');
  });

  it('is undefined when nothing is routable', () => {
    expect(
      deploymentUrl(dep({ endpoints: [{ name: 't', url: 'tcp://x', protocol: 'tcp' }] }))
    ).toBeUndefined();
  });
});
