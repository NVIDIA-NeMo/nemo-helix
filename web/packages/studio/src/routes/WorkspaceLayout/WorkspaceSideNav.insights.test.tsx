// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceSideNav } from '@studio/routes/WorkspaceLayout/WorkspaceSideNav';
import { getOptimizerRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

describe('WorkspaceSideNav with Optimizer enabled', () => {
  it('lists Insights and Traces under Agent Development after Experiments', () => {
    const element = <WorkspaceSideNav />;
    renderRoute(element, {
      history: '/workspaces/test-workspace/dashboard',
      routes: [{ path: '/workspaces/:workspace/*', element }],
    });

    const insights = screen.getByRole('link', { name: 'Insights' });
    expect(insights).toHaveAttribute('href', getOptimizerRoute('test-workspace'));

    const ordered = [
      screen.getByText('Agent Development'),
      screen.getByRole('link', { name: 'Experiments' }),
      insights,
      screen.getByRole('link', { name: 'Traces' }),
      screen.getByText('Models'),
    ];
    ordered.slice(1).forEach((node, i) => {
      expect(ordered[i].compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });
  });
});
