// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

// Mock the DataView component with a minimal implementation
vi.mock('@studio/components/dataViews/SafeSynthesizerJobsDataView', () => ({
  GenerateJobsDataView: () => <div data-testid="safe-synthesizer-data-view">DataView</div>,
}));
vi.mock('@studio/providers/breadcrumbs/useBreadcrumbs', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@studio/providers/breadcrumbs/useBreadcrumbs')>();
  return {
    ...actual,
    useBreadcrumbs: vi.fn(),
  };
});
describe('SafeSynthesizerListRoute', () => {
  beforeEach(() => {
    vi.resetModules();
    mockUseParams({
      workspace: 'test-workspace',
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('should render the Safe Synthesizer list page', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', PLATFORM_BASE_URL);

    const { SafeSynthesizerListRoute } = await import('./index');

    render(
      <TestProviders>
        <RouterProvider
          router={createMemoryRouter(
            [{ path: ROUTES.workspace.safeSynthesizer, element: <SafeSynthesizerListRoute /> }],
            { initialEntries: ['/workspaces/test-workspace/safe-synthesizer'] }
          )}
        />
      </TestProviders>
    );

    expect(screen.getByText('Safe Synthesizer')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /synthesize data/i })).toBeInTheDocument();
  });
});
