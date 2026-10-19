// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LINK_GITHUB_ISSUES } from '@studio/constants/links';
import { ROUTES } from '@studio/constants/routes';
import { WorkspaceDashboardRoute } from '@studio/routes/WorkspaceDashboardRoute';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import { createMemoryRouter, generatePath, RouterProvider } from 'react-router';

const TEST_WORKSPACE = 'test-workspace';

// Mock the ReportTraceModal to avoid complex modal rendering in tests
vi.mock('@studio/components/ReportTraceModal', () => ({
  ReportTraceModal: vi.fn(() => null),
}));

vi.mock('@studio/constants/environment', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/environment')>();
  return { ...actual, CUSTOMIZER_ENABLED: true };
});

const renderRoute = (initialPath?: string) => {
  const dashboardPath = generatePath(ROUTES.workspace.dashboard, { workspace: TEST_WORKSPACE });

  const router = createMemoryRouter(
    [
      {
        path: ROUTES.workspace.dashboard,
        element: <WorkspaceDashboardRoute />,
      },
      {
        path: '*',
        element: <LocationDisplay />,
      },
    ],
    {
      initialEntries: [initialPath ?? dashboardPath],
    }
  );

  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('WorkspaceDashboardRoute', () => {
  describe('page rendering', () => {
    it('should display the welcome header', async () => {
      renderRoute();
      expect(await screen.findByText('Welcome to NeMo Studio')).toBeInTheDocument();
      expect(
        screen.getByText(
          'Fine-tune and evaluate models, generate synthetic data, and monitor NeMo Helix jobs.'
        )
      ).toBeInTheDocument();
    });

    it('should display the Get Started section title', async () => {
      renderRoute();
      expect(await screen.findByText('Get Started')).toBeInTheDocument();
    });

    it('should display the Resources section title', async () => {
      renderRoute();
      expect(await screen.findByText('Resources')).toBeInTheDocument();
    });
  });

  describe('dashboard cards', () => {
    it.each([
      {
        title: 'Chat with a Model',
        description: 'Chat with base models and explore capabilities.',
        actionButtonName: 'Chat' as string | undefined,
      },
    ])(
      'should render the $title card and conditionally display the action button',
      async ({ title, description, actionButtonName }) => {
        renderRoute();
        expect(await screen.findByText(title)).toBeInTheDocument();
        expect(screen.getByText(description)).toBeInTheDocument();
        if (actionButtonName) {
          expect(await screen.findByRole('button', { name: actionButtonName })).toBeInTheDocument();
        }
      }
    );

    it('should display docs links', async () => {
      renderRoute();
      const docsLinks = await screen.findAllByRole('link', { name: 'Docs' });
      expect(docsLinks.length).toBeGreaterThan(0);

      docsLinks.forEach((link) => {
        expect(link).toHaveAttribute('target', '_blank');
        expect(link).toHaveAttribute('rel', 'noopener noreferrer');
      });
    });
  });

  describe('resources section', () => {
    it('should display documentation links', async () => {
      renderRoute();
      expect(await screen.findByText('Documentation')).toBeInTheDocument();
      expect(screen.getByText('Studio Documentation')).toBeInTheDocument();
      expect(screen.getByText('SDK Documentation')).toBeInTheDocument();
    });

    it('should display help and support section', async () => {
      renderRoute();
      expect(await screen.findByText('Help & Support')).toBeInTheDocument();
    });

    it('should link Report a Bug to GitHub issues', async () => {
      renderRoute();

      const reportBugLink = await screen.findByRole('link', { name: 'Report a Bug' });
      expect(reportBugLink).toHaveAttribute('href', LINK_GITHUB_ISSUES);
      expect(reportBugLink).toHaveAttribute('target', '_blank');
      expect(reportBugLink).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });
});
