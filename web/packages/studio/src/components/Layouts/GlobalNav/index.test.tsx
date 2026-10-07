// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { SIDE_NAV_OPEN_KEY } from '@studio/util/localStorage';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, generatePath, MemoryRouter, RouterProvider } from 'react-router';

vi.mock('@studio/components/Breadcrumbs', () => ({
  Breadcrumbs: () => <div data-testid="breadcrumbs" />,
}));

vi.mock('@studio/components/UserPopover', () => ({
  UserPopover: () => <div data-testid="user-popover" />,
}));

vi.mock('@studio/components/Layouts/GlobalNav/components/ThemeSwitch', () => ({
  ThemeSwitch: () => <div data-testid="theme-switch" />,
}));

vi.mock('@studio/routes/agents/AssistantChatRoute/AssistantTopBarChat', () => ({
  AssistantTopBarChat: () => <div data-testid="assistant-top-bar-chat" />,
}));

vi.mock('@studio/components/Layouts/GlobalNav/components/DocumentationLink', () => ({
  DocumentationLink: () => <div data-testid="documentation-link" />,
}));

vi.mock('@studio/components/WelcomeTour', () => ({
  WelcomeTour: () => <div data-testid="welcome-tour" />,
}));

type ChangeListener = (e: { matches: boolean }) => void;

const createMatchMediaMock = (initialMatches: boolean) => {
  let listener: ChangeListener | undefined;

  const mql = {
    matches: initialMatches,
    media: '(min-width: 768px)',
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn((_event: string, cb: ChangeListener) => {
      listener = cb;
    }),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  };

  const matchMediaFn = vi.fn().mockReturnValue(mql);
  window.matchMedia = matchMediaFn;

  const fireChange = async (matches: boolean) => {
    await act(async () => {
      listener?.({ matches });
    });
  };

  return { mql, matchMediaFn, fireChange };
};

const renderGlobalNav = async (initialPath = '/workspaces/test-workspace/jobs') => {
  const { GlobalNav } = await import('@studio/components/Layouts/GlobalNav/index');
  render(
    <MemoryRouter initialEntries={[initialPath]}>
      <GlobalNav sideNav={() => <div data-testid="side-nav">Side Nav Content</div>} />
    </MemoryRouter>
  );
  // Wait for Suspense / lazy components to settle before the test proceeds
  await screen.findAllByRole('button', { name: /(Collapse|Expand) sidebar/i });
};

describe('GlobalNav', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.style.setProperty('--breakpoint-md', '768px');
    mockUseParams({ workspace: 'test-workspace' });
  });

  it('renders the documentation link', async () => {
    createMatchMediaMock(true);

    await renderGlobalNav();

    expect(screen.getByTestId('documentation-link')).toBeInTheDocument();
  });

  describe('Responsive auto-collapse', () => {
    it('stays expanded when initial viewport is wide', async () => {
      createMatchMediaMock(true);
      await renderGlobalNav();

      expect(screen.getByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();
      expect(screen.getByText('NeMo Studio')).toBeInTheDocument();
    });

    it('starts collapsed on the NeMo Assistant route when no preference is saved', async () => {
      createMatchMediaMock(true);

      await renderGlobalNav(
        generatePath(ROUTES.workspace.assistantChat, { workspace: 'test-workspace' })
      );

      expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();
      expect(screen.queryByText('NeMo Studio')).not.toBeInTheDocument();
    });

    it('respects a saved expanded preference on the NeMo Assistant route', async () => {
      localStorage.setItem(SIDE_NAV_OPEN_KEY, JSON.stringify('true'));
      createMatchMediaMock(true);

      await renderGlobalNav(
        generatePath(ROUTES.workspace.assistantChat, { workspace: 'test-workspace' })
      );

      expect(screen.getByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();
      expect(screen.getByText('NeMo Studio')).toBeInTheDocument();
    });

    it('uses the NeMo Assistant default during in-app navigation', async () => {
      createMatchMediaMock(true);
      const { GlobalNav } = await import('@studio/components/Layouts/GlobalNav/index');
      const router = createMemoryRouter(
        [
          {
            path: '*',
            element: (
              <GlobalNav sideNav={() => <div data-testid="side-nav">Side Nav Content</div>} />
            ),
          },
        ],
        { initialEntries: ['/workspaces/test-workspace/jobs'] }
      );
      render(<RouterProvider router={router} />);

      expect(screen.getByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();

      await act(async () => {
        await router.navigate(
          generatePath(ROUTES.workspace.assistantChat, { workspace: 'test-workspace' })
        );
      });

      expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();

      await act(async () => {
        await router.navigate('/workspaces/test-workspace/jobs');
      });

      expect(screen.getByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();
    });

    it('auto-collapses sidebar when initial viewport is narrow', async () => {
      createMatchMediaMock(false);
      await renderGlobalNav();

      expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();
      expect(screen.queryByText('NeMo Studio')).not.toBeInTheDocument();
    });

    it('collapses on transition from wide to narrow', async () => {
      const { fireChange } = createMatchMediaMock(true);
      await renderGlobalNav();

      expect(screen.getByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();

      await fireChange(false);

      expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();
      expect(screen.queryByText('NeMo Studio')).not.toBeInTheDocument();
    });

    it('does not re-collapse when already narrow', async () => {
      const { fireChange } = createMatchMediaMock(false);
      await renderGlobalNav();

      const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');
      const callsBefore = setItemSpy.mock.calls.filter(([key]) => key === SIDE_NAV_OPEN_KEY).length;

      await fireChange(false);

      const callsAfter = setItemSpy.mock.calls.filter(([key]) => key === SIDE_NAV_OPEN_KEY).length;
      expect(callsAfter).toBe(callsBefore);

      setItemSpy.mockRestore();
    });

    it('re-arms collapse after returning to wide then going narrow again', async () => {
      const { fireChange } = createMatchMediaMock(false);
      await renderGlobalNav();

      expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();

      // Manually expand, then go wide to re-arm wasWideRef
      const user = userEvent.setup();
      await user.click(screen.getByRole('button', { name: 'Expand sidebar' }));
      expect(screen.getByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();

      await fireChange(true);

      // Going narrow again should auto-collapse
      await fireChange(false);
      expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();
    });
  });

  describe('Manual toggle', () => {
    it('toggles between expanded and collapsed on click', async () => {
      createMatchMediaMock(true);
      await renderGlobalNav();
      const user = userEvent.setup();

      expect(screen.getByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();
      expect(screen.getByText('NeMo Studio')).toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: 'Collapse sidebar' }));

      expect(screen.queryByText('NeMo Studio')).not.toBeInTheDocument();

      await user.click(await screen.findByRole('button', { name: 'Expand sidebar' }));

      expect(await screen.findByRole('button', { name: 'Collapse sidebar' })).toBeInTheDocument();
      expect(screen.getByText('NeMo Studio')).toBeInTheDocument();
    });
  });

  describe('collapsed prop', () => {
    it('passes collapsed=false to sideNav when expanded', async () => {
      createMatchMediaMock(true);
      const { GlobalNav } = await import('@studio/components/Layouts/GlobalNav/index');
      const sideNavSpy = vi.fn(() => <div data-testid="side-nav" />);
      render(
        <MemoryRouter>
          <GlobalNav sideNav={sideNavSpy} />
        </MemoryRouter>
      );

      expect(sideNavSpy).toHaveBeenCalledWith(false);
    });
  });

  describe('NeMo Assistant top bar chat', () => {
    it('mounts outside the dashboard and full NeMo Assistant routes', async () => {
      createMatchMediaMock(true);

      await renderGlobalNav('/workspaces/test-workspace/jobs');

      expect(screen.getByTestId('assistant-top-bar-chat')).toBeInTheDocument();
    });

    it('does not mount on the dashboard route', async () => {
      createMatchMediaMock(true);

      await renderGlobalNav(
        generatePath(ROUTES.workspace.dashboard, { workspace: 'test-workspace' })
      );

      expect(screen.queryByTestId('assistant-top-bar-chat')).not.toBeInTheDocument();
    });

    it('does not mount on the full NeMo Assistant route', async () => {
      createMatchMediaMock(true);

      await renderGlobalNav(
        generatePath(ROUTES.workspace.assistantChat, { workspace: 'test-workspace' })
      );

      expect(screen.queryByTestId('assistant-top-bar-chat')).not.toBeInTheDocument();
    });
  });
});
