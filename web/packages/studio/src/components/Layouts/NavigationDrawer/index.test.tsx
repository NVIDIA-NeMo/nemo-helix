// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ACTIVE_NAV_ITEM_CLASS } from '@studio/components/Layouts/NavigationDrawer/styles';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { PageLayout } from '@studio/routes/PageLayout';
import { getWorkspaceIndexRoute } from '@studio/routes/utils';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { NAV_ACCORDION_STATE_KEY, SIDE_NAV_OPEN_KEY } from '@studio/util/localStorage';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider, useLocation } from 'react-router';

const mockItems = [
  { id: 'projects', slotLabel: 'Projects', icon: 'ProjectsIcon', href: ROUTES.workspace.index },
  {
    id: 'customizations',
    slotLabel: 'Customizations',
    icon: 'CustomizationsIcon',
    href: ROUTES.workspace.customizationJobList,
  },
  {
    group: 'Evaluate',
    items: [
      {
        id: 'traces',
        slotLabel: 'Traces',
        subItems: [
          {
            id: 'entries',
            slotLabel: 'Entries',
          },
          {
            id: 'export-jobs',
            slotLabel: 'Export Jobs',
          },
        ],
      },
    ],
  },
];

const renderWithProjectRoute = (element: React.ReactElement) => {
  const router = createMemoryRouter([{ path: ROUTES.workspace.index, element }], {
    initialEntries: [getWorkspaceIndexRoute(workspace1.workspace)],
  });
  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

/** Awaited so the dynamic import doesn't fire act warnings. */
const importNavigationDrawer = async () => {
  let NavigationDrawer;
  await act(async () => {
    const { NavigationDrawer: NavigationDrawerComponent } =
      await import('@studio/components/Layouts/NavigationDrawer/index');
    expect(NavigationDrawerComponent).toBeDefined();
    NavigationDrawer = NavigationDrawerComponent;
  });
  return NavigationDrawer!;
};

describe('NavigationDrawer', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    window.localStorage.clear();
  });
  describe('General functionality', () => {
    it('renders the navigation drawer with the correct buttons', async () => {
      const NavigationDrawer = await importNavigationDrawer();
      renderWithProjectRoute(<NavigationDrawer items={mockItems} />);
      expect(await screen.findByText('Projects')).toBeInTheDocument();
      expect(await screen.findByText('Customizations')).toBeInTheDocument();
      expect(await screen.findByText('Traces')).toBeInTheDocument();
      expect(await screen.findByText('Entries')).toBeInTheDocument();
      expect(await screen.findByText('Export Jobs')).toBeInTheDocument();
      // Expanded, a group is a heading; only the rail turns it into a divider.
      expect(await screen.findByText('Evaluate')).toBeInTheDocument();
      expect(screen.queryByRole('separator')).not.toBeInTheDocument();
    });

    it('respects the local storage if it exists', async () => {
      const NavigationDrawer = await importNavigationDrawer();

      const prevLocalStorage = window.localStorage;
      const localStorageMock = {
        getItem: vi.fn().mockImplementation((key) => {
          if (key === SIDE_NAV_OPEN_KEY) {
            return 'false';
          }
          return null;
        }),
        setItem: vi.fn(),
        removeItem: vi.fn(),
        clear: vi.fn(),
      };
      Object.defineProperty(window, 'localStorage', {
        value: localStorageMock,
      });
      renderWithProjectRoute(
        <PageLayout
          sideNav={(collapsed) => <NavigationDrawer items={mockItems} collapsed={collapsed} />}
        />
      );
      // In the rail a label is only ever exposed via its tooltip.
      expect(
        screen.queryByText('Projects', { ignore: '[role="tooltip"]' })
      ).not.toBeInTheDocument();
      Object.defineProperty(window, 'localStorage', {
        value: prevLocalStorage,
      });
    });

    it('keeps every link reachable when collapsed, parents and children alike', async () => {
      const NavigationDrawer = await importNavigationDrawer();
      renderWithProjectRoute(
        <NavigationDrawer
          collapsed
          items={[
            { id: 'projects', slotLabel: 'Projects', href: ROUTES.workspace.index },
            {
              group: 'Components',
              items: [
                {
                  // Links somewhere itself — stays in the rail alongside its children.
                  id: 'models',
                  slotLabel: 'Models',
                  href: ROUTES.workspace.baseModels,
                  subItems: [
                    {
                      id: 'virtual',
                      slotLabel: 'Virtual Models',
                      href: ROUTES.workspace.virtualModels,
                    },
                  ],
                },
                {
                  // Container only — not a link, so only its children appear.
                  id: 'datasets',
                  slotLabel: 'Datasets',
                  subItems: [
                    {
                      id: 'designer',
                      slotLabel: 'Data Designer',
                      href: ROUTES.workspace.dataDesignerJobList,
                    },
                  ],
                },
              ],
            },
          ]}
        />
      );

      for (const label of ['Projects', 'Models', 'Virtual Models', 'Data Designer']) {
        expect(await screen.findByRole('link', { name: label })).toBeInTheDocument();
      }
      expect(screen.queryByRole('link', { name: 'Datasets' })).not.toBeInTheDocument();
    });

    it('replaces a group heading with a named divider when collapsed', async () => {
      const NavigationDrawer = await importNavigationDrawer();
      // Its own fixture: the rail only keeps items that link somewhere, and `mockItems`' Evaluate
      // group is label-only.
      renderWithProjectRoute(
        <NavigationDrawer
          collapsed
          items={[
            { id: 'projects', slotLabel: 'Projects', href: ROUTES.workspace.index },
            {
              group: 'Evaluate',
              items: [
                { id: 'traces', slotLabel: 'Traces', href: ROUTES.workspace.customizationJobList },
              ],
            },
          ]}
        />
      );

      expect(await screen.findByRole('separator', { name: 'Evaluate' })).toBeInTheDocument();
      expect(screen.queryByText('Evaluate')).not.toBeInTheDocument();
    });

    it('drops a group heading the rail has nothing to put under', async () => {
      const NavigationDrawer = await importNavigationDrawer();
      renderWithProjectRoute(
        <NavigationDrawer
          collapsed
          items={[
            { group: 'Reachable', items: [{ id: 'jobs', slotLabel: 'Jobs', href: '/jobs' }] },
            {
              // Nothing here can be a rail link, so the heading must go rather than leave a
              // divider over empty space.
              group: 'Unreachable',
              items: [
                {
                  id: 'traces',
                  slotLabel: 'Traces',
                  subItems: [{ id: 'entries', slotLabel: 'Entries' }],
                },
              ],
            },
          ]}
        />
      );

      expect(await screen.findByRole('link', { name: 'Jobs' })).toBeInTheDocument();
      expect(screen.queryByRole('separator', { name: 'Unreachable' })).not.toBeInTheDocument();
      expect(screen.queryByRole('separator')).not.toBeInTheDocument();
    });

    it('does not open the collapsed rail with a divider', async () => {
      const NavigationDrawer = await importNavigationDrawer();
      renderWithProjectRoute(
        <NavigationDrawer
          collapsed
          items={[
            { group: 'Evaluate', items: [{ id: 'traces', slotLabel: 'Traces', href: '/traces' }] },
            { group: 'System', items: [{ id: 'jobs', slotLabel: 'Jobs', href: '/jobs' }] },
          ]}
        />
      );

      expect(await screen.findByRole('separator', { name: 'System' })).toBeInTheDocument();
      expect(screen.queryByRole('separator', { name: 'Evaluate' })).not.toBeInTheDocument();
    });

    it('moves the highlight onto a parent that is hiding the active child', async () => {
      const user = userEvent.setup();
      const NavigationDrawer = await importNavigationDrawer();

      const router = createMemoryRouter(
        [
          {
            path: '*',
            element: (
              <NavigationDrawer
                items={[
                  {
                    id: 'traces',
                    slotLabel: 'Traces',
                    href: '/traces',
                    subItems: [{ id: 'entries', slotLabel: 'Entries', href: '/traces/entries' }],
                  },
                ]}
              />
            ),
          },
        ],
        { initialEntries: ['/traces/entries'] }
      );
      render(
        <TestProviders>
          <RouterProvider router={router} />
        </TestProviders>
      );

      // Expanded, the child that matched owns the highlight and the parent stays plain.
      expect(await screen.findByRole('link', { name: 'Entries' })).toHaveClass(
        ACTIVE_NAV_ITEM_CLASS
      );
      expect(screen.getByRole('link', { name: 'Traces' })).not.toHaveClass(ACTIVE_NAV_ITEM_CLASS);

      // Collapsed, the child is gone, so the parent has to carry it.
      await user.click(screen.getByRole('button', { name: /traces/i }));
      expect(screen.queryByRole('link', { name: 'Entries' })).not.toBeInTheDocument();
      expect(screen.getByRole('link', { name: 'Traces' })).toHaveClass(ACTIVE_NAV_ITEM_CLASS);
    });

    it('renders subitems and chevron open/close icons', async () => {
      const user = userEvent.setup();

      const NavigationDrawer = await importNavigationDrawer();
      renderWithProjectRoute(<NavigationDrawer items={mockItems} />);
      expect(await screen.findByText('Traces')).toBeInTheDocument();
      expect(await screen.findByText('Entries')).toBeInTheDocument();
      expect(await screen.findByText('Export Jobs')).toBeInTheDocument();

      // This parent has no href of its own, so the whole row is the disclosure control.
      const trigger = screen.getByRole('button', { name: 'Traces' });

      expect(trigger).toHaveAttribute('aria-expanded', 'true');
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-chevron-down')).toBeInTheDocument();

      await user.click(trigger);
      expect(trigger).toHaveAttribute('aria-expanded', 'false');
      expect(screen.queryByText('Entries')).not.toBeInTheDocument();
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-chevron-right')).toBeInTheDocument();
    });

    it('renders a sub-item with nowhere to go as text rather than a link', async () => {
      const NavigationDrawer = await importNavigationDrawer();
      renderWithProjectRoute(<NavigationDrawer items={mockItems} />);

      // 'Entries' carries no href, so it must not become an anchor back to the current page.
      expect(await screen.findByText('Entries')).toBeInTheDocument();
      expect(screen.queryByRole('link', { name: 'Entries' })).not.toBeInTheDocument();
    });

    it('keeps the label a link and the chevron the sole toggle when a parent has an href', async () => {
      const user = userEvent.setup();

      const NavigationDrawer = await importNavigationDrawer();
      renderWithProjectRoute(
        <NavigationDrawer
          items={[
            {
              id: 'traces',
              slotLabel: 'Traces',
              href: ROUTES.workspace.index,
              subItems: [{ id: 'entries', slotLabel: 'Entries', href: ROUTES.workspace.index }],
            },
          ]}
        />
      );

      // Clicking the label navigates rather than collapsing the sub-list.
      const chevron = screen.getByRole('button', { name: /traces/i });
      await user.click(screen.getByRole('link', { name: 'Traces' }));
      expect(chevron).toHaveAttribute('aria-expanded', 'true');
      expect(screen.getByText('Entries')).toBeInTheDocument();

      await user.click(chevron);
      expect(chevron).toHaveAttribute('aria-expanded', 'false');
      expect(screen.queryByText('Entries')).not.toBeInTheDocument();
    });

    it('keeps a manually toggled parent at the user preference across navigation', async () => {
      const user = userEvent.setup();
      const NavigationDrawer = await importNavigationDrawer();

      // `defaultOpen` follows the route, the way WorkspaceSideNav computes it.
      const Drawer = () => {
        const { pathname } = useLocation();
        return (
          <NavigationDrawer
            items={[
              {
                id: 'traces',
                slotLabel: 'Traces',
                href: '/traces',
                defaultOpen: pathname.startsWith('/traces'),
                subItems: [{ id: 'entries', slotLabel: 'Entries', href: '/traces/entries' }],
              },
              { id: 'jobs', slotLabel: 'Jobs', href: '/jobs' },
            ]}
          />
        );
      };
      const router = createMemoryRouter([{ path: '*', element: <Drawer /> }], {
        initialEntries: ['/traces'],
      });
      render(
        <TestProviders>
          <RouterProvider router={router} />
        </TestProviders>
      );

      const chevron = () => screen.getByRole('button', { name: /traces/i });
      // The current section opens on its own to start.
      expect(chevron()).toHaveAttribute('aria-expanded', 'true');

      // Collapse it by hand.
      await user.click(chevron());
      expect(chevron()).toHaveAttribute('aria-expanded', 'false');

      // Leaving and returning must not reopen it: the user's choice outlives navigation.
      await user.click(screen.getByRole('link', { name: 'Jobs' }));
      await user.click(screen.getByRole('link', { name: 'Traces' }));
      expect(chevron()).toHaveAttribute('aria-expanded', 'false');
      expect(screen.queryByText('Entries')).not.toBeInTheDocument();
    });

    it('restores a stored preference from a prior session over defaultOpen', async () => {
      const NavigationDrawer = await importNavigationDrawer();
      // A previous session collapsed Traces; a fresh load honors it even though the route would
      // otherwise open the section.
      window.localStorage.setItem(NAV_ACCORDION_STATE_KEY, JSON.stringify({ traces: false }));

      const router = createMemoryRouter(
        [
          {
            path: '*',
            element: (
              <NavigationDrawer
                items={[
                  {
                    id: 'traces',
                    slotLabel: 'Traces',
                    href: '/traces',
                    defaultOpen: true,
                    subItems: [{ id: 'entries', slotLabel: 'Entries', href: '/traces/entries' }],
                  },
                ]}
              />
            ),
          },
        ],
        { initialEntries: ['/traces'] }
      );
      render(
        <TestProviders>
          <RouterProvider router={router} />
        </TestProviders>
      );

      const chevron = await screen.findByRole('button', { name: /traces/i });
      expect(chevron).toHaveAttribute('aria-expanded', 'false');
      expect(screen.queryByText('Entries')).not.toBeInTheDocument();
    });
  });
});
