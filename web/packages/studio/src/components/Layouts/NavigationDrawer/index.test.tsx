// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VerticalNavTestIds } from '@nvidia/foundations-react-core';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { PageLayout } from '@studio/routes/PageLayout';
import { getWorkspaceIndexRoute } from '@studio/routes/utils';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { NAV_ACCORDION_STATE_KEY, SIDE_NAV_OPEN_KEY } from '@studio/util/localStorage';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider, useLocation } from 'react-router';

const KUI_ACTIVE_CLASS = 'nv-vertical-nav-item--active';
const KUI_COLLAPSED_CLASS = 'nv-vertical-nav-root--collapsed';

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
      expect(await screen.findByTestId(VerticalNavTestIds.VerticalNavHeading)).toHaveTextContent(
        'Evaluate'
      );
      expect(screen.getByTestId(VerticalNavTestIds.VerticalNavRoot)).not.toHaveClass(
        KUI_COLLAPSED_CLASS
      );
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
      try {
        renderWithProjectRoute(
          <PageLayout
            sideNav={(collapsed) => <NavigationDrawer items={mockItems} collapsed={collapsed} />}
          />
        );
        expect(await screen.findByTestId(VerticalNavTestIds.VerticalNavRoot)).toHaveClass(
          KUI_COLLAPSED_CLASS
        );
      } finally {
        Object.defineProperty(window, 'localStorage', {
          value: prevLocalStorage,
        });
      }
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

    it('keeps a group heading in the rail for KUI to draw as a divider', async () => {
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

      expect(await screen.findByTestId(VerticalNavTestIds.VerticalNavHeading)).toHaveTextContent(
        'Evaluate'
      );
      expect(screen.getByTestId(VerticalNavTestIds.VerticalNavRoot)).toHaveClass(
        KUI_COLLAPSED_CLASS
      );
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
      expect(screen.queryByText('Unreachable')).not.toBeInTheDocument();
      expect(screen.queryByTestId(VerticalNavTestIds.VerticalNavHeading)).not.toBeInTheDocument();
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

      expect(await screen.findByTestId(VerticalNavTestIds.VerticalNavHeading)).toHaveTextContent(
        'System'
      );
      expect(screen.queryByText('Evaluate')).not.toBeInTheDocument();
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
      expect(await screen.findByRole('link', { name: 'Entries' })).toHaveClass(KUI_ACTIVE_CLASS);
      expect(screen.getByRole('link', { name: 'Traces' })).not.toHaveClass(KUI_ACTIVE_CLASS);

      // Collapsed, the child is gone, so the parent has to carry it.
      await user.click(screen.getByRole('button', { name: /traces/i }));
      expect(screen.queryByRole('link', { name: 'Entries' })).not.toBeInTheDocument();
      expect(screen.getByRole('link', { name: 'Traces' })).toHaveClass(KUI_ACTIVE_CLASS);
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

      await user.click(trigger);
      expect(trigger).toHaveAttribute('aria-expanded', 'false');
      expect(screen.queryByText('Entries')).not.toBeInTheDocument();
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

  describe('Collapsible section headings', () => {
    const sectionItems = [
      { id: 'dashboard', slotLabel: 'Dashboard', href: '/dashboard' },
      {
        group: 'Agent Development',
        items: [
          { id: 'agents', slotLabel: 'Agents', href: '/agents' },
          { id: 'monitor', slotLabel: 'Monitor', href: '/monitor' },
        ],
      },
      { group: 'Models', items: [{ id: 'catalog', slotLabel: 'Model Catalog', href: '/catalog' }] },
    ];

    const renderSections = async (initialEntry = '/dashboard') => {
      const NavigationDrawer = await importNavigationDrawer();
      const router = createMemoryRouter(
        [{ path: '*', element: <NavigationDrawer items={sectionItems} /> }],
        { initialEntries: [initialEntry] }
      );
      render(
        <TestProviders>
          <RouterProvider router={router} />
        </TestProviders>
      );
      await screen.findByRole('link', { name: 'Dashboard' });
    };

    const storedState = () =>
      JSON.parse(window.localStorage.getItem(NAV_ACCORDION_STATE_KEY) ?? '{}');

    // KUI renders a collapsible heading as <summary> inside a native <details>.
    const summaryOf = (heading: string) => {
      const summary = screen.getByText(heading).closest('summary');
      expect(summary, `"${heading}" should be a collapsible <summary>`).not.toBeNull();
      return summary as HTMLElement;
    };
    const sectionOf = (heading: string) => summaryOf(heading).closest('details') as HTMLDetailsElement;

    it('starts every headed section open with its rows visible', async () => {
      await renderSections();

      for (const heading of ['Agent Development', 'Models']) {
        expect(sectionOf(heading)).toHaveAttribute('open');
      }
      for (const row of ['Agents', 'Monitor', 'Model Catalog']) {
        expect(screen.getByRole('link', { name: row })).toBeVisible();
      }
    });

    it('collapses a section on heading click, leaves others open, and reopens on a second click', async () => {
      const user = userEvent.setup();
      await renderSections();

      await user.click(summaryOf('Agent Development'));

      expect(sectionOf('Agent Development')).not.toHaveAttribute('open');
      expect(sectionOf('Models')).toHaveAttribute('open');

      await user.click(summaryOf('Agent Development'));

      expect(sectionOf('Agent Development')).toHaveAttribute('open');
    });

    it('persists the choice under group:<heading> as a boolean', async () => {
      const user = userEvent.setup();
      await renderSections();

      await user.click(summaryOf('Models'));
      expect(storedState()).toEqual({ 'group:Models': false });

      await user.click(summaryOf('Models'));
      expect(storedState()).toEqual({ 'group:Models': true });
    });

    it('starts a section collapsed when storage says so', async () => {
      window.localStorage.setItem(
        NAV_ACCORDION_STATE_KEY,
        JSON.stringify({ 'group:Agent Development': false })
      );
      await renderSections();

      expect(sectionOf('Agent Development')).not.toHaveAttribute('open');
      expect(sectionOf('Models')).toHaveAttribute('open');
    });

    it('keeps a collapsed section collapsed after navigating, beside item accordion keys', async () => {
      const user = userEvent.setup();
      window.localStorage.setItem(NAV_ACCORDION_STATE_KEY, JSON.stringify({ agents: false }));
      await renderSections();

      await user.click(summaryOf('Models'));
      await user.click(screen.getByRole('link', { name: 'Agents' }));

      expect(sectionOf('Models')).not.toHaveAttribute('open');
      expect(storedState()).toEqual({ agents: false, 'group:Models': false });
    });

    it('gives the unlabeled first group no collapsible heading', async () => {
      await renderSections();

      expect(screen.getByRole('link', { name: 'Dashboard' }).closest('details')).toBeNull();
      expect(document.querySelectorAll('summary')).toHaveLength(2);
    });
  });
});
