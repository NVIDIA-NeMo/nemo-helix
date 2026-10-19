// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { FilesetListRoute } from '@studio/routes/FilesetListRoute';
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { createMemoryRouter, MemoryRouter, RouterProvider } from 'react-router';

// Mock child components to isolate testing
vi.mock('@studio/routes/FilesetListRoute/ActionMenu', () => ({
  ActionMenu: vi.fn(({ dataset, onNavigateToDetails, onDatasetUpdated, onDatasetDeleted }) => (
    <div data-testid={`action-menu-${dataset.name}`}>
      <button
        onClick={() => onNavigateToDetails?.(dataset)}
        data-testid={`navigate-to-details-${dataset.name}`}
      >
        View Details
      </button>
      <button
        onClick={() => onDatasetUpdated?.({ ...dataset, description: 'Updated description' })}
        data-testid={`update-dataset-${dataset.name}`}
      >
        Update
      </button>
      <button
        onClick={() => onDatasetDeleted?.(dataset)}
        data-testid={`delete-dataset-${dataset.name}`}
      >
        Delete
      </button>
    </div>
  )),
}));

vi.mock('@studio/components/BulkDeleteModal', () => ({
  BulkDeleteModal: vi.fn(({ items, open, onClose }) =>
    open ? (
      <div data-testid="bulk-delete-modal">
        <span>Bulk Delete Modal</span>
        <span data-testid="selected-datasets-count">{items.length}</span>
        <button onClick={() => onClose?.()} data-testid="confirm-bulk-delete">
          Confirm Delete
        </button>
      </div>
    ) : null
  ),
}));

// Mock the route utils
vi.mock('@studio/routes/utils', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/routes/utils')>();
  return {
    ...actual,
    // Mirrors the real helper: callers pass raw values and the helper encodes
    // path params (generatePath does this in production).
    getFilesetDetailsRoute: (project: string, datasetName: string) =>
      `/projects/${project}/filesets/${encodeURIComponent(datasetName)}`,
    getNewFilesetRoute: (project: string) => `/projects/${project}/filesets/new`,
  };
});

vi.mock('@studio/routes/FilesetListRoute/PanelManagement', () => ({
  PanelManagement: vi.fn(({ workspace }) => <div data-testid="panel-management">{workspace}</div>),
}));

const renderRoute = () => {
  return render(
    <TestProviders>
      <MemoryRouter>
        <FilesetListRoute />
      </MemoryRouter>
    </TestProviders>
  );
};

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, () => {}],
}));

describe('FilesetListRoute', () => {
  let user: ReturnType<typeof userEvent.setup>;

  // Mock datasets for testing (using FilesetOutput structure for V2 API)
  const mockDatasets: FilesetOutput[] = [
    {
      id: 'dataset-1-id',
      name: 'dataset-1',
      workspace: 'default',
      description: 'First test dataset',
      purpose: 'dataset',
      storage: { type: 'local', path: '/data/dataset-1' },
      metadata: {},
      custom_fields: {},
      project: 'default',
      created_at: '2024-12-17T16:08:56.880768',
      updated_at: '2024-12-17T16:08:56.880771',
    },
    {
      id: 'dataset-2-id',
      name: 'dataset-2',
      workspace: 'default',
      description: 'Second test dataset',
      purpose: 'dataset',
      storage: { type: 'local', path: '/data/dataset-2' },
      metadata: {},
      custom_fields: {},
      project: 'default',
      created_at: '2024-12-17T15:08:56.880768',
      updated_at: '2024-12-17T15:08:56.880771',
    },
    {
      id: 'dataset-3-id',
      name: 'searchable-dataset',
      workspace: 'default',
      description: 'Dataset for search testing',
      purpose: 'dataset',
      storage: { type: 'local', path: '/data/searchable-dataset' },
      metadata: {},
      custom_fields: {},
      project: 'default',
      created_at: '2024-12-17T14:08:56.880768',
      updated_at: '2024-12-17T14:08:56.880771',
    },
  ];

  const pageEnvelope = (datasets: FilesetOutput[]) => ({
    object: 'list',
    data: datasets,
    pagination: {
      page: 1,
      page_size: 50,
      current_page_size: datasets.length,
      total_pages: 1,
      total_results: datasets.length,
    },
    sort: '-created_at',
  });

  const filterDatasetsByName = (datasets: FilesetOutput[], request: Request): FilesetOutput[] => {
    const nameLike = new URL(request.url).searchParams.get('filter[name][$like]');
    if (!nameLike) return datasets;

    const term = nameLike.replace(/%/g, '').toLowerCase();
    return datasets.filter((d) => d.name.toLowerCase().includes(term));
  };

  beforeEach(() => {
    user = userEvent.setup();
    mockUseParams({
      [ROUTE_PARAMS.workspace]: workspace1.workspace,
    });
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
        ({ request }) => {
          const filtered = filterDatasetsByName(mockDatasets, request);
          return HttpResponse.json(pageEnvelope(filtered));
        }
      )
    );
  });

  describe('Basic Rendering', () => {
    it('renders the filesets list page with header and create button', async () => {
      renderRoute();

      // Target the specific page header element rather than any "Filesets" text
      expect(
        await screen.findByTestId('nv-page-header-heading', undefined, {
          timeout: LG_SELECTOR_TIMEOUT,
        })
      ).toHaveTextContent('Filesets');
      expect(screen.getByRole('link', { name: 'Create Fileset' })).toBeInTheDocument();
    });
  });

  describe('Search Functionality', () => {
    it('filters datasets when search query is entered', async () => {
      renderRoute();

      const searchInput = await screen.findByPlaceholderText('Search filesets...');

      await user.click(searchInput);
      await user.paste('searchable');

      expect(await screen.findByText('searchable-dataset')).toBeInTheDocument();
      await waitFor(() => {
        expect(screen.queryByText('dataset-1')).not.toBeInTheDocument();
      });
      expect(screen.queryByText('dataset-2')).not.toBeInTheDocument();
    });
  });

  // Dataset selection checkbox behavior (select, deselect, select-all, indeterminate)
  // is provided by KUI DataView's built-in rowSelectionColumn and tested upstream.

  describe('Navigation', () => {
    it('navigates to dataset details when a dataset row is clicked', async () => {
      const listPath = `/projects/${workspace1.workspace}/filesets`;

      const router = createMemoryRouter(
        [
          {
            path: '/projects/:workspace/filesets',
            element: (
              <TestProviders>
                <FilesetListRoute />
              </TestProviders>
            ),
          },
          {
            path: '/projects/:workspace/filesets/:filesetId',
            element: <div data-testid="dataset-details-route" />,
          },
        ],
        { initialEntries: [listPath] }
      );

      render(<RouterProvider router={router} />);

      // 1. Wait for data to load and the row click handler to be wired up.
      //    The "Open row" button is injected by useRowClick into every row once
      //    the click handler is active — its presence signals the table is ready.
      await waitFor(
        () => {
          screen.getByText('dataset-1');
          expect(screen.getAllByRole('button', { name: 'Open row' }).length).toBeGreaterThan(0);
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
      // 2. Activate the first row via keyboard on the "Open row" button.
      //    Keyboard activation calls onActivate() directly, bypassing click-delegation
      //    which is unreliable in happy-dom + user-event 14.6.x (pointer coords change).
      const openRowButton = screen.getAllByRole('button', { name: 'Open row' })[0];
      openRowButton.focus();
      await user.keyboard('{Enter}');

      // 3. Wait for navigation to complete
      await waitFor(
        () => {
          expect(screen.getByTestId('dataset-details-route')).toBeInTheDocument();
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
    });

    it('renders action menu with navigation button', async () => {
      renderRoute();

      await screen.findByTestId('navigate-to-details-dataset-1', undefined, {
        timeout: LG_SELECTOR_TIMEOUT,
      });
    });
  });

  describe('Default view (ASTD-830)', () => {
    const make = (
      name: string,
      purpose: FilesetOutput['purpose'],
      over: Partial<FilesetOutput> = {}
    ) =>
      ({
        ...mockDatasets[0],
        id: `${name}-id`,
        name,
        purpose,
        ...over,
      }) as FilesetOutput;

    it('shows dataset/environment/model by default with distinct pills, hides generic, and sorts by recency', async () => {
      const requests: URL[] = [];
      const rows = [
        make('a-dataset', 'dataset'),
        make('b-environment', 'environment'),
        make('c-model', 'model'),
        make('d-generic', 'generic'),
      ];
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
          ({ request }) => {
            const url = new URL(request.url);
            requests.push(url);
            const allowed = url.searchParams.getAll('filter[purpose][$in]').join(',').split(',');
            const data = allowed.includes('')
              ? rows
              : rows.filter((r) => allowed.includes(r.purpose));
            return HttpResponse.json(pageEnvelope(data));
          }
        )
      );

      renderRoute();
      await screen.findByText('a-dataset', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      // (a) single $in filter of dataset/environment/model, default sort is recency
      const lastUrl = requests.at(-1)!;
      expect(
        lastUrl.searchParams.getAll('filter[purpose][$in]').join(',').split(',').sort()
      ).toEqual(['dataset', 'environment', 'model']);
      expect(lastUrl.searchParams.get('sort')).toBe('-created_at');

      // (b) generic never rendered
      expect(screen.queryByText('d-generic')).not.toBeInTheDocument();

      // (c) one pill per purpose, each with a distinct color class
      const purposes = ['dataset', 'environment', 'model'] as const;
      const classes = purposes.map((p) => screen.getAllByTestId(`purpose-pill-${p}`)[0].className);
      expect(new Set(classes).size).toBe(purposes.length);

      // dropdown: every filter defaults to "All"; Purpose offers no "Generic" option
      await user.click(screen.getByRole('button', { name: /filter/i }));
      const options = screen.getAllByRole('option');
      const names = options.map((o) => o.textContent);
      expect(names).toEqual(expect.arrayContaining(['Dataset', 'Environment', 'Model']));
      expect(names).not.toContain('Generic');
      options
        .filter((o) => o.textContent === 'All')
        .forEach((o) => expect(o).toHaveAttribute('aria-selected', 'true'));
    });
  });
});
