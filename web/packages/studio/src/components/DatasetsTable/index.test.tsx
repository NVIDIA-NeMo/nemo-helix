// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { FilesetPurpose, type FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { DatasetCreateModalMode } from '@studio/components/DatasetCreateModal/constants';
import { DatasetsTable } from '@studio/components/DatasetsTable';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import type { ComponentProps } from 'react';
import { MemoryRouter } from 'react-router';

// Signed-out by default; the owned-first ordering is covered by useOwnedFirstFilesets.test.tsx
vi.mock('@studio/providers/auth/useAuthProfile', () => ({ useAuthProfile: () => undefined }));

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, () => {}],
}));

vi.mock('@studio/components/BulkDeleteModal', () => ({
  BulkDeleteModal: vi.fn(
    ({
      items,
      open,
      onDelete,
      onClose,
    }: {
      items: FilesetOutput[];
      open: boolean;
      onDelete: (items: FilesetOutput[]) => Promise<void>;
      onClose: () => void;
    }) =>
      open ? (
        <div data-testid="bulk-delete-modal">
          <span data-testid="bulk-modal-count">{items.length}</span>
          <button
            type="button"
            data-testid="bulk-confirm"
            onClick={async () => {
              await onDelete(items);
              onClose();
            }}
          >
            Confirm
          </button>
        </div>
      ) : null
  ),
}));

vi.mock('@nemo/common/src/components/DeleteConfirmationModal', () => ({
  DeleteConfirmationModal: vi.fn(
    ({
      open,
      title,
      confirmationText,
    }: {
      open?: boolean;
      title?: string;
      confirmationText?: string;
    }) =>
      open ? (
        <div data-testid="delete-confirmation-modal">
          <span data-testid="delete-title">{title}</span>
          <span data-testid="delete-confirmation-text">{confirmationText}</span>
        </div>
      ) : null
  ),
}));

vi.mock('@studio/components/DatasetCreateModal', () => ({
  DatasetCreateModal: vi.fn(
    ({ open, mode, dataset }: { open?: boolean; mode?: string; dataset?: FilesetOutput }) =>
      open ? (
        <div data-testid="dataset-create-modal">
          <span data-testid="modal-mode">{mode}</span>
          <span data-testid="modal-dataset-name">{dataset?.name}</span>
        </div>
      ) : null
  ),
}));

vi.mock('@studio/components/FilesetCreateModal', () => ({
  FilesetCreateModal: vi.fn(({ open, purpose }: { open?: boolean; purpose?: string }) =>
    open ? (
      <div data-testid="fileset-create-modal">
        <span data-testid="fileset-modal-purpose">{purpose}</span>
      </div>
    ) : null
  ),
}));

vi.mock('@studio/constants/environment', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/environment')>();
  return { ...actual, FILESET_DETAILS_ENABLED: true };
});

const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;

const makeDataset = (overrides: Partial<FilesetOutput> & { name: string }): FilesetOutput =>
  ({
    id: `${overrides.name}-id`,
    workspace: 'default',
    description: `Description for ${overrides.name}`,
    purpose: 'dataset',
    storage: { type: 'local', path: `/data/${overrides.name}` },
    metadata: {},
    custom_fields: {},
    project: 'default',
    created_at: '2024-12-17T16:08:56.880768',
    updated_at: '2024-12-17T16:08:56.880771',
    ...overrides,
  }) as FilesetOutput;

const defaultDatasets: FilesetOutput[] = [
  makeDataset({ name: 'dataset-1' }),
  makeDataset({ name: 'dataset-2' }),
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

/**
 * Replace the filesets list handler and record each request. Returns refs so
 * tests can assert on outgoing params (sort, page, page_size, etc.) and count.
 */
const installListHandler = (datasets: FilesetOutput[] = defaultDatasets) => {
  const state = { lastUrl: null as URL | null, requestCount: 0 };
  server.use(
    http.get(FILESETS_URL, ({ request }) => {
      state.lastUrl = new URL(request.url);
      state.requestCount += 1;
      return HttpResponse.json(pageEnvelope(datasets));
    })
  );
  return state;
};

const renderTable = (props: Partial<ComponentProps<typeof DatasetsTable>> = {}) =>
  render(
    <TestProviders>
      <MemoryRouter>
        <DatasetsTable {...props} />
      </MemoryRouter>
    </TestProviders>
  );

describe('DatasetsTable', () => {
  let user: ReturnType<typeof userEvent.setup>;

  /**
   * Click a row-selection checkbox and wait for the selection to register.
   * KUI disables checkboxes while `requestStatus` is "loading" and re-renders
   * swap DOM nodes, so we wait for the checkbox to be enabled, click it once
   * outside waitFor, then wait for the expected aria-checked state.
   */
  const getCheckboxAt = (index: number) =>
    screen.getAllByRole('checkbox', { name: /(De)?select row/i })[index];

  const selectRow = async (index: number) => {
    await waitFor(
      () => {
        expect(getCheckboxAt(index)).toBeEnabled();
      },
      { timeout: LG_SELECTOR_TIMEOUT }
    );
    const checkbox = getCheckboxAt(index) as HTMLInputElement;
    if (!checkbox.checked) {
      await user.click(checkbox);
    }
    await waitFor(
      () => {
        expect(getCheckboxAt(index)).toBeChecked();
      },
      { timeout: LG_SELECTOR_TIMEOUT }
    );
  };

  const deselectRow = async (index: number) => {
    await waitFor(
      () => {
        expect(getCheckboxAt(index)).toBeEnabled();
      },
      { timeout: LG_SELECTOR_TIMEOUT }
    );
    const checkbox = getCheckboxAt(index) as HTMLInputElement;
    if (checkbox.checked) {
      await user.click(checkbox);
    }
    await waitFor(
      () => {
        expect(getCheckboxAt(index)).not.toBeChecked();
      },
      { timeout: LG_SELECTOR_TIMEOUT }
    );
  };

  beforeEach(() => {
    user = userEvent.setup();
    mockUseParams({ [ROUTE_PARAMS.workspace]: workspace1.workspace });
  });

  describe('Columns', () => {
    it('renders the standard columns (Name, Purpose, Storage Backend, Path, Description, Created By, Created)', async () => {
      installListHandler();
      renderTable();

      // Name is a plain header when enableFilters is off
      expect(
        await screen.findByText('Name', undefined, { timeout: LG_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();
      expect(screen.getAllByText('Purpose').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Storage Backend').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Path')).toBeInTheDocument();
      expect(screen.getByText('Description')).toBeInTheDocument();
      expect(screen.getByText('Created By')).toBeInTheDocument();
      expect(screen.getByText('Created')).toBeInTheDocument();
    });

    it('renders created_by as returned by the API', async () => {
      installListHandler([
        makeDataset({ name: 'email-ds', created_by: 'someone@example.com' }),
        makeDataset({ name: 'id-ds', created_by: 'a1b2c3d4-principal' }),
      ]);
      renderTable();

      await screen.findByText('email-ds', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      expect(screen.getByText('someone@example.com')).toBeInTheDocument();
      expect(screen.getByText('a1b2c3d4-principal')).toBeInTheDocument();
    });

    it('renders the empty placeholder when created_by is null or missing', async () => {
      installListHandler([
        makeDataset({ name: 'null-owner-ds', created_by: null }),
        makeDataset({ name: 'missing-owner-ds' }),
      ]);
      renderTable();

      await screen.findByText('null-owner-ds', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      expect(screen.getAllByText(EMPTY_FIELD_VALUE)).toHaveLength(2);
      expect(screen.queryByText('null')).not.toBeInTheDocument();
      expect(screen.queryByText('undefined')).not.toBeInTheDocument();
    });

    it('renders friendly purpose labels in the Purpose column cells', async () => {
      installListHandler([
        makeDataset({ name: 'generic-ds', purpose: 'generic' }),
        makeDataset({ name: 'dataset-ds', purpose: 'dataset' }),
        makeDataset({ name: 'model-ds', purpose: 'model' }),
      ]);
      renderTable();

      await screen.findByText('generic-ds', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      // Labels appear in cells (and may also appear in filter options if the filter panel is open) — cells is enough
      expect(screen.getAllByText('Generic').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Dataset').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Model').length).toBeGreaterThanOrEqual(1);
    });

    it('omits the selection column when enableSelection is false', async () => {
      installListHandler();
      renderTable({ enableSelection: false });

      await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
      expect(screen.queryByRole('checkbox', { name: /(De)?select row/i })).not.toBeInTheDocument();
    });

    it('renders the selection column when enableSelection is true', async () => {
      installListHandler();
      renderTable({ enableSelection: true });

      await waitFor(
        () => {
          expect(
            screen.getAllByRole('checkbox', { name: /(De)?select row/i }).length
          ).toBeGreaterThan(0);
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
    });

    it('labels storage backends using the friendly label map', async () => {
      installListHandler([
        makeDataset({ name: 'local-ds' }),
        makeDataset({
          name: 'ngc-ds',
          storage: {
            type: 'ngc',
            org: 'nv',
            team: 'core',
            target: 'foo',
          } as FilesetOutput['storage'],
        }),
        makeDataset({
          name: 'hf-ds',
          storage: { type: 'huggingface', repo_id: 'owner/repo' } as FilesetOutput['storage'],
        }),
        makeDataset({
          name: 's3-ds',
          storage: { type: 's3', bucket: 'my-bucket', prefix: 'data' } as FilesetOutput['storage'],
        }),
      ]);
      renderTable();

      await screen.findByText('local-ds', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      // Labels appear in both the column cells and filter accordion — cells is enough
      expect(screen.getAllByText('Local').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('NGC').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Hugging Face').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('S3').length).toBeGreaterThanOrEqual(1);
    });

    it('derives the path cell from the storage config shape', async () => {
      installListHandler([
        makeDataset({
          name: 'local-ds',
          storage: { type: 'local', path: '/mnt/data' } as FilesetOutput['storage'],
        }),
        makeDataset({
          name: 'ngc-ds',
          storage: {
            type: 'ngc',
            org: 'nv',
            team: 'core',
            target: 'foo',
          } as FilesetOutput['storage'],
        }),
        makeDataset({
          name: 'hf-ds',
          storage: { type: 'huggingface', repo_id: 'owner/repo' } as FilesetOutput['storage'],
        }),
        makeDataset({
          name: 's3-prefix-ds',
          storage: { type: 's3', bucket: 'bk', prefix: 'pre' } as FilesetOutput['storage'],
        }),
        makeDataset({
          name: 's3-no-prefix-ds',
          storage: { type: 's3', bucket: 'bk-only' } as FilesetOutput['storage'],
        }),
      ]);
      renderTable();

      await screen.findByText('local-ds', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      expect(screen.getByText('/mnt/data')).toBeInTheDocument();
      expect(screen.getByText('nv/core/foo')).toBeInTheDocument();
      expect(screen.getByText('owner/repo')).toBeInTheDocument();
      expect(screen.getByText('bk/pre')).toBeInTheDocument();
      expect(screen.getByText('bk-only')).toBeInTheDocument();
    });
  });

  describe('Filtering', () => {
    it('sends the checked purposes as a $in filter, including generic when checked', async () => {
      const state = installListHandler();
      const filtersParam = encodeURIComponent(
        JSON.stringify([{ id: 'purpose', value: { dataset: true, generic: true } }])
      );

      render(
        <TestProviders>
          <MemoryRouter initialEntries={[`/?filters=${filtersParam}`]}>
            <DatasetsTable enableFilters />
          </MemoryRouter>
        </TestProviders>
      );

      await waitFor(
        () => {
          expect(state.lastUrl?.searchParams.getAll('filter[purpose][$in]').sort()).toEqual([
            'dataset',
            'generic',
          ]);
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
    });

    it('keeps other URL filters and still seeds the default purposes', async () => {
      const state = installListHandler();
      const filtersParam = encodeURIComponent(
        JSON.stringify([{ id: 'storage_type', value: 'local' }])
      );
      render(
        <TestProviders>
          <MemoryRouter initialEntries={[`/?filters=${filtersParam}`]}>
            <DatasetsTable enableFilters />
          </MemoryRouter>
        </TestProviders>
      );
      await waitFor(
        () => {
          expect(state.lastUrl?.searchParams.get('filter[storage_type]')).toBe('local');
          expect(state.lastUrl?.searchParams.getAll('filter[purpose][$in]').sort()).toEqual([
            'dataset',
            'environment',
            'model',
          ]);
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
    });

    it('does not narrow purposes when filters are off (pickers list every purpose)', async () => {
      const state = installListHandler();
      renderTable();
      await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
      expect(state.lastUrl?.searchParams.has('filter[purpose][$in]')).toBe(false);
    });

    it('honours a legacy single-select purpose value from a saved URL', async () => {
      const state = installListHandler();
      const filtersParam = encodeURIComponent(
        JSON.stringify([{ id: 'purpose', value: 'generic' }])
      );
      render(
        <TestProviders>
          <MemoryRouter initialEntries={[`/?filters=${filtersParam}`]}>
            <DatasetsTable enableFilters />
          </MemoryRouter>
        </TestProviders>
      );
      await waitFor(
        () => {
          expect(state.lastUrl?.searchParams.getAll('filter[purpose][$in]')).toEqual(['generic']);
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
    });
  });

  describe('Sorting', () => {
    it('sends sort=name when Name header is clicked (ascending on first click)', async () => {
      const state = installListHandler();
      renderTable({ enableFilters: true });

      const nameHeader = await screen.findByRole('button', {
        name: /^Name/,
      });
      await user.click(nameHeader);

      await waitFor(
        () => {
          expect(state.lastUrl?.searchParams.get('sort')).toBe('name');
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
    });

    it('toggles sort direction on second click of the same header', async () => {
      const state = installListHandler();
      renderTable({ enableFilters: true });

      const nameHeader = await screen.findByRole('button', { name: /^Name/ });

      await user.click(nameHeader);
      await waitFor(() => expect(state.lastUrl?.searchParams.get('sort')).toBe('name'));

      await user.click(screen.getByRole('button', { name: /^Name/ }));
      await waitFor(() => expect(state.lastUrl?.searchParams.get('sort')).toBe('-name'));
    });

    it('omits the sort param when enableFilters is false', async () => {
      const state = installListHandler();
      renderTable({ enableFilters: false });

      await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
      expect(state.lastUrl?.searchParams.get('sort')).toBeNull();
    });
  });

  describe('Loading/Error/Empty states', () => {
    it('shows the "Loading filesets..." spinner while the initial request is in flight', async () => {
      let resolveResponse!: () => void;
      const gate = new Promise<void>((r) => {
        resolveResponse = r;
      });
      server.use(
        http.get(FILESETS_URL, async () => {
          await gate;
          return HttpResponse.json(pageEnvelope(defaultDatasets));
        })
      );

      renderTable();

      expect(
        await screen.findByText('Loading filesets...', undefined, { timeout: LG_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();

      resolveResponse();

      expect(
        await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();
      expect(screen.queryByText('Loading filesets...')).not.toBeInTheDocument();
    });

    it('shows "Failed to fetch filesets" with a Retry button on API error and re-requests on retry', async () => {
      let requestCount = 0;
      let shouldFail = true;
      server.use(
        http.get(FILESETS_URL, () => {
          requestCount += 1;
          if (shouldFail) {
            return HttpResponse.json({ error: 'boom' }, { status: 500 });
          }
          return HttpResponse.json(pageEnvelope(defaultDatasets));
        })
      );

      renderTable();

      const retryButton = await screen.findByRole(
        'button',
        { name: 'Retry' },
        {
          timeout: LG_SELECTOR_TIMEOUT,
        }
      );
      expect(screen.getByText('Failed to fetch filesets')).toBeInTheDocument();

      const countBeforeRetry = requestCount;
      shouldFail = false;
      await user.click(retryButton);

      await waitFor(() => expect(requestCount).toBeGreaterThan(countBeforeRetry), {
        timeout: LG_SELECTOR_TIMEOUT,
      });
    });

    it('shows the first-use empty state with no filters active and opens the create modal', async () => {
      installListHandler([]);
      renderTable();

      expect(
        await screen.findByText(ENTITY_EMPTY_STATES.filesets.heading, undefined, {
          timeout: LG_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();
      expect(screen.getByText(ENTITY_EMPTY_STATES.filesets.subheading)).toBeInTheDocument();

      const createButton = screen.getByRole('button', {
        name: ENTITY_EMPTY_STATES.filesets.createAction?.label,
      });
      await user.click(createButton);

      const modal = await screen.findByTestId('fileset-create-modal');
      expect(within(modal).getByTestId('fileset-modal-purpose')).toHaveTextContent(
        FilesetPurpose.dataset
      );
    });

    it('still shows the first-use empty state when only the seeded default purposes are set', async () => {
      installListHandler([]);
      renderTable({ enableFilters: true });

      expect(
        await screen.findByText(ENTITY_EMPTY_STATES.filesets.heading, undefined, {
          timeout: LG_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();
    });

    it('shows the no-results empty state with a Clear filters button when search is active', async () => {
      installListHandler([]);
      renderTable({ enableFilters: true });

      // Type into the search bar to activate hasSearchApplied
      const searchInput = await screen.findByPlaceholderText(/search/i, undefined, {
        timeout: LG_SELECTOR_TIMEOUT,
      });
      await user.click(searchInput);
      await user.paste('no-match');

      expect(
        await screen.findByText('No results found', undefined, { timeout: LG_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();
      expect(
        screen.getByText('No items match your current search or filters.')
      ).toBeInTheDocument();
      // the applied-filters bar (default purposes) also offers one, so expect at least the empty state's
      expect(screen.getAllByRole('button', { name: /Clear filters/i }).length).toBeGreaterThan(0);
    });
  });

  describe('Row selection', () => {
    it('invokes onDatasetsSelected with the selected dataset when a row is selected', async () => {
      installListHandler();
      const onDatasetsSelected = vi.fn();
      renderTable({ enableSelection: true, onDatasetsSelected });

      await selectRow(0);

      await waitFor(() => {
        const lastCall = onDatasetsSelected.mock.calls.at(-1);
        expect(lastCall?.[0]).toHaveLength(1);
        expect(lastCall?.[0][0].name).toBe('dataset-1');
      });
    });

    it('invokes onDatasetsSelected with [] when a selected row is deselected', async () => {
      installListHandler();
      const onDatasetsSelected = vi.fn();
      renderTable({ enableSelection: true, onDatasetsSelected });

      await selectRow(0);
      await deselectRow(0);

      await waitFor(() => {
        const lastCall = onDatasetsSelected.mock.calls.at(-1);
        expect(lastCall?.[0]).toHaveLength(0);
      });
    });

    it('keeps only the most recently selected row when selectionType="single"', async () => {
      installListHandler();
      const onDatasetsSelected = vi.fn();
      renderTable({
        enableSelection: true,
        selectionType: 'single',
        onDatasetsSelected,
      });

      await selectRow(0);
      await selectRow(1);

      await waitFor(() => {
        const lastCall = onDatasetsSelected.mock.calls.at(-1);
        expect(lastCall?.[0]).toHaveLength(1);
        expect(lastCall?.[0][0].name).toBe('dataset-2');
      });
    });

    it('does not crash when no onDatasetsSelected callback is provided', async () => {
      installListHandler();
      renderTable({ enableSelection: true });

      await selectRow(0);
      // Implicit assertion: reaching this line without throwing is a pass
      expect(screen.getAllByRole('checkbox', { name: /(De)?select row/i })[0]).toBeChecked();
    });
  });

  describe('Row click', () => {
    it('fires both onRowClick and navigates when both are provided', async () => {
      installListHandler();
      const onRowClick = vi.fn();
      const getDatasetRoute = vi.fn((dataset: FilesetOutput) => `/datasets/${dataset.name}`);
      renderTable({ onRowClick, getDatasetRoute });

      await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      const openButtons = await screen.findAllByRole('button', { name: 'Open row' });
      openButtons[0].focus();
      await user.keyboard('{Enter}');

      await waitFor(() => {
        expect(onRowClick).toHaveBeenCalledTimes(1);
        expect(onRowClick.mock.calls[0][0].name).toBe('dataset-1');
      });
      expect(getDatasetRoute).toHaveBeenCalledWith(expect.objectContaining({ name: 'dataset-1' }));
    });

    it('selects the row on click when enableSelection is true and enableFilters is false', async () => {
      installListHandler();
      const onDatasetsSelected = vi.fn();
      const onRowClick = vi.fn();
      renderTable({ enableSelection: true, enableFilters: false, onRowClick, onDatasetsSelected });

      await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
      const openButtons = await screen.findAllByRole('button', { name: 'Open row' });
      openButtons[0].focus();
      await user.keyboard('{Enter}');

      await waitFor(() => {
        const lastCall = onDatasetsSelected.mock.calls.at(-1);
        expect(lastCall?.[0]).toHaveLength(1);
        expect(lastCall?.[0][0].name).toBe('dataset-1');
      });
    });

    it('does NOT auto-select the row on click when enableFilters is true', async () => {
      installListHandler();
      const onDatasetsSelected = vi.fn();
      const onRowClick = vi.fn();
      renderTable({ enableSelection: true, enableFilters: true, onRowClick, onDatasetsSelected });

      await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
      const openButtons = await screen.findAllByRole('button', { name: 'Open row' });
      openButtons[0].focus();
      await user.keyboard('{Enter}');

      await waitFor(() => expect(onRowClick).toHaveBeenCalledTimes(1));
      // Selection should remain empty — onDatasetsSelected should never fire with a selection
      expect(
        onDatasetsSelected.mock.calls.some((args) => (args[0] as FilesetOutput[]).length > 0)
      ).toBe(false);
    });
  });

  describe('Row actions (renderRowActions)', () => {
    type Callbacks = {
      onEdit: () => void;
      onDelete: () => void;
      onDatasetDeleted: (dataset: FilesetOutput) => void;
    };

    const renderWithActionButtons = (onDatasetsSelected?: (datasets: FilesetOutput[]) => void) =>
      renderTable({
        enableSelection: true,
        onDatasetsSelected,
        renderRowActions: (dataset, callbacks: Callbacks) => (
          <div data-testid={`row-actions-${dataset.name}`}>
            <button
              type="button"
              data-testid={`edit-${dataset.name}`}
              onClick={() => callbacks.onEdit()}
            >
              Edit
            </button>
            <button
              type="button"
              data-testid={`delete-${dataset.name}`}
              onClick={() => callbacks.onDelete()}
            >
              Delete
            </button>
            <button
              type="button"
              data-testid={`notify-deleted-${dataset.name}`}
              onClick={() => callbacks.onDatasetDeleted(dataset)}
            >
              Notify deleted
            </button>
          </div>
        ),
      });

    it('opens DatasetCreateModal in edit mode with the selected dataset when onEdit fires', async () => {
      installListHandler();
      renderWithActionButtons();

      const editButton = await screen.findByTestId('edit-dataset-1', undefined, {
        timeout: LG_SELECTOR_TIMEOUT,
      });
      await user.click(editButton);

      const modal = await screen.findByTestId('dataset-create-modal');
      expect(within(modal).getByTestId('modal-mode')).toHaveTextContent(
        DatasetCreateModalMode.Edit
      );
      expect(within(modal).getByTestId('modal-dataset-name')).toHaveTextContent('dataset-1');
    });

    it('opens DeleteConfirmationModal with the dataset name as confirmation text when onDelete fires', async () => {
      installListHandler();
      renderWithActionButtons();

      const deleteButton = await screen.findByTestId('delete-dataset-1', undefined, {
        timeout: LG_SELECTOR_TIMEOUT,
      });
      await user.click(deleteButton);

      const modal = await screen.findByTestId('delete-confirmation-modal');
      expect(within(modal).getByTestId('delete-title')).toHaveTextContent(
        'Delete Dataset: dataset-1'
      );
      expect(within(modal).getByTestId('delete-confirmation-text')).toHaveTextContent('dataset-1');
    });

    it('removes the deleted dataset from selection when onDatasetDeleted fires', async () => {
      installListHandler();
      const onDatasetsSelected = vi.fn();
      renderWithActionButtons(onDatasetsSelected);

      await selectRow(0);
      await waitFor(() => expect(onDatasetsSelected.mock.calls.at(-1)?.[0]).toHaveLength(1));

      const notifyButton = screen.getByTestId('notify-deleted-dataset-1');
      await user.click(notifyButton);

      await waitFor(() => {
        expect(onDatasetsSelected.mock.calls.at(-1)?.[0]).toHaveLength(0);
      });
    });
  });

  describe('Bulk delete', () => {
    const clickBulkDeleteTrigger = async () => {
      const trigger = await screen.findByRole('button', { name: /delete selected datasets/i });
      await user.click(trigger);
    };

    it('does not render BulkDeleteModal when enableBulkDelete is false', async () => {
      installListHandler();
      renderTable({ enableSelection: true, enableBulkDelete: false });

      await selectRow(0);
      expect(screen.queryByTestId('bulk-delete-modal')).not.toBeInTheDocument();
    });

    it('renders BulkDeleteModal with the selected datasets when enableBulkDelete is true', async () => {
      installListHandler();
      renderTable({ enableSelection: true, enableBulkDelete: true });

      await selectRow(0);
      await clickBulkDeleteTrigger();

      const modal = await screen.findByTestId('bulk-delete-modal');
      expect(within(modal).getByTestId('bulk-modal-count')).toHaveTextContent('1');
    });

    it('clears selection and re-fetches after a successful bulk delete', async () => {
      const state = installListHandler();
      const onDatasetsSelected = vi.fn();
      renderTable({
        enableSelection: true,
        enableBulkDelete: true,
        onDatasetsSelected,
      });

      await selectRow(0);
      await waitFor(() => expect(onDatasetsSelected.mock.calls.at(-1)?.[0]).toHaveLength(1));

      await clickBulkDeleteTrigger();

      const countBeforeConfirm = state.requestCount;
      await user.click(screen.getByTestId('bulk-confirm'));

      await waitFor(() => {
        expect(onDatasetsSelected.mock.calls.at(-1)?.[0]).toHaveLength(0);
      });
      await waitFor(() => {
        expect(state.requestCount).toBeGreaterThan(countBeforeConfirm);
      });
    });
  });

  describe('Pagination', () => {
    it('sends page=1 and the default page_size on first request', async () => {
      const state = installListHandler();
      renderTable();

      await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
      expect(state.lastUrl?.searchParams.get('page')).toBe('1');
      expect(state.lastUrl?.searchParams.get('page_size')).toBe('50');
    });

    it('renders pagination controls (items-per-page selector + page range text)', async () => {
      installListHandler();
      renderTable();

      expect(
        await screen.findByText('Items per page', undefined, { timeout: LG_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();
      expect(screen.getByText('1-2 of 2 items')).toBeInTheDocument();
    });
  });
});
