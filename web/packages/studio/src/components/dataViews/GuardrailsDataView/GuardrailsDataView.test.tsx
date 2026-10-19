// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import type { GuardrailConfig } from '@nemo/sdk/generated/platform/schema';
import { GuardrailsDataView } from '@studio/components/dataViews/GuardrailsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { mockGuardrailConfigs } from '@studio/mocks/handlers/guardrails';
import { server } from '@studio/mocks/node';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { createMemoryRouter, RouterProvider } from 'react-router';

const workspace = 'default';

const findPiiFilterRow = async () => {
  return await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });
};

const renderComponent = (
  props: {
    onRowClick?: (config: GuardrailConfig) => void;
    onRequestDelete?: (config: GuardrailConfig) => void;
    onCreate?: () => void;
    onRequestBulkDelete?: (configs: GuardrailConfig[]) => void;
  } = {}
) => {
  const router = createMemoryRouter([
    {
      path: '/',
      element: (
        <GuardrailsDataView
          workspace={workspace}
          onRowClick={props.onRowClick ?? vi.fn()}
          onRequestDelete={props.onRequestDelete}
          onCreate={props.onCreate ?? vi.fn()}
          onRequestBulkDelete={props.onRequestBulkDelete}
        />
      ),
    },
  ]);

  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

/** Override the list handler to return data sorted by the `sort` URL param. */
const mockSortedConfigs = (configs = mockGuardrailConfigs) => {
  server.use(
    http.get(
      `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`,
      ({ request }) => {
        const url = new URL(request.url);
        const sort = url.searchParams.get('sort') ?? '-created_at';
        const desc = sort.startsWith('-');
        const field = (desc ? sort.slice(1) : sort) as keyof GuardrailConfig;
        const sorted = [...configs].sort((a, b) => {
          const cmp = String(a[field] ?? '').localeCompare(String(b[field] ?? ''));
          return desc ? -cmp : cmp;
        });
        return HttpResponse.json({
          data: sorted,
          pagination: {
            page: 1,
            page_size: 25,
            current_page_size: sorted.length,
            total_pages: 1,
            total_results: sorted.length,
          },
        });
      }
    )
  );
};

/** Wait for checkboxes to become enabled, then click one to select it. */
const rowCheckboxAt = (index: number) =>
  screen.getAllByRole('checkbox', { name: /(De)?select row/i })[index];

const selectRow = async (user: ReturnType<typeof userEvent.setup>, index: number) => {
  await waitFor(() => expect(rowCheckboxAt(index)).toBeEnabled());
  if (!(rowCheckboxAt(index) as HTMLInputElement).checked) {
    await user.click(rowCheckboxAt(index));
  }
  await waitFor(() => expect(rowCheckboxAt(index)).toBeChecked());
};

describe('GuardrailsDataView', () => {
  it('renders config names from the API', async () => {
    renderComponent();
    expect(await findPiiFilterRow()).toBeInTheDocument();
    expect(screen.getByText('toxicity-guard')).toBeInTheDocument();
  });

  it('renders the main model name column', async () => {
    renderComponent();
    await findPiiFilterRow();
    // pii-filter's main model is gpt-4; toxicity-guard also uses gpt-4
    expect(screen.getAllByText('gpt-4').length).toBeGreaterThanOrEqual(1);
  });

  it('renders a Flows column with Input/Output badges', async () => {
    renderComponent();
    await findPiiFilterRow();
    expect(screen.getByRole('columnheader', { name: 'Flows' })).toBeInTheDocument();
    // Both configs have input and output flows configured
    expect(screen.getAllByText('Input').length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText('Output').length).toBeGreaterThanOrEqual(2);
  });

  it('calls onRowClick when a row is clicked', async () => {
    const user = userEvent.setup();
    const onRowClick = vi.fn();
    renderComponent({ onRowClick });
    const row = await findPiiFilterRow();
    await user.click(row);
    expect(onRowClick).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'pii-filter' }),
      expect.anything()
    );
  });

  it('shows empty state when there are no configs', async () => {
    const user = userEvent.setup();
    const onCreate = vi.fn();
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`, () =>
        HttpResponse.json({
          data: [],
          pagination: {
            page: 1,
            page_size: 25,
            current_page_size: 0,
            total_pages: 0,
            total_results: 0,
          },
        })
      )
    );
    renderComponent({ onCreate });
    expect(
      await screen.findByText(ENTITY_EMPTY_STATES.guardrails.heading, undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      })
    ).toBeInTheDocument();
    const createButton = screen.getByRole('button', {
      name: ENTITY_EMPTY_STATES.guardrails.createAction?.label,
    });
    expect(createButton).toBeInTheDocument();

    await user.click(createButton);
    expect(onCreate).toHaveBeenCalledTimes(1);
  });

  it('calls onRequestDelete when the Delete row action is selected', async () => {
    const user = userEvent.setup();
    const onRequestDelete = vi.fn();
    renderComponent({ onRequestDelete });
    await findPiiFilterRow();

    const menuButtons = screen.getAllByRole('button', { name: /actions/i });
    await user.click(menuButtons[0]);
    await user.click((await screen.findAllByRole('menuitem', { name: 'Delete' }))[0]);
    await waitFor(() => {
      expect(onRequestDelete).toHaveBeenCalledWith(expect.objectContaining({ name: 'pii-filter' }));
    });
  });

  it('shows error state when the API request fails', async () => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`, () =>
        HttpResponse.error()
      )
    );
    renderComponent();
    expect(
      await screen.findByTestId('error-panel', undefined, { timeout: XL_SELECTOR_TIMEOUT })
    ).toBeInTheDocument();
  });

  describe('sorting', () => {
    it('defaults to sorting by created_at descending', async () => {
      const seenSort: string[] = [];
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`,
          ({ request }) => {
            seenSort.push(new URL(request.url).searchParams.get('sort') ?? '');
            return HttpResponse.json({
              data: [],
              pagination: {
                page: 1,
                page_size: 25,
                current_page_size: 0,
                total_pages: 0,
                total_results: 0,
              },
            });
          }
        )
      );
      renderComponent();
      await waitFor(() => expect(seenSort.length).toBeGreaterThan(0), {
        timeout: XL_SELECTOR_TIMEOUT,
      });
      expect(seenSort[0]).toBe('-created_at');
    });

    it('sends sort=name when the Name column header is clicked', async () => {
      const user = userEvent.setup();
      mockSortedConfigs();
      renderComponent();
      await findPiiFilterRow();

      const nameHeader = screen.getByRole('columnheader', { name: 'Name' });
      await user.click(within(nameHeader).getByRole('button', { name: 'Name' }));

      await waitFor(
        () => {
          const cells = screen
            .getAllByRole('cell')
            .filter((c) => c.textContent === 'pii-filter' || c.textContent === 'toxicity-guard');
          // Alphabetical ascending: pii-filter < toxicity-guard
          expect(cells[0].textContent).toBe('pii-filter');
          expect(cells[1].textContent).toBe('toxicity-guard');
        },
        { timeout: XL_SELECTOR_TIMEOUT }
      );
    });

    it('sends sort=updated_at when the Updated column header is clicked', async () => {
      const user = userEvent.setup();
      const seenSorts: string[] = [];
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`,
          ({ request }) => {
            seenSorts.push(new URL(request.url).searchParams.get('sort') ?? '');
            return HttpResponse.json({
              data: mockGuardrailConfigs,
              pagination: {
                page: 1,
                page_size: 25,
                current_page_size: 2,
                total_pages: 1,
                total_results: 2,
              },
            });
          }
        )
      );
      renderComponent();
      await findPiiFilterRow();

      const updatedHeader = screen.getByRole('columnheader', { name: 'Updated' });
      await user.click(within(updatedHeader).getByRole('button', { name: 'Updated' }));

      await waitFor(
        () => expect(seenSorts.some((s) => s === 'updated_at' || s === '-updated_at')).toBe(true),
        { timeout: XL_SELECTOR_TIMEOUT }
      );
    });
  });

  describe('filter panel', () => {
    it('has a filter toggle button', async () => {
      renderComponent();
      await findPiiFilterRow();
      expect(screen.getByTestId('open-filters-button')).toBeInTheDocument();
    });

    it('shows Updated At and Created At date range filters in the panel', async () => {
      const user = userEvent.setup();
      renderComponent();
      await findPiiFilterRow();

      await user.click(screen.getByTestId('open-filters-button'));

      expect(
        await screen.findByTestId('column-filter-updated_at', undefined, {
          timeout: XL_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();
      expect(screen.getByTestId('column-filter-created_at')).toBeInTheDocument();
    });

    it('shows "No Results Found" when a search matches nothing', async () => {
      const user = userEvent.setup();
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`, () =>
          HttpResponse.json({
            data: [],
            pagination: {
              page: 1,
              page_size: 25,
              current_page_size: 0,
              total_pages: 0,
              total_results: 0,
            },
          })
        )
      );
      renderComponent();
      await user.type(
        await screen.findByPlaceholderText('Search Guardrail Configs...', undefined, {
          timeout: XL_SELECTOR_TIMEOUT,
        }),
        'no-such-config'
      );
      expect(
        await screen.findByTestId('entity-empty-state-no-results', undefined, {
          timeout: XL_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Clear filters/i })).toBeInTheDocument();
    });
  });

  describe('bulk delete', () => {
    it('calls onRequestBulkDelete with the selected configs when Delete is clicked', async () => {
      const user = userEvent.setup();
      const onRequestBulkDelete = vi.fn();
      renderComponent({ onRequestBulkDelete });
      await findPiiFilterRow();

      await selectRow(user, 0);
      await selectRow(user, 1);

      await user.click(screen.getByRole('button', { name: 'Delete selected guardrails' }));

      expect(onRequestBulkDelete).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ name: 'pii-filter' }),
          expect.objectContaining({ name: 'toxicity-guard' }),
        ])
      );
    });

    it('clears row selection after Delete is clicked', async () => {
      const user = userEvent.setup();
      renderComponent({ onRequestBulkDelete: vi.fn() });
      await findPiiFilterRow();

      await selectRow(user, 0);

      await user.click(screen.getByRole('button', { name: 'Delete selected guardrails' }));

      await waitFor(() => {
        const checkboxes = screen.queryAllByRole('checkbox', { name: /(De)?select row/i });
        checkboxes.forEach((cb) => expect(cb).not.toBeChecked());
      });
    });
  });
});
