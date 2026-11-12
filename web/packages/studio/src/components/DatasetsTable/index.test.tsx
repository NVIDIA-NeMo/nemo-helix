// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import { DatasetsTable } from '@studio/components/DatasetsTable';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import type { ComponentProps } from 'react';
import { MemoryRouter } from 'react-router';

vi.mock('@studio/providers/auth/useAuthProfile', () => ({ useAuthProfile: () => undefined }));

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, () => {}],
}));

const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;

const makeDataset = (name: string): FilesetOutput => ({
  id: `${name}-id`,
  name,
  workspace: 'default',
  description: `Description for ${name}`,
  purpose: 'dataset',
  storage: { type: 'local', path: `/data/${name}` },
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2024-12-17T16:08:56.880768',
  updated_at: '2024-12-17T16:08:56.880771',
});

const installListHandler = (datasets: FilesetOutput[] = [makeDataset('dataset-1')]) => {
  const state = { lastUrl: null as URL | null };
  server.use(
    http.get(FILESETS_URL, ({ request }) => {
      state.lastUrl = new URL(request.url);
      return HttpResponse.json({
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
  beforeEach(() => {
    mockUseParams({ [ROUTE_PARAMS.workspace]: workspace1.workspace });
  });

  it('fetches only dataset filesets, without a Purpose filter', async () => {
    const state = installListHandler();
    renderTable();

    await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
    expect(state.lastUrl?.searchParams.get('filter[purpose]')).toBe(FilesetPurpose.dataset);
    expect(state.lastUrl?.searchParams.has('filter[purpose][$in]')).toBe(false);
  });

  it('renders Name, Storage Type, and Created, each sortable', async () => {
    const user = userEvent.setup();
    const state = installListHandler();
    renderTable();

    await screen.findByText('dataset-1', undefined, { timeout: LG_SELECTOR_TIMEOUT });
    for (const header of ['Purpose', 'Path', 'Description', 'Created By']) {
      expect(screen.queryByText(header)).not.toBeInTheDocument();
    }

    for (const [header, sort] of [
      ['Name', 'name'],
      ['Storage Type', 'storage_type'],
      ['Created', 'created_at'],
    ]) {
      await user.click(screen.getByRole('button', { name: new RegExp(`^${header}`) }));
      await waitFor(
        () => {
          expect(state.lastUrl?.searchParams.get('sort')).toBe(sort);
        },
        { timeout: LG_SELECTOR_TIMEOUT }
      );
    }
  });

  it('renders the row actions it is given', async () => {
    installListHandler();
    renderTable({
      renderRowActions: (dataset) => <button type="button">Actions for {dataset.name}</button>,
    });

    expect(
      await screen.findByRole(
        'button',
        { name: 'Actions for dataset-1' },
        {
          timeout: LG_SELECTOR_TIMEOUT,
        }
      )
    ).toBeInTheDocument();
  });

  it('uses dataset wording for search and the first-use empty state', async () => {
    installListHandler([]);
    renderTable();

    expect(
      await screen.findByText(ENTITY_EMPTY_STATES.datasets.heading, undefined, {
        timeout: LG_SELECTOR_TIMEOUT,
      })
    ).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search datasets...')).toBeInTheDocument();
  });

  it('opens the dataset creation panel from the first-use empty state', async () => {
    const user = userEvent.setup();
    installListHandler([]);
    renderTable();

    await user.click(
      await screen.findByRole(
        'button',
        { name: 'Create Dataset' },
        { timeout: LG_SELECTOR_TIMEOUT }
      )
    );

    expect(await screen.findByRole('tab', { name: 'HuggingFace' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Upload' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() =>
      expect(screen.queryByRole('tab', { name: 'HuggingFace' })).not.toBeInTheDocument()
    );
  });
});
