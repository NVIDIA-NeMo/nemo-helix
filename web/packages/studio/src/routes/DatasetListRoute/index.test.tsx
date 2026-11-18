// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPurpose, type FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { DatasetListRoute } from '@studio/routes/DatasetListRoute';
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { MemoryRouter } from 'react-router';

vi.mock('@studio/providers/auth/useAuthProfile', () => ({ useAuthProfile: () => undefined }));

const dataset: FilesetOutput = {
  id: 'qa-data-id',
  name: 'qa-data',
  workspace: 'default',
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: '/data/qa-data' },
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2024-12-17T16:08:56.880768',
  updated_at: '2024-12-17T16:08:56.880771',
};

describe('DatasetListRoute', () => {
  let lastUrl: URL | undefined;

  beforeEach(() => {
    mockUseParams({ [ROUTE_PARAMS.workspace]: workspace1.workspace });
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
        ({ request }) => {
          lastUrl = new URL(request.url);
          return HttpResponse.json({
            object: 'list',
            data: [dataset],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
            sort: '-created_at',
          });
        }
      )
    );
  });

  it('lists dataset filesets under a Datasets header with a Create Dataset action', async () => {
    render(
      <TestProviders>
        <MemoryRouter>
          <DatasetListRoute />
        </MemoryRouter>
      </TestProviders>
    );

    expect(
      await screen.findByTestId('nv-page-header-heading', undefined, {
        timeout: LG_SELECTOR_TIMEOUT,
      })
    ).toHaveTextContent('Datasets');
    expect(screen.getByRole('button', { name: 'Create Dataset' })).toBeInTheDocument();

    await screen.findByText('qa-data', undefined, { timeout: LG_SELECTOR_TIMEOUT });
    expect(lastUrl?.searchParams.get('filter[purpose]')).toBe(FilesetPurpose.dataset);
  });

  it('opens a dataset on its detail page under /datasets', async () => {
    const user = userEvent.setup();
    const navigate = vi.fn();
    mockUseNavigate(navigate);
    render(
      <TestProviders>
        <MemoryRouter>
          <DatasetListRoute />
        </MemoryRouter>
      </TestProviders>
    );

    await user.click(
      await screen.findByText('qa-data', undefined, { timeout: LG_SELECTOR_TIMEOUT })
    );
    expect(navigate).toHaveBeenCalledWith(`/workspaces/${workspace1.workspace}/datasets/qa-data`);
  });
});
