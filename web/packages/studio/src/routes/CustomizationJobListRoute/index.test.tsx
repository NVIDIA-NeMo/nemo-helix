// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { CustomizationJobListRoute } from '@studio/routes/CustomizationJobListRoute';
import { getCustomizationJobListRoute } from '@studio/routes/utils';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const MODEL_URL = `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models/:name`;

const outputModel = {
  id: 'm1',
  name: 'bird-sql-003',
  workspace: 'ws',
  created_at: '2026-10-06T00:00:00Z',
  updated_at: '2026-10-06T00:00:00Z',
  api_endpoint: { url: 'https://example.invalid/v1' },
};

vi.mock('@nemo/common/src/components/DataView/StudioDataView', async () => {
  const { studioDataViewMock } = await import('@studio/tests/util');
  return studioDataViewMock();
});

const renderAt = (path: string) =>
  renderRoute(<CustomizationJobListRoute />, {
    history: path,
    routes: [
      { path: ROUTES.workspace.customizationJobList, element: <CustomizationJobListRoute /> },
    ],
  });

describe('CustomizationJobListRoute', () => {
  it('opens the panel for the model named in the url, so a job can link to its output', async () => {
    server.use(http.get(MODEL_URL, () => HttpResponse.json(outputModel)));

    renderAt(getCustomizationJobListRoute('ws', { model: 'bird-sql-003' }));

    // The panel is the only place the model's own details render.
    expect(await screen.findByRole('dialog')).toHaveTextContent('bird-sql-003');
  });

  it('leaves the panel closed without the parameter', async () => {
    server.use(http.get(MODEL_URL, () => new HttpResponse(null, { status: 404 })));

    renderAt(getCustomizationJobListRoute('ws'));

    expect(await screen.findByText('Fine-tuning')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});
