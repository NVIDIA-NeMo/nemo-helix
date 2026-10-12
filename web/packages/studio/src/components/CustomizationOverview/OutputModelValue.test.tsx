// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { OutputModelValue } from '@studio/components/CustomizationOverview/OutputModelValue';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const MODEL_URL = `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models/:name`;

const renderValue = (
  modelName: string | undefined,
  jobStatus: HelixJobStatus | undefined = HelixJobStatus.active
) =>
  render(
    <TestProviders>
      <OutputModelValue workspace="ws" modelName={modelName} jobStatus={jobStatus} />
    </TestProviders>
  );

describe('OutputModelValue', () => {
  it('says the model is still to come while the run has not registered it', async () => {
    server.use(http.get(MODEL_URL, () => new HttpResponse(null, { status: 404 })));

    renderValue('bird-sql-003');

    expect(await screen.findByText('Registered when this run completes')).toBeInTheDocument();
    // Nothing to link to yet, so the name is plain text.
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('links to the model entry once it exists', async () => {
    server.use(
      http.get(MODEL_URL, () =>
        HttpResponse.json({
          id: 'm1',
          name: 'bird-sql-003',
          workspace: 'ws',
          created_at: '2026-10-06T00:00:00Z',
          updated_at: '2026-10-06T00:00:00Z',
          api_endpoint: { url: 'https://example.invalid/v1' },
        })
      )
    );

    renderValue('bird-sql-003');

    const link = await screen.findByRole('link', { name: 'bird-sql-003' });
    expect(link).toHaveAttribute('href', '/workspaces/ws/fine-tune?model=bird-sql-003');
    expect(await screen.findByText(/open it to (test|deploy|follow)/)).toBeInTheDocument();
  });

  it('renders a placeholder when the run declares no output model', () => {
    renderValue(undefined);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  it.each([HelixJobStatus.error, HelixJobStatus.cancelled])(
    'does not promise a model for a %s run',
    async (status) => {
      server.use(http.get(MODEL_URL, () => new HttpResponse(null, { status: 404 })));

      renderValue('bird-sql-003', status);

      // A run that ended without success never reaches the step that registers the model,
      // so "Registered when this run completes" would be waiting for something impossible.
      expect(
        await screen.findByText('No model was registered — this run did not finish successfully')
      ).toBeInTheDocument();
      expect(screen.queryByText('Registered when this run completes')).not.toBeInTheDocument();
    }
  );
});
