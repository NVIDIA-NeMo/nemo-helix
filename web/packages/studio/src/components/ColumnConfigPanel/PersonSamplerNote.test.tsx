// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { getFilesRetrieveFilesetQueryKey } from '@nemo/sdk/generated/platform/files';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { DataDesignerJobBuildRoute } from '@studio/routes/DataDesignerJobBuildRoute';
import { getDataDesignerJobBuildRoute } from '@studio/routes/utils';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { fireEvent, render, screen } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { createMemoryRouter, RouterProvider } from 'react-router';

const personJob = (locale?: string) =>
  ({
    name: 'personas',
    spec: {
      num_records: 10,
      config: {
        columns: [
          {
            name: 'person',
            column_type: 'sampler',
            sampler_type: 'person',
            params: locale ? { locale } : {},
          },
        ],
        model_configs: [],
      },
    },
  }) as unknown as CreateJobRequest;

const mockRegisteredLocales = (...filesets: string[]) =>
  server.use(
    http.get(mockApiUrl(getFilesRetrieveFilesetQueryKey, 'system', ':name'), ({ params }) =>
      filesets.includes(String(params.name))
        ? HttpResponse.json({ name: params.name, workspace: 'system' })
        : HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    )
  );

const openPersonColumn = async (locale?: string) => {
  const router = createMemoryRouter(
    [{ path: ROUTES.workspace.dataDesignerJobBuild, element: <DataDesignerJobBuildRoute /> }],
    {
      initialEntries: [
        {
          pathname: getDataDesignerJobBuildRoute('default'),
          state: { generatedJobRequest: personJob(locale) },
        },
      ],
    }
  );
  render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
  fireEvent.click(await screen.findByText('person'));
  await screen.findByRole('complementary', { name: /Configure .* column/ });
};

describe('PersonSamplerNote', () => {
  it('offers to register the dataset when the locale is missing', async () => {
    mockRegisteredLocales();
    await openPersonColumn('ja_JP');

    expect(await screen.findByText("Nemotron Personas for ja_JP isn't set up")).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Register dataset' })).toBeDisabled();
  });

  it('stays hidden once the default en_US dataset is registered', async () => {
    mockRegisteredLocales('nemotron-personas-dataset-en_us');
    await openPersonColumn();

    await screen.findByRole('textbox', { name: /Column name/ });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByText(/Nemotron Personas/)).not.toBeInTheDocument();
  });

  it('names the supported locales for one without a dataset', async () => {
    await openPersonColumn('es_ES');

    expect(await screen.findByText('No Nemotron Personas dataset for es_ES')).toBeInTheDocument();
  });
});
