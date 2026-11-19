// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type FilesetOutput, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { ROUTES } from '@studio/constants/routes';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { DatasetDetailRoute } from '@studio/routes/DatasetDetailRoute';
import { useFilesetDetailParent } from '@studio/routes/FilesetDetailRoute/FilesetDetailParent';
import { FilesetDetailTabContent } from '@studio/routes/FilesetDetailRoute/FilesetDetailTabContent';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render as rtlRender, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

const makeFileset = (purpose: FilesetPurpose) =>
  ({
    name: 'qa-data',
    workspace: 'default',
    purpose,
    storage: { type: 'internal' },
    metadata: {},
  }) as unknown as FilesetOutput;

let fileset = makeFileset(FilesetPurpose.dataset);

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  useFilesRetrieveFileset: () => ({ data: fileset, isPending: false, isError: false }),
  useFilesListFilesetFiles: () => ({
    data: { data: [] },
    isPending: false,
    isFetching: false,
    isError: false,
  }),
}));

vi.mock('@studio/providers/breadcrumbs/useBreadcrumbs', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/providers/breadcrumbs/useBreadcrumbs')>()),
  useBreadcrumbs: vi.fn(),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'default',
}));

vi.mock('@studio/util/hooks/useRequiredPathParams', () => ({
  useRequiredPathParams: () => ({ filesetName: 'qa-data' }),
}));

// Reads the detail-page parent the way the file explorer's links do; stands in for the
// default Dataset Card tab.
const ParentProbe = () => {
  const { getDetailRoute } = useFilesetDetailParent();
  return <span data-testid="parent-probe">{getDetailRoute('default', 'qa-data')}</span>;
};

vi.mock('@studio/routes/FilesetDetailRoute/FilesetCard', () => ({
  FilesetCard: () => <ParentProbe />,
}));

const openAtUrl = (url: string) => {
  const router = createMemoryRouter(
    [
      {
        path: ROUTES.workspace.datasetDetail,
        element: <DatasetDetailRoute />,
        children: [
          { path: ROUTES.workspace.datasetDetailTab, element: <FilesetDetailTabContent /> },
        ],
      },
      { path: ROUTES.workspace.filesetDetailTab, element: <div data-testid="fileset-detail" /> },
    ],
    { initialEntries: [url] }
  );
  rtlRender(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
  return router;
};

describe('DatasetDetailRoute', () => {
  beforeEach(() => {
    fileset = makeFileset(FilesetPurpose.dataset);
  });

  it('sends a fileset that is not a dataset to its Filesets page', async () => {
    fileset = makeFileset(FilesetPurpose.model);
    const router = openAtUrl('/workspaces/default/datasets/qa-data/files');

    expect(await screen.findByTestId('fileset-detail')).toBeInTheDocument();
    expect(router.state.location.pathname).toBe('/workspaces/default/filesets/qa-data/files');
  });

  it('keeps the file and folder selection when sending a non-dataset to Filesets', async () => {
    fileset = makeFileset(FilesetPurpose.model);
    const router = openAtUrl(
      '/workspaces/default/datasets/qa-data?filesetFolder=docs&file=docs/a.md'
    );

    expect(await screen.findByTestId('fileset-detail')).toBeInTheDocument();
    expect(router.state.location.pathname).toBe('/workspaces/default/filesets/qa-data/files');
    expect(Object.fromEntries(new URLSearchParams(router.state.location.search))).toEqual({
      filesetFolder: 'docs',
      file: 'docs/a.md',
    });
  });

  it('puts the fileset detail page under the Datasets breadcrumb', async () => {
    openAtUrl('/workspaces/default/datasets/qa-data/card');

    expect(await screen.findByTestId('parent-probe')).toBeInTheDocument();
    expect(screen.getByText('qa-data')).toBeInTheDocument();
    expect(useBreadcrumbs).toHaveBeenCalledWith({
      items: [
        { href: '/workspaces/default/datasets', slotLabel: 'Datasets' },
        { slotLabel: 'qa-data' },
      ],
    });
  });

  it('redirects to the default tab under /datasets', async () => {
    const router = openAtUrl('/workspaces/default/datasets/qa-data');

    expect(await screen.findByTestId('parent-probe')).toBeInTheDocument();
    expect(router.state.location.pathname).toBe('/workspaces/default/datasets/qa-data/card');
  });

  it('keeps tab links and links the page builds to itself under /datasets', async () => {
    openAtUrl('/workspaces/default/datasets/qa-data/card');

    expect(await screen.findByTestId('parent-probe')).toHaveTextContent(
      '/workspaces/default/datasets/qa-data'
    );
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute(
      'href',
      '/workspaces/default/datasets/qa-data/files'
    );
  });
});
