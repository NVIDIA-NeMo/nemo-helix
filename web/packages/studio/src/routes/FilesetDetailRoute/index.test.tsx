// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import { ROUTES } from '@studio/constants/routes';
import { FilesetDetailRoute } from '@studio/routes/FilesetDetailRoute';
import { FilesetDetailTabContent } from '@studio/routes/FilesetDetailRoute/FilesetDetailTabContent';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render as rtlRender, act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router';

// The fileset returned by the route's fetch. Mutated per-test to exercise the
// purpose-dependent branches (Dataset Card vs Model Card).
let mockFileset: FilesetOutput;

const makeFileset = (purpose: FilesetPurpose): FilesetOutput =>
  ({
    name: 'test-fileset',
    namespace: 'default',
    workspace: 'default',
    purpose,
    storage: { type: 'internal' },
    metadata: {},
  }) as unknown as FilesetOutput;

// Override only the two data hooks the route reads; keep every other SDK
// export intact so the file explorer's own hooks still resolve.
vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  useFilesRetrieveFileset: () => ({ data: mockFileset, isPending: false, isError: false }),
  useFilesListFilesetFiles: () => ({
    data: { data: [] },
    isPending: false,
    isFetching: false,
    isError: false,
  }),
}));

// Stub the file preview and the dataset schema editor — both have their own
// specs and need handlers this route-level spec shouldn't care about.
vi.mock('@studio/components/FilesetFilePreviewPanel/FilesetFilePreviewContent', () => ({
  FilesetFilePreviewContent: ({ filePath }: { filePath: string }) => (
    <div data-testid="mock-file-preview-content">{filePath}</div>
  ),
}));

vi.mock('@studio/routes/FilesetDetailRoute/DatasetSchemaEditor', () => ({
  DatasetSchemaEditor: () => <div data-testid="mock-dataset-schema-editor" />,
}));

vi.mock('@studio/providers/workers/useWorkers', () => ({
  useWorkers: () => ({
    createWorker: vi.fn(),
  }),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'default',
}));

vi.mock('@studio/util/hooks/useRequiredPathParams', () => ({
  useRequiredPathParams: () => ({ filesetName: 'test-fileset' }),
}));

const detailUrl = (tab?: string) =>
  `/workspaces/default/filesets/test-fileset${tab ? `/${tab}` : ''}`;

const openAtUrl = (url: string) => {
  const router = createMemoryRouter(
    [
      {
        path: ROUTES.workspace.filesetDetail,
        element: <FilesetDetailRoute />,
        children: [
          { path: ROUTES.workspace.filesetDetailTab, element: <FilesetDetailTabContent /> },
        ],
      },
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

// The empty Files tab's self-service help renders a CodeSnippet whose syntax
// highlighting resolves asynchronously after mount. Flush it so the state
// update lands inside act() instead of leaking past the test.
const flushFilesTab = () =>
  act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

describe('FilesetDetailRoute', () => {
  beforeEach(() => {
    mockFileset = makeFileset(FilesetPurpose.dataset);
  });

  it('redirects a dataset with no tab in the path to its card', async () => {
    const router = openAtUrl(detailUrl());

    expect(await screen.findByTestId('fileset-card')).toBeInTheDocument();
    expect(router.state.location.pathname).toBe(detailUrl('card'));
  });

  it('redirects an unknown tab to the default tab', async () => {
    const router = openAtUrl(detailUrl('garbage'));

    expect(await screen.findByTestId('fileset-card')).toBeInTheDocument();
    expect(router.state.location.pathname).toBe(detailUrl('card'));
  });

  it('labels the card tab by purpose — "Dataset Card" for a dataset fileset', async () => {
    openAtUrl(detailUrl('card'));

    expect(await screen.findByRole('tab', { name: 'Dataset Card' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'false');
  });

  it('labels the card tab "Model Card" for a model fileset and renders the shared card', async () => {
    mockFileset = makeFileset(FilesetPurpose.model);
    openAtUrl(detailUrl('card'));

    expect(await screen.findByRole('tab', { name: 'Model Card' })).toBeInTheDocument();
    expect(screen.getByTestId('fileset-card')).toBeInTheDocument();
  });

  it('navigates to the Files path when the Files tab is clicked', async () => {
    const user = userEvent.setup();
    const router = openAtUrl(detailUrl('card'));

    await user.click(await screen.findByRole('tab', { name: 'Files' }));

    expect(router.state.location.pathname).toBe(detailUrl('files'));
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('fileset-files-tab')).toBeInTheDocument();
    await flushFilesTab();
  });

  it('opens the Files tab from the path', async () => {
    openAtUrl(detailUrl('files'));

    expect(await screen.findByTestId('fileset-files-tab')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    await flushFilesTab();
  });

  it('renders the file preview when ?file= is set on the Files tab', async () => {
    openAtUrl(`${detailUrl('files')}?file=folder%2Fdata.json`);

    expect(await screen.findByTestId('fileset-files-tab-preview')).toBeInTheDocument();
    expect(screen.getByTestId('mock-file-preview-content')).toHaveTextContent('folder/data.json');
  });

  it('renders the explorer when no file is selected on the Files tab', async () => {
    openAtUrl(detailUrl('files'));

    expect(await screen.findByTestId('fileset-files-tab')).toBeInTheDocument();
    expect(screen.queryByTestId('fileset-files-tab-preview')).toBeNull();
    expect(screen.queryByTestId('mock-file-preview-content')).toBeNull();
    await flushFilesTab();
  });
});
