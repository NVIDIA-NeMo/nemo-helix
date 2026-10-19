// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetOutput, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { FilesetDetailRoute } from '@studio/routes/FilesetDetailRoute';
import { render } from '@studio/tests/util/render';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render as rtlRender, act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';

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

describe('FilesetDetailRoute', () => {
  beforeEach(() => {
    mockFileset = makeFileset(FilesetPurpose.dataset);
  });

  it('labels the card tab by purpose — "Dataset Card" for a dataset fileset', () => {
    render(<FilesetDetailRoute />);

    expect(screen.getByRole('tab', { name: 'Dataset Card' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByTestId('fileset-card')).toBeInTheDocument();
  });

  it('labels the card tab "Model Card" for a model fileset and renders the shared card', async () => {
    const user = userEvent.setup();
    mockFileset = makeFileset(FilesetPurpose.model);
    render(<FilesetDetailRoute />);

    const modelCardTab = screen.getByRole('tab', { name: 'Model Card' });
    expect(modelCardTab).toBeInTheDocument();

    // Both purposes share FilesetCard; selecting the card tab mounts it.
    await user.click(modelCardTab);
    expect(screen.getByTestId('fileset-card')).toBeInTheDocument();
  });

  it('switches to the Files tab when clicked and renders the file explorer', async () => {
    const user = userEvent.setup();
    render(<FilesetDetailRoute />);

    await user.click(screen.getByRole('tab', { name: 'Files' }));

    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('fileset-files-tab')).toBeInTheDocument();
  });

  const renderAtUrl = (url: string) =>
    rtlRender(
      <TestProviders>
        <MemoryRouter initialEntries={[url]}>
          <FilesetDetailRoute />
        </MemoryRouter>
      </TestProviders>
    );

  it('opens the Files tab when the initial URL has ?tab=files', async () => {
    renderAtUrl('/?tab=files');

    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('fileset-files-tab')).toBeInTheDocument();

    // The empty Files tab's self-service help renders a CodeSnippet whose
    // syntax highlighting resolves asynchronously after mount. Flush it here
    // so the state update lands inside act() instead of leaking past this
    // test's synchronous assertions.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  });

  it('falls back to the default tab when ?tab= is an unknown value', () => {
    renderAtUrl('/?tab=garbage');

    expect(screen.getByRole('tab', { name: 'Dataset Card' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
  });

  it('renders the file preview when ?file= is set on the Files tab', () => {
    renderAtUrl('/?tab=files&file=folder%2Fdata.json');

    expect(screen.getByTestId('fileset-files-tab-preview')).toBeInTheDocument();
    expect(screen.getByTestId('mock-file-preview-content')).toHaveTextContent('folder/data.json');
  });

  it('renders the explorer when no file is selected on the Files tab', async () => {
    const user = userEvent.setup();
    render(<FilesetDetailRoute />);
    await user.click(screen.getByRole('tab', { name: 'Files' }));

    expect(screen.queryByTestId('fileset-files-tab-preview')).toBeNull();
    expect(screen.queryByTestId('mock-file-preview-content')).toBeNull();
  });
});
