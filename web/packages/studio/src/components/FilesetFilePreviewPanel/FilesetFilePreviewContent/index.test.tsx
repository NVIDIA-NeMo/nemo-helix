// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetFilePreviewContent } from '@studio/components/FilesetFilePreviewPanel/FilesetFilePreviewContent';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { fireEvent, render, screen } from '@testing-library/react';

vi.mock('@studio/providers/workers/useWorkers', () => ({
  useWorkers: () => ({ createWorker: vi.fn() }),
}));

vi.mock('@studio/components/filesets/hooks/useIsBinaryFile', () => ({
  useIsBinaryFile: () => ({ isBinary: false, isLoading: false }),
}));

vi.mock('@studio/components/FilesetFilePreviewPanel/FilesetImagePreview', () => ({
  FilesetImagePreview: ({ filePath }: { filePath: string }) => <img alt={filePath} />,
}));

const baseProps = {
  workspace: 'default',
  filesetName: 'test-dataset',
  filePath: 'folder/data.json',
};

describe('FilesetFilePreviewContent', () => {
  it('renders breadcrumbs + editor in the inline (default) mode', async () => {
    render(
      <TestProviders>
        <FilesetFilePreviewContent
          {...baseProps}
          fileContent='{"key": "value"}'
          isLoading={false}
          file={{ type: 'file', path: 'folder/data.json', size: 100, oid: 'oid-1' }}
        />
      </TestProviders>
    );
    // Breadcrumbs render with fileset name, folder, and file segments.
    expect(screen.getByText('test-dataset')).toBeInTheDocument();
    expect(screen.getByText('folder')).toBeInTheDocument();
    expect(screen.getByText('data.json')).toBeInTheDocument();
    // Editor present.
    expect(await screen.findByTestId('nv-code-editor-root')).toBeInTheDocument();
  });

  it('hides the inline header when hideHeader is true (for SidePanel-wrapper hosts)', () => {
    render(
      <TestProviders>
        <FilesetFilePreviewContent
          {...baseProps}
          fileContent="content"
          isLoading={false}
          hideHeader
        />
      </TestProviders>
    );
    // Header content does NOT render: no fileset name, no folder, no filename anywhere.
    expect(screen.queryByText('test-dataset')).toBeNull();
    expect(screen.queryByText('folder')).toBeNull();
    expect(screen.queryByText('data.json')).toBeNull();
  });

  it('keeps the breadcrumbs but drops the file actions menu when hideFileActions is true', () => {
    const file = { type: 'file' as const, path: 'folder/data.json', size: 100, oid: 'oid-1' };
    const { rerender } = render(
      <TestProviders>
        <FilesetFilePreviewContent
          {...baseProps}
          fileContent="content"
          isLoading={false}
          file={file}
        />
      </TestProviders>
    );
    expect(screen.getByRole('button', { name: 'Open file actions menu' })).toBeInTheDocument();

    rerender(
      <TestProviders>
        <FilesetFilePreviewContent
          {...baseProps}
          fileContent="content"
          isLoading={false}
          file={file}
          hideFileActions
        />
      </TestProviders>
    );
    expect(screen.getByText('data.json')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open file actions menu' })).toBeNull();
  });

  it('shows the loading state', () => {
    render(
      <TestProviders>
        <FilesetFilePreviewContent {...baseProps} isLoading />
      </TestProviders>
    );
    // Loading + error UI now lives inside FileContentPreview (the spinner has aria-label="Loading...").
    expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
  });

  it('shows the error state', () => {
    render(
      <TestProviders>
        <FilesetFilePreviewContent {...baseProps} isLoading={false} error={new Error('boom')} />
      </TestProviders>
    );
    expect(screen.getByText('Error: boom')).toBeInTheDocument();
  });

  it('uses the image preview instead of the text editor for image files', () => {
    render(
      <TestProviders>
        <FilesetFilePreviewContent {...baseProps} filePath="folder/chart.png" />
      </TestProviders>
    );

    expect(screen.getByRole('img', { name: 'folder/chart.png' })).toBeInTheDocument();
    expect(screen.queryByTestId('nv-code-editor-root')).not.toBeInTheDocument();
  });

  it('invokes onFolderClick with the cumulative folder path', () => {
    const onFolderClick = vi.fn();
    render(
      <TestProviders>
        <FilesetFilePreviewContent
          {...baseProps}
          filePath="folder1/folder2/file.txt"
          fileContent=""
          isLoading={false}
          onFolderClick={onFolderClick}
        />
      </TestProviders>
    );
    fireEvent.click(screen.getByText('folder2'));
    expect(onFolderClick).toHaveBeenCalledWith('folder1/folder2');
  });

  it('invokes onFilesetClick on fileset breadcrumb click', () => {
    const onFilesetClick = vi.fn();
    render(
      <TestProviders>
        <FilesetFilePreviewContent
          {...baseProps}
          fileContent=""
          isLoading={false}
          onFilesetClick={onFilesetClick}
        />
      </TestProviders>
    );
    fireEvent.click(screen.getByText('test-dataset'));
    expect(onFilesetClick).toHaveBeenCalledTimes(1);
  });
});
