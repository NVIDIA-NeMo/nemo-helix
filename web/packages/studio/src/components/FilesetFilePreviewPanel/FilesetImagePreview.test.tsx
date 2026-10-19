// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as platformApi from '@nemo/sdk/generated/platform/files';
import { FilesetImagePreview } from '@studio/components/FilesetFilePreviewPanel/FilesetImagePreview';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { fireEvent, render, screen } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>();
  return { ...actual, useFilesDownloadFile: vi.fn() };
});

const mockCreateObjectURL = vi.fn();
const mockRevokeObjectURL = vi.fn();

describe('FilesetImagePreview', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreateObjectURL.mockReturnValue('blob:preview-image');
    global.URL.createObjectURL = mockCreateObjectURL;
    global.URL.revokeObjectURL = mockRevokeObjectURL;
    vi.mocked(platformApi.useFilesDownloadFile).mockReturnValue({
      data: 'blob:preview-image',
      error: null,
      isLoading: false,
    } as ReturnType<typeof platformApi.useFilesDownloadFile>);
  });

  it('downloads an image and renders it from an object URL', async () => {
    render(
      <TestProviders>
        <FilesetImagePreview
          workspace="default"
          filesetName="images"
          filePath="examples/chart.png"
          enabled
        />
      </TestProviders>
    );

    const image = await screen.findByRole('img', { name: 'chart.png' });
    expect(image).toHaveAttribute('src', 'blob:preview-image');
    expect(platformApi.useFilesDownloadFile).toHaveBeenCalledWith(
      'default',
      'images',
      'examples/chart.png',
      { query: { enabled: true, select: URL.createObjectURL } }
    );

    fireEvent.load(image);
    expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:preview-image');
  });

  it('shows the download error', () => {
    vi.mocked(platformApi.useFilesDownloadFile).mockReturnValue({
      data: undefined,
      error: new Error('Unable to load image'),
      isLoading: false,
    } as ReturnType<typeof platformApi.useFilesDownloadFile>);

    render(
      <TestProviders>
        <FilesetImagePreview
          workspace="default"
          filesetName="images"
          filePath="examples/chart.png"
          enabled
        />
      </TestProviders>
    );

    expect(screen.getByText('Error: Unable to load image')).toBeInTheDocument();
  });
});
