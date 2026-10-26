// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useFilesListFilesetFiles,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FilesTab } from '@studio/routes/ModelDetailRoute/FilesTab';
import { render } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  useFilesRetrieveFileset: vi.fn(),
  useFilesListFilesetFiles: vi.fn(),
}));
vi.mock('@studio/routes/FilesetDetailRoute/FilesTab', () => ({
  FilesTab: ({ workspace, filesetId }: { workspace: string; filesetId: string }) => (
    <div>
      explorer {workspace} {filesetId}
    </div>
  ),
}));

const fileset = { workspace: 'files-ws', name: 'weights' };

const mockQueries = (
  filesetQuery: Partial<ReturnType<typeof useFilesRetrieveFileset>>,
  filesQuery: Partial<ReturnType<typeof useFilesListFilesetFiles>> = {
    isPending: false,
    isError: false,
  }
) => {
  vi.mocked(useFilesRetrieveFileset).mockReturnValue(
    filesetQuery as ReturnType<typeof useFilesRetrieveFileset>
  );
  vi.mocked(useFilesListFilesetFiles).mockReturnValue(
    filesQuery as ReturnType<typeof useFilesListFilesetFiles>
  );
};

describe('ModelDetailRoute FilesTab', () => {
  it('shows an empty state when the model has no fileset', () => {
    mockQueries({ isPending: true, isError: false });
    render(<FilesTab fileset={null} />);

    expect(screen.getByText('No files')).toBeInTheDocument();
    expect(useFilesRetrieveFileset).toHaveBeenCalledWith(
      '',
      '',
      expect.objectContaining({ query: expect.objectContaining({ enabled: false }) })
    );
  });

  it('loads while the fileset is fetched', () => {
    mockQueries({ isPending: true, isError: false });
    render(<FilesTab fileset={fileset} />);

    expect(screen.getByText('Loading files...')).toBeInTheDocument();
  });

  it('reports a fileset that cannot be loaded', () => {
    mockQueries({ data: undefined, isPending: false, isError: true });
    render(<FilesTab fileset={fileset} />);

    expect(screen.getByText('Fileset not found')).toBeInTheDocument();
    expect(screen.getByText(/files-ws\/weights/)).toBeInTheDocument();
  });

  it('browses the fileset in its own workspace', () => {
    mockQueries({ data: { name: 'weights' } as FilesetOutput, isPending: false, isError: false });
    render(<FilesTab fileset={fileset} />);

    expect(screen.getByText('explorer files-ws files-ws/weights')).toBeInTheDocument();
  });
});
