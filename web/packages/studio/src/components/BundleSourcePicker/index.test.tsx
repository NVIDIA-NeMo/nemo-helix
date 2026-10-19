// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { BundleSourcePicker } from '@studio/components/BundleSourcePicker';
import type { BundleFileSpec } from '@studio/components/BundleSourcePicker/types';
import { useBundleSource } from '@studio/components/BundleSourcePicker/useBundleSource';
import type { DownloadFileAsArrayBufferArgs } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { render, screen, waitFor } from '@studio/tests/util/render';
import { fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const filesetContents = new Map<string, string>();

vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer', () => ({
  useFetchFileAsArrayBuffer:
    () =>
    async ({ path }: DownloadFileAsArrayBufferArgs) =>
      new TextEncoder().encode(filesetContents.get(path) ?? '').buffer,
}));

interface Manifest {
  assets: string[];
}

/** A made-up driving file, to show the picker knows nothing about optimize configs. */
const MANIFEST_SPEC: BundleFileSpec<Manifest> = {
  label: 'manifest',
  description: 'a JSON file with an assets list',
  candidateNoun: 'JSON files',
  isCandidate: (path) => path.endsWith('.json'),
  parse: (text) => {
    try {
      const parsed: unknown = JSON.parse(text);
      return parsed && typeof parsed === 'object' && 'assets' in parsed
        ? (parsed as Manifest)
        : undefined;
    } catch {
      return undefined;
    }
  },
  validate: (manifest, bundlePaths) =>
    manifest.assets
      .filter((asset) => !bundlePaths.has(asset))
      .map((asset) => `${asset} is missing`),
};

const Harness = () => {
  const state = useBundleSource({ workspace: DEFAULT_WORKSPACE, spec: MANIFEST_SPEC });
  return (
    <>
      <BundleSourcePicker workspace={DEFAULT_WORKSPACE} state={state} title="Asset bundle" />
      <output data-testid="selection">{JSON.stringify(state.active.selection ?? null)}</output>
    </>
  );
};

const makeFile = (relativePath: string, contents: string): File => {
  const file = new File([contents], relativePath.split('/').pop() ?? relativePath);
  Object.defineProperty(file, 'webkitRelativePath', { value: relativePath });
  return file;
};

const mockFileset = (files: Record<string, string>) => {
  filesetContents.clear();
  Object.entries(files).forEach(([path, contents]) => filesetContents.set(path, contents));
  server.use(
    http.get(mockApiUrl(getFilesListFilesetsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [{ name: 'assets', workspace: DEFAULT_WORKSPACE }],
        pagination: { total: 1, page: 1, page_size: 20 },
      })
    ),
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({
        data: Object.keys(files).map((path) => ({
          file_ref: `${DEFAULT_WORKSPACE}/assets#${path}`,
          file_url: `/${path}`,
          path,
          size: 10,
        })),
      })
    )
  );
};

const selection = () => JSON.parse(screen.getByTestId('selection').textContent ?? 'null');

describe('BundleSourcePicker', () => {
  it('selects an uploaded bundle once its file validates', async () => {
    render(<Harness />);

    fireEvent.change(screen.getByTestId('bundle-input'), {
      target: {
        files: [
          makeFile('bundle/manifest.json', '{"assets": ["a.txt"]}'),
          makeFile('bundle/a.txt', 'a'),
        ],
      },
    });

    await waitFor(() => expect(selection()).toMatchObject({ path: 'manifest.json' }));
    expect(selection().file).toEqual({ assets: ['a.txt'] });
    expect(selection().source.entries).toHaveLength(2);
  });

  it('reports a selection with no matching file in the spec’s words', async () => {
    render(<Harness />);

    fireEvent.change(screen.getByTestId('bundle-input'), {
      target: { files: [makeFile('bundle/a.txt', 'a')] },
    });

    expect(
      await screen.findByText(
        'No manifest in that selection: expected a JSON file with an assets list.'
      )
    ).toBeInTheDocument();
  });

  it('selects a fileset in place, with the spec’s validation', async () => {
    const user = userEvent.setup();
    mockFileset({ 'manifest.json': '{"assets": ["a.txt", "b.txt"]}', 'a.txt': 'a' });
    render(<Harness />);

    await user.click(screen.getByRole('radio', { name: 'Choose from a fileset' }));
    await user.click(await screen.findByRole('combobox', { name: 'Fileset' }));
    await user.click(await screen.findByRole('option', { name: 'assets' }));

    expect(await screen.findByText('b.txt is missing')).toBeInTheDocument();
    expect(selection()).toBeNull();
    expect(screen.getByRole('combobox', { name: 'Manifest' })).toHaveTextContent('manifest.json');
  });
});
