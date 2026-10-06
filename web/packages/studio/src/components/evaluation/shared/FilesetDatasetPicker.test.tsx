// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { FilesetDatasetPicker } from '@studio/components/evaluation/shared/FilesetDatasetPicker';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { type FC } from 'react';
import { useForm } from 'react-hook-form';

vi.mock('@studio/api/datasets/useDatasetFileContent', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/api/datasets/useDatasetFileContent')>()),
  datasetFileContentQueryOptions: vi.fn(),
}));

const JSONL_ROWS = '{"prompt":"hi"}\n{"prompt":"yo"}\n';

const fileEntry = (path: string) => ({
  file_ref: `${DEFAULT_WORKSPACE}/generated#${path}`,
  file_url: `/${path}`,
  path,
  size: 10,
});

const mockFilesets = (paths: string[]) => {
  server.use(
    http.get(mockApiUrl(getFilesListFilesetsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [{ name: 'generated', workspace: DEFAULT_WORKSPACE }],
        pagination: { total: 1, page: 1, page_size: 20 },
      })
    ),
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({ data: paths.map(fileEntry) })
    )
  );
};

const mockFileContent = (read: () => Promise<string>) => {
  vi.mocked(datasetFileContentQueryOptions).mockImplementation(
    ({ path }) =>
      ({
        queryKey: ['test-file-content', path],
        queryFn: read,
      }) as never
  );
};

interface PickerFormValues {
  fileset: string;
}

const PickerHarness: FC<
  Pick<React.ComponentProps<typeof FilesetDatasetPicker>, 'onPick' | 'onClear'>
> = (props) => {
  const { control } = useForm<PickerFormValues>({ defaultValues: { fileset: '' } });
  return (
    <FilesetDatasetPicker<PickerFormValues>
      workspace={DEFAULT_WORKSPACE}
      filesetControllerProps={{ control, name: 'fileset' }}
      {...props}
    />
  );
};

const renderPicker = () => {
  const onPick = vi.fn();
  const onClear = vi.fn();
  render(<PickerHarness onPick={onPick} onClear={onClear} />);
  return { onPick, onClear };
};

const chooseFileset = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  await user.click(await screen.findByRole('combobox', { name: 'Fileset' }));
  await user.click(await screen.findByRole('option', { name }));
};

const chooseFile = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  await user.click(await screen.findByRole('combobox', { name: 'File' }));
  await user.click(await screen.findByRole('option', { name }));
};

afterEach(() => {
  vi.clearAllMocks();
});

describe('FilesetDatasetPicker', () => {
  it('hands the picked file over under its own name', async () => {
    mockFilesets(['rows.jsonl']);
    mockFileContent(async () => JSONL_ROWS);
    const user = userEvent.setup();
    const { onPick } = renderPicker();

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'rows.jsonl');

    await waitFor(() => expect(onPick).toHaveBeenCalledTimes(1));
    const file: File = onPick.mock.calls[0][0];
    expect(file.name).toBe('rows.jsonl');
    expect(await file.text()).toBe(JSONL_ROWS);
  });

  it('renames a parquet file to the JSONL it was decoded into', async () => {
    mockFilesets(['output/part-0.parquet']);
    mockFileContent(async () => JSONL_ROWS);
    const user = userEvent.setup();
    const { onPick } = renderPicker();

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'output/part-0.parquet');

    await waitFor(() => expect(onPick).toHaveBeenCalledTimes(1));
    expect(onPick.mock.calls[0][0].name).toBe('part-0.jsonl');
  });

  it('lists only dataset formats', async () => {
    mockFilesets(['rows.jsonl', 'README.md', 'notes.txt']);
    const user = userEvent.setup();
    renderPicker();

    await chooseFileset(user, 'generated');
    await user.click(await screen.findByRole('combobox', { name: 'File' }));

    expect(await screen.findByRole('option', { name: 'rows.jsonl' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'README.md' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'notes.txt' })).not.toBeInTheDocument();
  });

  it('shows why a file could not be read and hands nothing over', async () => {
    mockFilesets(['rows.jsonl']);
    mockFileContent(async () => {
      throw new Error('File is too large to edit in the browser.');
    });
    const user = userEvent.setup();
    const { onPick } = renderPicker();

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'rows.jsonl');

    expect(await screen.findByText('File is too large to edit in the browser.')).toBeVisible();
    expect(onPick).not.toHaveBeenCalled();
  });
});
