// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { FilesetDatasetPicker } from '@studio/components/evaluation/shared/FilesetDatasetPicker';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useForm, useWatch } from 'react-hook-form';

interface PickerFormValues {
  fileset: string;
  file: string;
}

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
        data: [
          { name: 'generated', workspace: DEFAULT_WORKSPACE },
          { name: 'other', workspace: DEFAULT_WORKSPACE },
        ],
        pagination: { total: 2, page: 1, page_size: 20 },
      })
    ),
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({ data: paths.map(fileEntry) })
    )
  );
};

const PickerHarness = ({ error }: { error?: string }) => {
  const { control } = useForm<PickerFormValues>({ defaultValues: { fileset: '', file: '' } });
  const file = useWatch({ control, name: 'file' });
  return (
    <>
      <FilesetDatasetPicker<PickerFormValues>
        workspace={DEFAULT_WORKSPACE}
        control={control}
        filesetName="fileset"
        fileName="file"
        error={error}
      />
      <output data-testid="form-file">{file}</output>
    </>
  );
};

const chooseFileset = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  await user.click(await screen.findByRole('combobox', { name: 'Fileset' }));
  await user.click(await screen.findByRole('option', { name }));
};

const chooseFile = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  await user.click(await screen.findByRole('combobox', { name: 'File' }));
  await user.click(await screen.findByRole('option', { name }));
};

describe('FilesetDatasetPicker', () => {
  it('writes the picked file path to the form', async () => {
    mockFilesets(['output/part-0.parquet']);
    const user = userEvent.setup();
    render(<PickerHarness />);

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'output/part-0.parquet');

    expect(screen.getByTestId('form-file')).toHaveTextContent('output/part-0.parquet');
  });

  it('clears the picked file when the fileset changes', async () => {
    mockFilesets(['rows.jsonl']);
    const user = userEvent.setup();
    render(<PickerHarness />);

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'rows.jsonl');
    await chooseFileset(user, 'other');

    expect(screen.getByTestId('form-file')).toBeEmptyDOMElement();
  });

  it('lists only dataset formats', async () => {
    mockFilesets(['rows.jsonl', 'rows.csv', 'README.md', 'notes.txt']);
    const user = userEvent.setup();
    render(<PickerHarness />);

    await chooseFileset(user, 'generated');
    await user.click(await screen.findByRole('combobox', { name: 'File' }));

    expect(await screen.findByRole('option', { name: 'rows.jsonl' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'rows.csv' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'README.md' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'notes.txt' })).not.toBeInTheDocument();
  });

  it('says when a fileset has no dataset files', async () => {
    mockFilesets(['README.md']);
    const user = userEvent.setup();
    render(<PickerHarness />);

    await chooseFileset(user, 'generated');

    expect(
      await screen.findByText('This fileset has no JSONL, JSON, CSV, or Parquet files.')
    ).toBeVisible();
  });

  it('shows the error it is given on the file field', async () => {
    mockFilesets(['rows.jsonl']);
    render(<PickerHarness error="File is too large to edit in the browser." />);

    expect(await screen.findByText('File is too large to edit in the browser.')).toBeVisible();
  });
});
