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
  batchGlob: string;
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
  const { control } = useForm<PickerFormValues>({
    defaultValues: { fileset: '', file: '', batchGlob: '' },
  });
  const file = useWatch({ control, name: 'file' });
  const batchGlob = useWatch({ control, name: 'batchGlob' });
  return (
    <>
      <FilesetDatasetPicker<PickerFormValues>
        workspace={DEFAULT_WORKSPACE}
        control={control}
        filesetName="fileset"
        fileName="file"
        batchGlobName="batchGlob"
        error={error}
      />
      <output data-testid="form-file">{file}</output>
      <output data-testid="form-batch-glob">{batchGlob}</output>
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

  it('offers every Parquet batch beside the picked file as a glob', async () => {
    mockFilesets(['out/batch_00000.parquet', 'out/batch_00001.parquet']);
    const user = userEvent.setup();
    render(<PickerHarness />);

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'out/batch_00000.parquet');
    await user.click(
      screen.getByRole('checkbox', { name: 'Evaluate all 2 Parquet files in out/' })
    );

    expect(screen.getByTestId('form-batch-glob')).toHaveTextContent('out/*.parquet');
  });

  it('drops the batch glob when another file is picked', async () => {
    mockFilesets(['out/batch_00000.parquet', 'out/batch_00001.parquet', 'rows.jsonl']);
    const user = userEvent.setup();
    render(<PickerHarness />);

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'out/batch_00000.parquet');
    await user.click(screen.getByRole('checkbox', { name: /Evaluate all 2 Parquet files/ }));
    await chooseFile(user, 'rows.jsonl');

    expect(screen.getByTestId('form-batch-glob')).toBeEmptyDOMElement();
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('does not offer batches for a lone Parquet file', async () => {
    mockFilesets(['out/batch_00000.parquet']);
    const user = userEvent.setup();
    render(<PickerHarness />);

    await chooseFileset(user, 'generated');
    await chooseFile(user, 'out/batch_00000.parquet');

    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });
});
