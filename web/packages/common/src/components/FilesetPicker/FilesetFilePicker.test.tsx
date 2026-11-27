// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledFilesetFilePicker } from '@nemo/common/src/components/FilesetPicker/ControlledFilesetFilePicker';
import {
  FilesetFilePicker,
  type FilesetFileValue,
} from '@nemo/common/src/components/FilesetPicker/FilesetFilePicker';
import {
  filesListFilesetFiles,
  filesListFilesets,
  filesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import type {
  FilesetFileOutput,
  FilesetOutput,
  FilesetOutputsPage,
} from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { type ReactNode, useState } from 'react';
import { useForm } from 'react-hook-form';

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesListFilesets: vi.fn(),
  filesRetrieveFileset: vi.fn(),
  filesListFilesetFiles: vi.fn(),
}));

const fileset = (name: string): FilesetOutput => ({
  id: `default/${name}`,
  name,
  workspace: 'default',
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: name },
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2026-11-01T00:00:00Z',
  updated_at: '2026-11-01T00:00:00Z',
});

const filesetsPage = (filesets: FilesetOutput[]): FilesetOutputsPage => ({
  data: filesets,
  pagination: {
    page: 1,
    page_size: 20,
    current_page_size: filesets.length,
    total_pages: 1,
    total_results: filesets.length,
  },
});

const file = (path: string, size = 2048): FilesetFileOutput => ({
  path,
  size,
  file_ref: path,
  file_url: path,
});

const mockFiles = (files: FilesetFileOutput[]) =>
  vi.mocked(filesListFilesetFiles).mockResolvedValue({ data: files });

const withQueryClient = (ui: ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {ui}
  </QueryClientProvider>
);

interface HarnessProps {
  initial?: FilesetFileValue;
  onValue?: (value: FilesetFileValue) => void;
}

const Harness = ({ initial = { filesetRef: null, filePath: null }, onValue }: HarnessProps) => {
  const [value, setValue] = useState<FilesetFileValue>(initial);
  return (
    <FilesetFilePicker
      workspace="default"
      label="Dataset"
      noun={{ one: 'dataset', many: 'datasets' }}
      acceptedFileExtensions={['.csv', '.parquet']}
      value={value}
      onChange={(next) => {
        setValue(next);
        onValue?.(next);
      }}
    />
  );
};

const pickFileset = async (name: string) => {
  await userEvent.click(screen.getByRole('combobox', { name: 'Dataset' }));
  await userEvent.click(await screen.findByRole('option', { name: new RegExp(name) }));
};

describe('FilesetFilePicker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(filesRetrieveFileset).mockImplementation(async (_workspace, name) => fileset(name));
    vi.mocked(filesListFilesets).mockResolvedValue(
      filesetsPage([fileset('eval-a'), fileset('eval-b')])
    );
    mockFiles([file('b.parquet'), file('a.csv'), file('notes.jsonl')]);
  });

  it('asks for a file only after a fileset is picked', async () => {
    render(withQueryClient(<Harness />));

    expect(screen.queryByRole('combobox', { name: 'File' })).not.toBeInTheDocument();
    await pickFileset('eval-a');

    expect(await screen.findByRole('combobox', { name: 'File' })).toBeInTheDocument();
    expect(screen.getByText('Select a .csv or .parquet file')).toBeInTheDocument();
  });

  it('lists ineligible files but never selects them', async () => {
    const onValue = vi.fn();
    render(withQueryClient(<Harness onValue={onValue} />));
    await pickFileset('eval-a');
    onValue.mockClear();

    await userEvent.click(await screen.findByRole('combobox', { name: 'File' }));
    const jsonl = await screen.findByRole('option', { name: /notes\.jsonl/ });
    expect(jsonl).toHaveTextContent('Needs .csv or .parquet');
    await userEvent.click(jsonl);
    expect(onValue).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('option', { name: /a\.csv/ }));
    expect(onValue).toHaveBeenLastCalledWith({ filesetRef: 'default/eval-a', filePath: 'a.csv' });
  });

  it('says when no file in the fileset can be used', async () => {
    mockFiles([file('notes.jsonl')]);
    render(withQueryClient(<Harness initial={{ filesetRef: 'default/eval-a', filePath: null }} />));

    expect(
      await screen.findByText('This dataset has no .csv or .parquet files')
    ).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'File' })).toBeEnabled();
  });

  it('says when the fileset is empty', async () => {
    mockFiles([]);
    render(withQueryClient(<Harness initial={{ filesetRef: 'default/eval-a', filePath: null }} />));

    expect(await screen.findByText('This dataset has no files')).toBeInTheDocument();
  });

  it('flags a saved file that is no longer in the fileset', async () => {
    render(
      withQueryClient(<Harness initial={{ filesetRef: 'default/eval-a', filePath: 'gone.csv' }} />)
    );

    expect(
      await screen.findByText('This file is no longer in this dataset. Select another one.')
    ).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'File' })).toHaveValue('gone.csv');
  });

  it('clears the file when the fileset changes', async () => {
    const onValue = vi.fn();
    render(
      withQueryClient(
        <Harness initial={{ filesetRef: 'default/eval-a', filePath: 'a.csv' }} onValue={onValue} />
      )
    );

    await pickFileset('eval-b');

    expect(onValue).toHaveBeenLastCalledWith({ filesetRef: 'default/eval-b', filePath: null });
  });
});

interface Form {
  source: string | null;
}

interface ControlledHarnessProps {
  defaultSource?: string | null;
  onValue: (value: Form) => void;
  required?: boolean;
}

const ControlledHarness = ({ defaultSource = null, onValue, required }: ControlledHarnessProps) => {
  const { control, watch, reset } = useForm<Form>({
    defaultValues: { source: defaultSource },
    mode: 'onTouched',
  });
  onValue(watch());
  return (
    <>
      <ControlledFilesetFilePicker
        workspace="default"
        label="Dataset"
        useControllerProps={{
          name: 'source',
          control,
          rules: required ? { required: 'Pick a file' } : undefined,
        }}
      />
      <button type="button" onClick={() => reset({ source: null })}>
        Reset
      </button>
    </>
  );
};

describe('ControlledFilesetFilePicker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(filesRetrieveFileset).mockImplementation(async (_workspace, name) => fileset(name));
    vi.mocked(filesListFilesets).mockResolvedValue(filesetsPage([fileset('eval-a')]));
    mockFiles([file('a.csv')]);
  });

  it('keeps the field empty until a file is picked, then writes workspace/name#path', async () => {
    let latest: Form = { source: null };
    render(withQueryClient(<ControlledHarness onValue={(value) => (latest = value)} />));

    await pickFileset('eval-a');
    expect(latest.source).toBeNull();

    await userEvent.click(await screen.findByRole('combobox', { name: 'File' }));
    await userEvent.click(await screen.findByRole('option', { name: /a\.csv/ }));

    await waitFor(() => expect(latest.source).toBe('default/eval-a#a.csv'));
  });

  it('shows the fileset and file path from an existing field value', async () => {
    render(
      withQueryClient(<ControlledHarness defaultSource="default/eval-a#a.csv" onValue={() => {}} />)
    );

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'File' })).toHaveValue('a.csv')
    );
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Dataset' })).toHaveValue('eval-a')
    );
  });

  it('drops a picked fileset when the form resets', async () => {
    render(withQueryClient(<ControlledHarness onValue={() => {}} />));
    await pickFileset('eval-a');
    await userEvent.click(await screen.findByRole('combobox', { name: 'File' }));
    await userEvent.click(await screen.findByRole('option', { name: /a\.csv/ }));

    await userEvent.click(screen.getByRole('button', { name: 'Reset' }));

    await waitFor(() =>
      expect(screen.queryByRole('combobox', { name: 'File' })).not.toBeInTheDocument()
    );
    expect(screen.getByRole('combobox', { name: 'Dataset' })).toHaveValue('');
  });

  it('reports the field touched only once the File field is left', async () => {
    render(withQueryClient(<ControlledHarness onValue={() => {}} required />));
    await pickFileset('eval-a');

    await userEvent.tab();
    expect(screen.queryByText('Pick a file')).not.toBeInTheDocument();

    await userEvent.click(await screen.findByRole('combobox', { name: 'File' }));
    await userEvent.tab();
    expect(await screen.findByText('Pick a file')).toBeInTheDocument();
  });

  it("reads a stored value without a workspace as one in the picker's workspace", async () => {
    render(withQueryClient(<ControlledHarness defaultSource="eval-a#a.csv" onValue={() => {}} />));

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Dataset' })).toHaveValue('eval-a')
    );
    expect(screen.getByRole('combobox', { name: 'File' })).toHaveValue('a.csv');
  });
});
