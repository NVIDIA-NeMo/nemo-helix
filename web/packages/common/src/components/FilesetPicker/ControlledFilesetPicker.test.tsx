// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledFilesetPicker } from '@nemo/common/src/components/FilesetPicker/ControlledFilesetPicker';
import { filesListFilesets } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import { useForm } from 'react-hook-form';

vi.mock('@nemo/sdk/generated/platform/files', () => ({
  filesListFilesets: vi.fn(),
  getFilesListFilesetsQueryKey: vi.fn((workspace: string) => ['filesets', workspace]),
  useFilesRetrieveFileset: vi.fn(() => ({ data: undefined })),
}));

interface Form {
  dataset: string | null;
}

const Harness = ({ error, onValue }: { error?: string; onValue: (value: Form) => void }) => {
  const { control, setError, watch } = useForm<Form>({ defaultValues: { dataset: null } });
  useEffect(() => {
    if (error) setError('dataset', { message: error });
  }, [error, setError]);
  onValue(watch());
  return (
    <ControlledFilesetPicker
      workspace="default"
      label="Dataset"
      useControllerProps={{ name: 'dataset', control }}
    />
  );
};

const renderHarness = (error?: string) => {
  let latest: Form = { dataset: null };
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <Harness error={error} onValue={(value) => (latest = value)} />
    </QueryClientProvider>
  );
  return { getValue: () => latest };
};

describe('ControlledFilesetPicker', () => {
  beforeEach(() => {
    vi.mocked(filesListFilesets).mockResolvedValue({
      data: [
        {
          id: 'default/sft',
          name: 'sft',
          workspace: 'default',
          purpose: 'dataset',
          storage: { type: 'local' },
        } as FilesetOutput,
      ],
      pagination: { page: 1, total_pages: 1 },
    } as Awaited<ReturnType<typeof filesListFilesets>>);
  });

  it('writes the workspace/name reference into the form field', async () => {
    const { getValue } = renderHarness();

    await userEvent.click(screen.getByRole('combobox', { name: 'Dataset' }));
    await userEvent.click(await screen.findByRole('option', { name: /sft/ }));

    await waitFor(() => expect(getValue().dataset).toBe('default/sft'));
  });

  it('shows the field error', async () => {
    renderHarness('Pick a dataset');

    expect(await screen.findByText('Pick a dataset')).toBeInTheDocument();
  });
});
