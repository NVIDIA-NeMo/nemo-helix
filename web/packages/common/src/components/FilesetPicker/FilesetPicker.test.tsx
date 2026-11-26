// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPicker, type FilesetPickerProps } from '@nemo/common/src/components/FilesetPicker';
import { FilesetPickerCore } from '@nemo/common/src/components/FilesetPicker/FilesetPickerCore';
import { filesListFilesets, filesRetrieveFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@nemo/sdk/generated/platform/files', () => ({
  filesListFilesets: vi.fn(),
  filesRetrieveFileset: vi.fn(),
}));

const fileset = (
  name: string,
  overrides: Partial<Pick<FilesetOutput, 'purpose' | 'storage'>> = {}
): FilesetOutput =>
  ({
    id: `default/${name}`,
    name,
    workspace: 'default',
    purpose: 'dataset',
    storage: { type: 'local' },
    ...overrides,
  }) as FilesetOutput;

const httpError = (status: number) => Object.assign(new Error(`HTTP ${status}`), { status });

const listResponse = (filesets: FilesetOutput[]) =>
  ({
    data: filesets,
    pagination: { page: 1, total_pages: 1 },
  }) as Awaited<ReturnType<typeof filesListFilesets>>;

const renderPicker = (props: Partial<FilesetPickerProps> = {}) => {
  const onChange = vi.fn();
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <FilesetPicker
        workspace="default"
        value={null}
        onChange={onChange}
        label="Training dataset"
        purpose="dataset"
        {...props}
      />
    </QueryClientProvider>
  );
  return { onChange, input: screen.getByRole('combobox', { name: 'Training dataset' }) };
};

/** KUI renders the listbox empty state from this attribute, not as text nodes. */
const expectEmptyMessage = async (message: string) =>
  waitFor(() => expect(screen.getByRole('listbox')).toHaveAttribute('data-empty-message', message));

describe('FilesetPicker', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    vi.mocked(filesRetrieveFileset).mockRejectedValue(httpError(404));
  });

  it('lists filesets with their storage backend', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(
      listResponse([
        fileset('sft-chat-v3'),
        fileset('hf-mirror', { storage: { type: 'huggingface' } as FilesetOutput['storage'] }),
      ])
    );
    const { input } = renderPicker();

    await userEvent.click(input);

    expect(await screen.findByRole('option', { name: /sft-chat-v3/ })).toHaveTextContent('Local');
    expect(screen.getByRole('option', { name: /hf-mirror/ })).toHaveTextContent('Hugging Face');
  });

  it('reports the picked reference and fileset', async () => {
    const picked = fileset('sft-chat-v3');
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([picked]));
    const { input, onChange } = renderPicker();

    await userEvent.click(input);
    await userEvent.click(await screen.findByRole('option', { name: /sft-chat-v3/ }));

    expect(onChange).toHaveBeenCalledWith('default/sft-chat-v3', picked);
  });

  it('does not auto-select a lone candidate', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([fileset('only-one')]));
    const { input, onChange } = renderPicker();

    await waitFor(() => expect(filesListFilesets).toHaveBeenCalled());

    expect(onChange).not.toHaveBeenCalled();
    expect(input).toHaveValue('');
  });

  it('says when the workspace has no filesets of this purpose', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([]));
    const { input } = renderPicker();

    await userEvent.click(input);

    await expectEmptyMessage('No filesets in this workspace yet');
  });

  it('distinguishes a search with no matches from an empty workspace', async () => {
    vi.mocked(filesListFilesets).mockImplementation(async (_workspace, params) =>
      listResponse(params?.filter ? [] : [fileset('sft-chat-v3')])
    );
    const { input } = renderPicker({ purpose: undefined });

    await userEvent.type(input, 'zzz');

    await expectEmptyMessage('No filesets match "zzz" — try a different name');
  });

  it('keeps the New row visible and never selects it', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([]));
    const onRequestNew = vi.fn();
    const { input, onChange } = renderPicker({ onRequestNew });

    await userEvent.click(input);
    expect(await screen.findByText('No filesets in this workspace yet')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'New fileset' }));

    expect(onRequestNew).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('shows each purpose only when several are allowed', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(
      listResponse([fileset('env-pkg', { purpose: 'generic' })])
    );
    const { input } = renderPicker({ purpose: ['environment', 'generic'] });

    await userEvent.click(input);

    expect(await screen.findByRole('option', { name: /env-pkg/ })).toHaveTextContent('Generic');
  });

  it('looks up a selected fileset that is not on the loaded page', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([fileset('newer')]));
    vi.mocked(filesRetrieveFileset).mockResolvedValue(fileset('older'));

    const { input } = renderPicker({ value: 'default/older' });

    await waitFor(() => expect(input).toHaveValue('older'));
    expect(filesRetrieveFileset).toHaveBeenCalledWith('default', 'older', expect.anything());
  });

  it('waits for the first page before looking up a preset selection', () => {
    vi.mocked(filesListFilesets).mockReturnValue(new Promise(() => {}));

    renderPicker({ value: 'default/sft-chat-v3' });

    expect(filesRetrieveFileset).not.toHaveBeenCalled();
  });

  it('flags a selection whose fileset no longer exists', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([fileset('other')]));
    vi.mocked(filesRetrieveFileset).mockRejectedValue(httpError(404));

    renderPicker({ value: 'default/deleted' });

    expect(
      await screen.findByText('This fileset no longer exists. Select another one.')
    ).toBeInTheDocument();
  });

  it('keeps the selection apart from the results of an unrelated search', async () => {
    const chosen = fileset('sft-chat-v3');
    vi.mocked(filesListFilesets).mockImplementation(async (_workspace, params) =>
      listResponse(params?.filter ? [] : [chosen])
    );
    const { input } = renderPicker({ value: 'default/sft-chat-v3', purpose: undefined });
    await waitFor(() => expect(input).toHaveValue('sft-chat-v3'));

    await userEvent.clear(input);
    await userEvent.type(input, 'zzz');

    expect(
      await screen.findByText('No filesets match "zzz" — try a different name')
    ).toBeInTheDocument();
    expect(screen.getByText('Selected')).toBeInTheDocument();
  });

  it('reports a blur when focus leaves the input without opening the list', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([]));
    const onBlur = vi.fn();
    const { input } = renderPicker({ onBlur });

    input.focus();
    await userEvent.tab();

    expect(onBlur).toHaveBeenCalled();
  });

  it('runs on the fetchers it is given, so plugins can pass host.sdk', async () => {
    const listFilesets = vi.fn().mockResolvedValue(listResponse([fileset('from-plugin')]));
    const retrieveFileset = vi.fn();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={queryClient}>
        <FilesetPickerCore
          workspace="default"
          value={null}
          onChange={vi.fn()}
          label="Dataset"
          listFilesets={listFilesets}
          retrieveFileset={retrieveFileset}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole('combobox', { name: 'Dataset' }));

    expect(await screen.findByRole('option', { name: /from-plugin/ })).toBeInTheDocument();
    expect(listFilesets).toHaveBeenCalled();
    expect(filesListFilesets).not.toHaveBeenCalled();
  });

  it('does not call a selection deleted when looking it up fails for another reason', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(listResponse([fileset('other')]));
    vi.mocked(filesRetrieveFileset).mockRejectedValue(httpError(500));

    renderPicker({ value: 'default/sft' });

    expect(
      await screen.findByText("Couldn't load the selected fileset. Try again later.")
    ).toBeInTheDocument();
    expect(screen.queryByText(/no longer exists/)).not.toBeInTheDocument();
  });

  it('does not look up a preset selection again when a search hides it', async () => {
    vi.mocked(filesListFilesets).mockImplementation(async (_workspace, params) =>
      listResponse(params?.filter ? [] : [fileset('sft-chat-v3')])
    );
    const { input } = renderPicker({ value: 'default/sft-chat-v3', purpose: undefined });
    await waitFor(() => expect(input).toHaveValue('sft-chat-v3'));

    await userEvent.clear(input);
    await userEvent.type(input, 'zzz');
    await screen.findByText('No filesets match "zzz" — try a different name');

    expect(filesRetrieveFileset).not.toHaveBeenCalled();
  });
});
