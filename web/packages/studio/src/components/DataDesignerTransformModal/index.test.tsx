// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DataDesignerTransformModal } from '@studio/components/DataDesignerTransformModal';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';

let SOURCE_ROWS: Record<string, string>[] = [];

const withIdColumn = [
  { task_id: 'a1', category: 'billing', user_request: 'Where is my refund?', ideal_response: '..' },
];
const withoutIdColumn = [
  { category: 'billing', user_request: 'Where is my refund?', ideal_response: '..' },
];
// No column an assistant turn could be guessed from.
const withoutResponseColumn = [{ task_id: 'a1', user_request: 'Where is my refund?' }];

const createJobMock = vi.fn();
const onCloseMock = vi.fn();

vi.mock('@studio/api/datasets/useDatasetFileContent', () => ({
  useDatasetFileContent: () => ({
    data: SOURCE_ROWS.map((row) => JSON.stringify(row)).join('\n'),
    isLoading: false,
  }),
}));

vi.mock('@nemo/sdk/generated/data-designer/data-designer', () => ({
  useDataDesignerCreateJob: () => ({
    mutateAsync: createJobMock,
    isPending: false,
    error: null,
  }),
}));

const renderModal = () =>
  render(
    <TestProviders>
      <MemoryRouter>
        <DataDesignerTransformModal
          open
          onClose={onCloseMock}
          workspace="default"
          sourceJobName="Support Evals"
          filesetWorkspace="default"
          filesetName="support-evals-artifacts"
          fileOptions={['dataset.parquet']}
          defaultNumRecords={500}
        />
      </MemoryRouter>
    </TestProviders>
  );

describe('DataDesignerTransformModal', () => {
  beforeEach(() => {
    createJobMock.mockReset();
    onCloseMock.mockReset();
    createJobMock.mockResolvedValue({ name: 'support-evals-agent-eval-tasks' });
    SOURCE_ROWS = withIdColumn;
  });

  it('auto-maps the default format from the source columns', () => {
    renderModal();

    expect(screen.getByText('inputs.instruction')).toBeInTheDocument();
    expect(
      screen.getByRole('combobox', { name: 'inputs.instruction source column' })
    ).toHaveTextContent('user_request');
  });

  it('shows a row preview once the source file is read', async () => {
    renderModal();

    // Rendering of the row itself is covered by the renderTemplate unit tests;
    // CodeSnippet highlights asynchronously, so only the panel is asserted here.
    expect(await screen.findByText('Preview output')).toBeInTheDocument();
  });

  it('creates a processor-only job seeded from the source file', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole('button', { name: 'Create transform job' }));

    expect(createJobMock).toHaveBeenCalledTimes(1);
    const { workspace, data } = createJobMock.mock.calls[0][0];
    expect(workspace).toBe('default');
    expect(data.spec.config.columns).toEqual([]);
    expect(data.spec.config.seed_config.source.path).toBe(
      'default/support-evals-artifacts#dataset.parquet'
    );
    expect(data.spec.config.processors[0].name).toBe('agent_eval_tasks');
    expect(data.spec.num_records).toBe(500);
  });

  it('closes without a prompt when nothing has been edited', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(screen.queryByText('Discard this transform?')).not.toBeInTheDocument();
    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  it('warns before discarding an edited mapping, and keeps it on Keep editing', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText('Output name'));
    await user.type(screen.getByLabelText('Output name'), 'my_tasks');
    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(screen.getByText('Discard this transform?')).toBeInTheDocument();
    expect(onCloseMock).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Keep editing' }));

    expect(screen.queryByText('Discard this transform?')).not.toBeInTheDocument();
    expect(onCloseMock).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Output name')).toHaveValue('my_tasks');
  });

  it('closes once the discard is confirmed', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText('Output name'));
    await user.type(screen.getByLabelText('Output name'), 'my_tasks');
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Discard' }));

    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  it('generates a UUID id column when the source has no unique key', async () => {
    SOURCE_ROWS = withoutIdColumn;
    const user = userEvent.setup();
    renderModal();

    expect(screen.getByText(/adds it as a\s+UUID sampler column/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Create transform job' }));

    const { data } = createJobMock.mock.calls[0][0];
    expect(data.spec.config.columns).toEqual([
      {
        name: 'row_id',
        column_type: 'sampler',
        sampler_type: 'uuid',
        params: { short_form: true },
      },
    ]);
    expect(data.spec.config.processors[0].template.id).toBe('{{ row_id }}');
  });

  it('declares no id column when the source already has one', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole('button', { name: 'Create transform job' }));

    const { data } = createJobMock.mock.calls[0][0];
    expect(data.spec.config.columns).toEqual([]);
    expect(data.spec.config.processors[0].template.id).toBe('{{ task_id }}');
  });

  it('blocks submission while a required field is unmapped', async () => {
    SOURCE_ROWS = withoutResponseColumn;
    const user = userEvent.setup();
    renderModal();

    // Messages needs an assistant turn, which nothing in this file matches.
    await user.click(screen.getByRole('radio', { name: 'Messages' }));

    expect(screen.getByText(/must have a source/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create transform job' })).toBeDisabled();
  });
});
