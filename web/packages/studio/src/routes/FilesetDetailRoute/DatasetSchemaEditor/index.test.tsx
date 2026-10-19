// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DatasetMetadataContent,
  FilesetFileOutput,
  FilesetOutput,
} from '@nemo/sdk/generated/platform/schema';
import { DatasetSchemaEditor } from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor';
import { render, screen } from '@studio/tests/util/render';
import { fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

const { mutateAsync, invalidateQueries } = vi.hoisted(() => ({
  mutateAsync: vi.fn().mockResolvedValue({}),
  invalidateQueries: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('@nemo/sdk/generated/platform/files', async () => {
  const actual = await vi.importActual<typeof import('@nemo/sdk/generated/platform/files')>(
    '@nemo/sdk/generated/platform/files'
  );
  return {
    ...actual,
    useFilesUpdateFilesetMetadata: () => ({ mutateAsync, isPending: false }),
  };
});

vi.mock('@tanstack/react-query', async () => {
  const actual =
    await vi.importActual<typeof import('@tanstack/react-query')>('@tanstack/react-query');
  return {
    ...actual,
    useQueryClient: () => ({ invalidateQueries }),
  };
});

vi.mock('@studio/providers/workers/useWorkers', () => ({
  useWorkers: () => ({ createWorker: vi.fn() }),
}));

vi.mock('@nemo/common/src/components/CodeEditor', () => ({
  CodeEditor: ({ content, onChange }: { content: string; onChange?: (next: string) => void }) => (
    <textarea
      data-testid="code-editor"
      aria-label="code editor"
      value={content}
      onChange={(e) => onChange?.(e.target.value)}
    />
  ),
}));

vi.mock('@nemo/common/src/components/CodeEditor/constants', () => ({
  ContentType: { JSON: 'json' },
}));

function buildFileset(metadata?: DatasetMetadataContent): FilesetOutput {
  return {
    id: 'test-fileset-id',
    name: 'test-dataset',
    workspace: 'default',
    description: '',
    purpose: 'dataset',
    storage: { type: 'local', path: '/data/test-dataset' },
    metadata: metadata ? { dataset: metadata } : {},
    custom_fields: {},
    project: 'default',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  };
}

const propsBase = {
  workspace: 'default',
  datasetName: 'test-dataset',
  filesList: [] as FilesetFileOutput[],
};

const filesFor = (paths: string[]): FilesetFileOutput[] =>
  paths.map((p) => ({ file_ref: p, file_url: '', path: p, size: 0 }));

beforeEach(() => {
  mutateAsync.mockClear();
  invalidateQueries.mockClear();
});

describe('DatasetSchemaEditor', () => {
  it('renders empty state when fileset has no metadata', () => {
    render(<DatasetSchemaEditor {...propsBase} fileset={buildFileset()} />);
    expect(screen.getByTestId('dataset-schema-editor-empty')).toBeInTheDocument();
    expect(screen.getByText('No schema yet')).toBeInTheDocument();
  });

  it('renders dropdown options: Default + each schema_defs key + Show All (inline default)', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema: { type: 'object', properties: { a: { type: 'string' } } },
          schema_defs: { schema_1: { type: 'object', properties: {} } },
          schemas_by_path: {},
        })}
      />
    );

    await user.click(screen.getByRole('combobox', { name: /selected schema/i }));

    expect(await screen.findByRole('option', { name: 'Default' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'schema_1' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Show All' })).toBeInTheDocument();
  });

  it('marks the ref-default key with "(default)" and omits the separate Default option', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema: 'schema_1',
          schema_defs: {
            schema_1: { type: 'object', properties: { a: { type: 'string' } } },
            schema_2: { type: 'object', properties: { b: { type: 'integer' } } },
          },
          schemas_by_path: {},
        })}
      />
    );

    await user.click(screen.getByRole('combobox', { name: /selected schema/i }));

    expect(screen.queryByRole('option', { name: 'Default' })).toBeNull();
    expect(await screen.findByRole('option', { name: 'schema_1 (default)' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'schema_2' })).toBeInTheDocument();
  });

  it('Show All renders the whole metadata.dataset JSON', async () => {
    const user = userEvent.setup();
    const metadata: DatasetMetadataContent = {
      schema: 'schema_1',
      schema_defs: { schema_1: { type: 'object', properties: { a: { type: 'string' } } } },
      schemas_by_path: {},
    };
    render(<DatasetSchemaEditor {...propsBase} fileset={buildFileset(metadata)} />);

    await user.click(screen.getByRole('combobox', { name: /selected schema/i }));
    await user.click(await screen.findByRole('option', { name: 'Show All' }));

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    expect(JSON.parse(editor.value)).toEqual(metadata);
  });

  it('single-schema selection renders properties-only (not the full schema shell)', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema: 'schema_1',
          schema_defs: {
            schema_1: {
              $schema: 'https://json-schema.org/draft/2020-12/schema',
              type: 'object',
              properties: { a: { type: 'string' } },
            },
          },
          schemas_by_path: {},
        })}
      />
    );

    await user.click(screen.getByRole('combobox', { name: /selected schema/i }));
    await user.click(await screen.findByRole('option', { name: 'schema_1 (default)' }));

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    expect(JSON.parse(editor.value)).toEqual({ a: { type: 'string' } });
  });

  it('editing + Save calls updateMetadata with properties merged back into the schema', async () => {
    const user = userEvent.setup();
    const metadata: DatasetMetadataContent = {
      schema_defs: {
        schema_1: {
          $schema: 'https://json-schema.org/draft/2020-12/schema',
          type: 'object',
          properties: { a: { type: 'string' } },
        },
      },
      schemas_by_path: {},
    };
    render(<DatasetSchemaEditor {...propsBase} fileset={buildFileset(metadata)} />);

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    const newProps = { a: { type: 'string' }, b: { type: 'integer' } };
    fireEvent.change(editor, { target: { value: JSON.stringify(newProps) } });

    const saveButton = screen.getByTestId('dataset-schema-save-button');
    await waitFor(() => expect(saveButton).not.toBeDisabled());
    await user.click(saveButton);

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    const call = mutateAsync.mock.calls[0][0];
    expect(call.workspace).toBe('default');
    expect(call.name).toBe('test-dataset');
    expect(call.data.metadata.dataset.schema_defs.schema_1).toEqual({
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      type: 'object',
      properties: newProps,
    });
    expect(invalidateQueries).toHaveBeenCalledTimes(1);
  });

  it('Save with empty editor in Show All clears metadata.dataset to null', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema_defs: { schema_1: { type: 'object', properties: {} } },
          schemas_by_path: {},
        })}
      />
    );

    await user.click(screen.getByRole('combobox', { name: /selected schema/i }));
    await user.click(await screen.findByRole('option', { name: 'Show All' }));

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: '' } });

    const saveButton = screen.getByTestId('dataset-schema-save-button');
    await waitFor(() => expect(saveButton).not.toBeDisabled());
    await user.click(saveButton);

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    expect(mutateAsync.mock.calls[0][0].data.metadata.dataset).toBeNull();
  });

  it('Reset is disabled until the user edits, and clears the edit when clicked', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema_defs: {
            schema_1: { type: 'object', properties: { a: { type: 'string' } } },
          },
          schemas_by_path: {},
        })}
      />
    );

    const resetButton = screen.getByTestId('dataset-schema-reset-button');
    expect(resetButton).toBeDisabled();

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    const originalText = editor.value;
    fireEvent.change(editor, { target: { value: '{"a":{"type":"integer"}}' } });

    await waitFor(() => expect(resetButton).not.toBeDisabled());
    await user.click(resetButton);

    expect((screen.getByTestId('code-editor') as HTMLTextAreaElement).value).toBe(originalText);
  });

  it('Set Default is disabled for the current default and enabled for other keys', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema: 'schema_1',
          schema_defs: {
            schema_1: { type: 'object', properties: {} },
            schema_2: { type: 'object', properties: {} },
          },
          schemas_by_path: {},
        })}
      />
    );

    const setDefaultButton = screen.getByTestId('dataset-schema-set-default-button');
    expect(setDefaultButton).toBeDisabled();

    await user.click(screen.getByRole('combobox', { name: /selected schema/i }));
    await user.click(await screen.findByRole('option', { name: 'schema_2' }));

    await waitFor(() => expect(setDefaultButton).not.toBeDisabled());
    await user.click(setDefaultButton);

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    expect(mutateAsync.mock.calls[0][0].data.metadata.dataset.schema).toBe('schema_2');
  });

  it('auto-selects the file mapping when selectedFilePath is provided', async () => {
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema_defs: {
            schema_1: { type: 'object', properties: { a: { type: 'string' } } },
            schema_2: { type: 'object', properties: { b: { type: 'integer' } } },
          },
          schemas_by_path: { 'training.jsonl': 'schema_2' },
        })}
        selectedFilePath="training.jsonl"
      />
    );

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: /selected schema/i })).toHaveTextContent(
        'schema_2'
      )
    );
  });

  it('saving an edit to a schema referenced by >1 file opens the shared-schema confirm modal', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema_defs: {
            schema_1: { type: 'object', properties: { a: { type: 'string' } } },
          },
          schemas_by_path: { 'a.jsonl': 'schema_1', 'b.jsonl': 'schema_1' },
        })}
        filesList={filesFor(['a.jsonl', 'b.jsonl'])}
      />
    );

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: '{"a":{"type":"integer"}}' } });

    const saveButton = screen.getByTestId('dataset-schema-save-button');
    await waitFor(() => expect(saveButton).not.toBeDisabled());
    await user.click(saveButton);

    expect(await screen.findByTestId('shared-schema-confirm-message')).toHaveTextContent(
      'used by 2 files'
    );
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it('shared-schema modal Cancel closes without saving', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema_defs: {
            schema_1: { type: 'object', properties: { a: { type: 'string' } } },
          },
          schemas_by_path: { 'a.jsonl': 'schema_1', 'b.jsonl': 'schema_1' },
        })}
        filesList={filesFor(['a.jsonl', 'b.jsonl'])}
      />
    );

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: '{"a":{"type":"integer"}}' } });

    const saveButton = screen.getByTestId('dataset-schema-save-button');
    await waitFor(() => expect(saveButton).not.toBeDisabled());
    await user.click(saveButton);

    await user.click(await screen.findByTestId('shared-schema-confirm-cancel'));

    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it('shared-schema modal OK saves the edit', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema_defs: {
            schema_1: { type: 'object', properties: { a: { type: 'string' } } },
          },
          schemas_by_path: { 'a.jsonl': 'schema_1', 'b.jsonl': 'schema_1' },
        })}
        filesList={filesFor(['a.jsonl', 'b.jsonl'])}
      />
    );

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: '{"a":{"type":"integer"}}' } });

    const saveButton = screen.getByTestId('dataset-schema-save-button');
    await waitFor(() => expect(saveButton).not.toBeDisabled());
    await user.click(saveButton);

    await user.click(await screen.findByTestId('shared-schema-confirm-ok'));

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
  });

  it('saving an edit to a schema referenced by 1 file bypasses the modal', async () => {
    const user = userEvent.setup();
    render(
      <DatasetSchemaEditor
        {...propsBase}
        fileset={buildFileset({
          schema_defs: {
            schema_1: { type: 'object', properties: { a: { type: 'string' } } },
          },
          schemas_by_path: { 'a.jsonl': 'schema_1' },
        })}
        filesList={filesFor(['a.jsonl'])}
      />
    );

    const editor = screen.getByTestId('code-editor') as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: '{"a":{"type":"integer"}}' } });

    const saveButton = screen.getByTestId('dataset-schema-save-button');
    await waitFor(() => expect(saveButton).not.toBeDisabled());
    await user.click(saveButton);

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    expect(screen.queryByTestId('shared-schema-confirm-message')).toBeNull();
  });

  it('jumps to Show All when selectedFilePath transitions from set to undefined', async () => {
    const metadata: DatasetMetadataContent = {
      schema_defs: { schema_1: { type: 'object', properties: {} } },
      schemas_by_path: { 'training.jsonl': 'schema_1' },
    };
    // Host wrapper that toggles selectedFilePath via a button. Mirrors what
    // FilesTab does in real usage and avoids rerender-from-render-helper
    // subtleties (e.g. fresh QueryClient per render in TestProviders).
    const Host = () => {
      const [path, setPath] = useState<string | undefined>('training.jsonl');
      return (
        <>
          <button data-testid="toggle-path" onClick={() => setPath(undefined)}>
            close
          </button>
          <DatasetSchemaEditor
            {...propsBase}
            fileset={buildFileset(metadata)}
            selectedFilePath={path}
          />
        </>
      );
    };

    const user = userEvent.setup();
    render(<Host />);

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: /selected schema/i })).toHaveTextContent(
        'schema_1'
      )
    );

    await user.click(screen.getByTestId('toggle-path'));

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: /selected schema/i })).toHaveTextContent(
        'Show All'
      )
    );
  });
});
