// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileRowEditor } from '@studio/components/FileRowEditor';
import { SAMPLE_ROWS } from '@studio/components/FileRowEditor/sampleRows';
import type { DataFileColumn, DataFileRow } from '@studio/components/FileRowEditor/types';
import { menuFor } from '@studio/tests/util/menuFor';
import { render, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

describe('FileRowEditor', () => {
  it('shows row actions with Delete row styled as danger', async () => {
    const user = userEvent.setup();
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    const [trigger] = screen.getAllByRole('button', { name: 'Open quick actions menu' });
    await user.click(trigger);

    const items = within(menuFor(trigger)).getAllByTestId('quick-actions-menu-item');
    expect(items.map((item) => item.textContent)).toEqual([
      'Edit row',
      'Duplicate row',
      'Delete row',
    ]);
    expect(items.at(-1)).toHaveClass('nv-menu-item--danger');
  });

  it('opens the row editor from the Edit row action', async () => {
    const user = userEvent.setup();
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    const [trigger] = screen.getAllByRole('button', { name: 'Open quick actions menu' });
    await user.click(trigger);
    await user.click(within(menuFor(trigger)).getByRole('menuitem', { name: 'Edit row' }));

    expect(screen.getByText('Edit Row')).toBeInTheDocument();
    expect(screen.getByLabelText('topic')).toHaveValue('CUDA streams');
  });

  it('adds a copy of the row from the Duplicate row action', async () => {
    const user = userEvent.setup();
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    expect(screen.getByText(/^10 rows/)).toBeInTheDocument();
    const [trigger] = screen.getAllByRole('button', { name: 'Open quick actions menu' });
    await user.click(trigger);
    await user.click(within(menuFor(trigger)).getByRole('menuitem', { name: 'Duplicate row' }));

    expect(await screen.findByText(/^11 rows/)).toBeInTheDocument();
  });

  it('renders the file header and sample rows', () => {
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    expect(screen.getByText('qa-sft-dataset-v1.parquet')).toBeInTheDocument();
    expect(screen.getByText('PARQUET')).toBeInTheDocument();
    expect(screen.getByText('TensorRT')).toBeInTheDocument();
    expect(screen.getByText('CUDA streams')).toBeInTheDocument();
  });

  it('opens the row editor when a row is clicked', async () => {
    const user = userEvent.setup();
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    await user.click(screen.getByText('TensorRT'));

    expect(screen.getByText('Edit Row')).toBeInTheDocument();
    expect(screen.getByLabelText('topic')).toHaveValue('TensorRT');
  });

  it('flags unsaved edits and clears the flag after saving', async () => {
    const user = userEvent.setup();
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    await user.click(screen.getByText('cuDNN'));
    const topicInput = screen.getByLabelText('topic');
    await user.type(topicInput, ' updated');

    expect(screen.getByText('Unsaved edits')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Stage Changes' }));

    expect(screen.queryByText('Unsaved edits')).not.toBeInTheDocument();
  });

  it('omits the Save File action when no onSaveFile handler is provided', () => {
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    expect(screen.queryByRole('button', { name: 'Save File' })).not.toBeInTheDocument();
  });

  it('enables Save File only after edits, persists rows, then disables it again', async () => {
    const user = userEvent.setup();
    const onSaveFile = vi.fn().mockResolvedValue(undefined);
    render(<FileRowEditor initialRows={SAMPLE_ROWS} onSaveFile={onSaveFile} />);

    // Clean on load → disabled.
    const saveFileButton = screen.getByRole('button', { name: 'Save File' });
    expect(saveFileButton).toBeDisabled();

    // Edit a row and commit it to the table.
    await user.click(screen.getByText('cuDNN'));
    await user.type(screen.getByLabelText('topic'), ' updated');
    await user.click(screen.getByRole('button', { name: 'Stage Changes' }));

    // Now dirty → enabled, with an "Unsaved changes" chip in the toolbar.
    expect(saveFileButton).toBeEnabled();
    expect(screen.getByText('Unsaved changes')).toBeInTheDocument();

    await user.click(saveFileButton);

    expect(onSaveFile).toHaveBeenCalledTimes(1);
    const savedRows = onSaveFile.mock.calls[0][0] as DataFileRow[];
    expect(savedRows.some((row) => row.topic === 'cuDNN updated')).toBe(true);

    // Baseline resets after a successful save → disabled again, chip gone.
    await waitFor(() => expect(saveFileButton).toBeDisabled());
    expect(screen.queryByText('Unsaved changes')).not.toBeInTheDocument();
  });

  it('re-enables Save File for a second edit after a successful save', async () => {
    const user = userEvent.setup();
    const onSaveFile = vi.fn().mockResolvedValue(undefined);
    render(<FileRowEditor initialRows={SAMPLE_ROWS} onSaveFile={onSaveFile} />);

    const saveFileButton = screen.getByRole('button', { name: 'Save File' });

    // Cycle 1: edit → stage → save.
    await user.click(screen.getByText('cuDNN'));
    await user.type(screen.getByLabelText('topic'), ' one');
    await user.click(screen.getByRole('button', { name: 'Stage Changes' }));
    expect(saveFileButton).toBeEnabled();
    await user.click(saveFileButton);
    await waitFor(() => expect(saveFileButton).toBeDisabled());
    expect(onSaveFile).toHaveBeenCalledTimes(1);

    // Cycle 2: edit a row again → stage → the button must re-enable and save again.
    await user.click(screen.getByText('TensorRT'));
    await user.type(screen.getByLabelText('topic'), ' two');
    await user.click(screen.getByRole('button', { name: 'Stage Changes' }));
    await waitFor(() => expect(saveFileButton).toBeEnabled());
    await user.click(saveFileButton);
    await waitFor(() => expect(onSaveFile).toHaveBeenCalledTimes(2));
  });

  it('keeps the dirty state when a save fails so the user can retry', async () => {
    const user = userEvent.setup();
    const onSaveFile = vi.fn().mockRejectedValue(new Error('upload failed'));
    render(<FileRowEditor initialRows={SAMPLE_ROWS} onSaveFile={onSaveFile} />);

    await user.click(screen.getByText('cuDNN'));
    await user.type(screen.getByLabelText('topic'), ' updated');
    await user.click(screen.getByRole('button', { name: 'Stage Changes' }));

    const saveFileButton = screen.getByRole('button', { name: 'Save File' });
    await user.click(saveFileButton);

    expect(onSaveFile).toHaveBeenCalledTimes(1);
    // Still dirty → still enabled for a retry.
    await waitFor(() => expect(saveFileButton).toBeEnabled());
  });

  it('disables Save File with a reason when the host blocks saving', () => {
    render(
      <FileRowEditor
        initialRows={SAMPLE_ROWS}
        onSaveFile={vi.fn()}
        saveDisabledReason="File too large"
      />
    );

    expect(screen.getByRole('button', { name: 'Save File' })).toBeDisabled();
  });

  it('filters rows by the search query', async () => {
    const user = userEvent.setup();
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    await user.type(screen.getByPlaceholderText('Search rows…'), 'TensorRT');

    // Search passes through the data-view SearchBar debounce and the hook's debounce,
    // so allow extra time for the filtered result to settle.
    await waitFor(() => expect(screen.queryByText('CUDA streams')).not.toBeInTheDocument(), {
      timeout: 5000,
    });
    expect(screen.getByText('TensorRT')).toBeInTheDocument();
  });

  it('adds a new blank row and opens it for editing using the provided schema', async () => {
    const user = userEvent.setup();
    const columns: DataFileColumn[] = [
      { key: 'prompt', label: 'prompt', type: 'string' },
      { key: 'score', label: 'score', type: 'float' },
    ];
    render(<FileRowEditor initialRows={[]} columns={columns} />);

    await user.click(screen.getByRole('button', { name: 'Add Row' }));

    expect(screen.getByText('Edit Row')).toBeInTheDocument();
    expect(screen.getByLabelText('prompt')).toHaveValue('');
    // float default is 0
    expect(screen.getByLabelText('score')).toHaveValue(0);
  });

  it('edits an enum-like column with a select instead of a free-text input', async () => {
    const user = userEvent.setup();
    render(<FileRowEditor initialRows={SAMPLE_ROWS} />);

    await user.click(screen.getByText('TensorRT'));

    // `difficulty` has only easy/medium/hard, so it is an enum → dropdown, not a textbox.
    expect(screen.queryByRole('textbox', { name: 'difficulty' })).not.toBeInTheDocument();
    expect(screen.getAllByLabelText('difficulty').length).toBeGreaterThan(0);
    // `topic` is high-cardinality free text → still a textbox.
    expect(screen.getByRole('textbox', { name: 'topic' })).toBeInTheDocument();
  });

  it('infers columns and editor controls from an arbitrary row shape', async () => {
    const user = userEvent.setup();
    const rows: DataFileRow[] = [
      { order_id: 'A-1001', customer: 'Acme Corp', total: 1299.5, shipped: true, tags: ['vip'] },
      { order_id: 'A-1002', customer: 'Globex', total: 49.99, shipped: false, tags: [] },
    ];
    render(<FileRowEditor fileName="orders.csv" initialRows={rows} />);

    // Inferred header chip + inferred cell values for a non-SFT schema. Click a
    // float cell (low-cardinality string columns also surface as filter options,
    // which would otherwise duplicate the cell text).
    expect(screen.getByText('CSV')).toBeInTheDocument();
    expect(screen.getByText('1299.5')).toBeInTheDocument();

    await user.click(screen.getByText('1299.5'));

    // number → numeric input, boolean/enum → labeled select, json (array) → JSON area.
    // (customer/shipped are low-cardinality here, so they render as dropdowns.)
    expect(screen.getByLabelText('total')).toHaveValue(1299.5);
    expect(screen.getAllByLabelText('customer').length).toBeGreaterThan(0);
    expect(screen.getAllByLabelText('shipped').length).toBeGreaterThan(0);
    expect(screen.getByLabelText('tags')).toHaveValue('[\n  "vip"\n]');
  });
});
