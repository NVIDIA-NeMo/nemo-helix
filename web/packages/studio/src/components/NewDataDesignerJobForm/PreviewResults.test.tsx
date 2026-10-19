// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PreviewResults } from '@studio/components/NewDataDesignerJobForm/PreviewResults';
import { fireEvent, render, screen, within } from '@studio/tests/util/render';

const LONG_ANSWER =
  'Confirm the reschedule with the line supervisor before moving the batch to the night shift.';

const records = [
  {
    question: 'Can we move batch 14 to Thursday?',
    scheduler: 'alice',
    expected_answer: LONG_ANSWER,
    period: 'Q3',
  },
  {
    question: 'Is line 2 free next week?',
    scheduler: 'bob',
    expected_answer: 'Yes.',
    period: 'Q4',
  },
];

describe('PreviewResults', () => {
  it('renders one table row per record with columns in config order', () => {
    render(
      <PreviewResults
        records={records}
        columnOrder={['scheduler', 'period', 'question', 'expected_answer']}
        isPreviewing={false}
        status={undefined}
        error={undefined}
        onStop={() => undefined}
      />
    );

    const headers = screen.getAllByRole('columnheader').map((header) => header.textContent);
    expect(headers).toEqual(['#', 'scheduler', 'period', 'question', 'expected_answer']);

    const rows = screen.getAllByRole('row').slice(1);
    expect(rows).toHaveLength(2);
    expect(within(rows[1]).getByText('bob')).toBeInTheDocument();
    expect(within(rows[1]).getByText('Is line 2 free next week?')).toBeInTheDocument();
  });

  it('opens a long cell in a modal', async () => {
    render(
      <PreviewResults
        records={records}
        columnOrder={[]}
        isPreviewing={false}
        status={undefined}
        error={undefined}
        onStop={() => undefined}
      />
    );

    fireEvent.click(screen.getByLabelText('Expand cell'));

    expect(await screen.findByText('Row 1 — expected_answer')).toBeInTheDocument();
  });

  it('shows a loading state while previewing without rows', () => {
    render(
      <PreviewResults
        records={undefined}
        columnOrder={[]}
        isPreviewing
        status={undefined}
        error={undefined}
        onStop={() => undefined}
      />
    );

    expect(screen.getByText('Generating preview rows…')).toBeInTheDocument();
  });

  it('shows the latest status under the loading spinner', () => {
    render(
      <PreviewResults
        records={undefined}
        columnOrder={[]}
        isPreviewing
        status="🚀 (1/1) Dispatching with 10 records"
        error={undefined}
        onStop={() => undefined}
      />
    );

    expect(screen.getByText('🚀 (1/1) Dispatching with 10 records')).toBeInTheDocument();
  });

  it('stops the run from the loading state', () => {
    const onStop = vi.fn();
    render(
      <PreviewResults
        records={undefined}
        columnOrder={[]}
        isPreviewing
        status={undefined}
        error={undefined}
        onStop={onStop}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: 'Stop' }));

    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('shows the error when the preview fails without rows', () => {
    render(
      <PreviewResults
        records={undefined}
        columnOrder={[]}
        isPreviewing={false}
        status={undefined}
        error="model unavailable"
        onStop={() => undefined}
      />
    );

    expect(screen.getByText(/The preview failed: model unavailable/)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('shows an empty state when the preview finishes without rows', () => {
    render(
      <PreviewResults
        records={[]}
        columnOrder={[]}
        isPreviewing={false}
        status={undefined}
        error={undefined}
        onStop={() => undefined}
      />
    );

    expect(screen.getByText(/didn't return any rows/)).toBeInTheDocument();
  });
});
