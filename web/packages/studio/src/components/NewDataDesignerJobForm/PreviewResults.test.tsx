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
        error={undefined}
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
      <PreviewResults records={records} columnOrder={[]} isPreviewing={false} error={undefined} />
    );

    fireEvent.click(screen.getByLabelText('Expand cell'));

    expect(await screen.findByText('Row 1 — expected_answer')).toBeInTheDocument();
  });

  it('shows a loading state while previewing without rows', () => {
    render(<PreviewResults records={undefined} columnOrder={[]} isPreviewing error={undefined} />);

    expect(screen.getByText('Generating preview rows…')).toBeInTheDocument();
  });

  it('shows the error when the preview fails without rows', () => {
    render(
      <PreviewResults
        records={undefined}
        columnOrder={[]}
        isPreviewing={false}
        error="model unavailable"
      />
    );

    expect(screen.getByText(/The preview failed: model unavailable/)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('shows an empty state when the preview finishes without rows', () => {
    render(<PreviewResults records={[]} columnOrder={[]} isPreviewing={false} error={undefined} />);

    expect(screen.getByText(/didn't return any rows/)).toBeInTheDocument();
  });
});
