// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AnonymizerRecordView } from '@studio/components/AnonymizerRecordView/AnonymizerRecordView';
import { buildAnonymizerRecord } from '@studio/components/AnonymizerRecordView/parse';
import { traceRow } from '@studio/components/AnonymizerRecordView/testFixtures';
import { render, screen, waitFor } from '@testing-library/react';
import type { FC } from 'react';
import { MemoryRouter, useLocation } from 'react-router';

const SearchProbe: FC = () => <span data-testid="search">{useLocation().search}</span>;

const renderRecord = (
  row: Record<string, unknown>,
  outputHeading = 'Replaced',
  initialEntry = '/'
) =>
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <AnonymizerRecordView
        outputHeading={outputHeading}
        record={buildAnonymizerRecord(row, 'biography')}
      />
      <SearchProbe />
    </MemoryRouter>
  );

describe('AnonymizerRecordView', () => {
  it('tags detected entities in both columns', () => {
    renderRecord(traceRow);

    expect(screen.getAllByText('Original').length).toBeGreaterThan(0);
    expect(screen.getByText('Replaced')).toBeInTheDocument();
    expect(screen.getAllByText('Bobby').length).toBeGreaterThan(0);
    expect(screen.getAllByText('first_name').length).toBeGreaterThan(0);
    expect(screen.getAllByText('45').length).toBeGreaterThan(0);
  });

  it('lists every replacement in the map', () => {
    renderRecord(traceRow);

    expect(screen.getByText('Replacement Map')).toBeInTheDocument();
    expect(screen.getAllByText('Teddy').length).toBeGreaterThan(0);
    expect(screen.getAllByText('age').length).toBeGreaterThan(0);
  });

  it('uses the caller heading for the output column', () => {
    renderRecord(traceRow, 'Rewritten');

    expect(screen.getByText('Rewritten')).toBeInTheDocument();
  });

  it('ignores the page search param and leaves it untouched for other tables on the route', async () => {
    renderRecord(traceRow, 'Replaced', '/?page=3');

    expect(screen.getAllByText('Teddy').length).toBeGreaterThan(0);
    expect(screen.queryByText('No Entries Found')).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId('search').textContent).toBe('?page=3'));
  });

  it('explains when nothing was replaced', () => {
    renderRecord({ biography: 'Nothing sensitive here.', biography_replaced: '' });

    expect(screen.getByText('No entities were replaced in this record.')).toBeInTheDocument();
    expect(screen.getByText('No output was produced for this record.')).toBeInTheDocument();
  });
});
