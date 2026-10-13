// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ResultsPreviewPanel } from '@studio/routes/AnonymizerJobDetailRoute/components/ResultsPreviewPanel';
import { fireEvent, render, renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import { useState, type FC } from 'react';

vi.mock('@studio/routes/AnonymizerJobDetailRoute/useResultPreview', () => ({
  useResultPreview: () => ({
    rows: [
      { text: 'Hello John', text_replaced: 'Hello [NAME]' },
      { text: 'Hello Jane', text_replaced: 'Hello [NAME]' },
    ],
    textColumn: 'text',
    isLoading: false,
    error: null,
  }),
}));

describe('ResultsPreviewPanel', () => {
  it('opens the record preview modal when a row is selected', async () => {
    render(<ResultsPreviewPanel workspace="default" artifactUrl="default/job#results" />);

    fireEvent.click(screen.getByText('Hello John'));

    const modal = within(await screen.findByRole('dialog'));
    expect(modal.getByText('Record Preview')).toBeInTheDocument();
    expect(modal.getByText('Hello John')).toBeInTheDocument();
    expect(modal.getByText('Hello [NAME]')).toBeInTheDocument();
  });

  it('opens the record preview modal from the Details button', async () => {
    render(<ResultsPreviewPanel workspace="default" artifactUrl="default/job#results" />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Details' })[0]);

    const modal = within(await screen.findByRole('dialog'));
    expect(modal.getByText('Record Preview')).toBeInTheDocument();
  });

  it('navigates between records with the pager', async () => {
    render(<ResultsPreviewPanel workspace="default" artifactUrl="default/job#results" />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Details' })[0]);

    const modal = within(await screen.findByRole('dialog'));
    expect(modal.getByText('Record 1 of 2')).toBeInTheDocument();
    expect(modal.getByText('Hello John')).toBeInTheDocument();

    fireEvent.click(modal.getByRole('button', { name: 'Next record' }));

    expect(modal.getByText('Record 2 of 2')).toBeInTheDocument();
    expect(modal.getByText('Hello Jane')).toBeInTheDocument();
  });

  it('sorts preview rows from the url sort param', () => {
    renderRoute(<ResultsPreviewPanel workspace="default" artifactUrl="default/job#results" />, {
      history: '/anonymizer/jobs/job?sort=text',
    });

    const records = screen.getAllByText(/^Hello (John|Jane)$/).map((el) => el.textContent);
    expect(records).toEqual(['Hello Jane', 'Hello John']);
  });

  it('closes the record preview when the artifact changes', async () => {
    const Harness: FC = () => {
      const [artifactUrl, setArtifactUrl] = useState('default/job#results');
      return (
        <>
          <button onClick={() => setArtifactUrl('default/other#results')}>swap artifact</button>
          <ResultsPreviewPanel workspace="default" artifactUrl={artifactUrl} />
        </>
      );
    };
    render(<Harness />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Details' })[0]);
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveAttribute('open');

    fireEvent.click(screen.getByRole('button', { name: 'swap artifact' }));

    await waitFor(() => expect(dialog).not.toHaveAttribute('open'));
  });
});
