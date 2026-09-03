// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ImportTracesModal } from '@studio/components/ImportTracesModal';
import { ingestTraceFiles } from '@studio/components/ImportTracesModal/ingestTraceFiles';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { renderRoute, screen } from '@studio/tests/util/render';
import { fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

vi.mock('@studio/components/ImportTracesModal/ingestTraceFiles', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  ingestTraceFiles: vi.fn(),
}));

const ingest = vi.mocked(ingestTraceFiles);

const workspace = workspace1.workspace;

const renderModal = (onClose = () => {}) =>
  renderRoute(
    <ImportTracesModal open onClose={onClose} workspace={workspace} agent="email-triage" />
  );

const jsonFile = (name: string, value: unknown) =>
  new File([JSON.stringify(value)], name, { type: 'application/json' });

const atifFile = jsonFile('trace.json', {
  schema_version: 'ATIF-v1.5',
  agent: { name: 'email-triage' },
  steps: [],
});

const spansFile = jsonFile('spans.json', [
  { span_id: 's1', trace_id: 't1', started_at: '2026-01-01T00:00:00Z' },
]);

/** The picker input is hidden behind a button, so files are handed to it directly. */
const choose = async (...files: File[]) => {
  const input = await screen.findByTestId('trace-files-input');
  fireEvent.change(input, { target: { files } });
};

/** Takes the files tab, picks `files`, and turns the insights trigger off. */
const startImport = async (user: ReturnType<typeof userEvent.setup>, ...files: File[]) => {
  await user.click(await screen.findByRole('tab', { name: 'Select files' }));
  await choose(...files);
  await user.click(await screen.findByRole('checkbox', { name: /Run insights analysis/ }));
  await user.click(screen.getByRole('button', { name: /Import/ }));
};

beforeEach(() => {
  ingest.mockReset();
  ingest.mockResolvedValue({ results: [], agents: [] });
});

describe('ImportTracesModal', () => {
  it('opens on the skill handoff, with no import action to take from Studio', async () => {
    renderModal();

    expect(await screen.findByText('Coding agent prompt')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Import$/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Upload files' })).not.toBeInTheDocument();
  });

  it('switches to the file picker when files are chosen as the method', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('tab', { name: 'Select files' }));

    expect(await screen.findByRole('button', { name: 'Upload files' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Import/ })).toBeDisabled();
    expect(screen.queryByText('Coding agent prompt')).not.toBeInTheDocument();
  });

  it('does not offer pasting trace JSON', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('tab', { name: 'Select files' }));

    expect(screen.queryByRole('textbox', { name: /paste/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/Or paste JSON/)).not.toBeInTheDocument();
  });

  it('tags each chosen file with the format detected for it', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('tab', { name: 'Select files' }));
    await choose(atifFile, spansFile);

    expect(await screen.findByText('trace.json · ATIF')).toBeInTheDocument();
    expect(screen.getByText('spans.json · Spans')).toBeInTheDocument();
    expect(screen.queryByText('No files chosen')).not.toBeInTheDocument();
  });

  it('removes a single file when its tag is clicked, leaving the rest', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('tab', { name: 'Select files' }));
    await choose(atifFile, spansFile);

    await user.click(await screen.findByRole('button', { name: 'Remove trace.json' }));

    await waitFor(() => expect(screen.queryByText('trace.json · ATIF')).not.toBeInTheDocument());
    expect(screen.getByText('spans.json · Spans')).toBeInTheDocument();
  });

  it('explains a file it cannot route, and refuses to import it', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('tab', { name: 'Select files' }));
    await choose(jsonFile('otlp.json', { resourceSpans: [] }));

    expect(await screen.findByText(/OTLP JSON cannot be uploaded/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Import/ })).toBeDisabled();
  });

  it('asks for a span source only once a spans file is chosen', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('tab', { name: 'Select files' }));
    await choose(atifFile);
    expect(screen.queryByLabelText('Span source')).not.toBeInTheDocument();

    await choose(spansFile);
    expect(await screen.findByLabelText('Span source')).toHaveValue('studio-upload');
  });

  it('closes itself once everything imported', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    ingest.mockResolvedValue({
      results: [{ label: 'trace.json', status: 'success', message: '1 trajectory imported.' }],
      agents: ['email-triage'],
    });
    renderModal(onClose);

    await startImport(user, atifFile);

    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it('stays open on a failure so its reason can be read', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    ingest.mockResolvedValue({
      results: [{ label: 'trace.json', status: 'error', message: 'Intake rejected step 2.' }],
      agents: [],
    });
    renderModal(onClose);

    await startImport(user, atifFile);

    expect(await screen.findByText('Intake rejected step 2.')).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('keeps the stored model pair after the modal is closed and reopened', async () => {
    const user = userEvent.setup();

    /** Owns `open` the way the routes that host the modal do. */
    const Harness = () => {
      const [open, setOpen] = useState(true);
      return (
        <>
          <button onClick={() => setOpen(true)}>Reopen</button>
          <ImportTracesModal
            open={open}
            onClose={() => setOpen(false)}
            workspace={workspace}
            agent="email-triage"
          />
        </>
      );
    };
    renderRoute(<Harness />);

    await user.click(await screen.findByRole('tab', { name: 'Select files' }));
    await waitFor(() =>
      expect(screen.queryByText('Select a default model')).not.toBeInTheDocument()
    );

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(await screen.findByRole('button', { name: 'Reopen' }));
    await user.click(await screen.findByRole('tab', { name: 'Select files' }));

    expect(screen.queryByText('Select a default model')).not.toBeInTheDocument();
    expect(screen.queryByText('Select a fast model')).not.toBeInTheDocument();
  });
});
