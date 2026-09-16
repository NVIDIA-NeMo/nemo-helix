// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { filesetRowsQueryOptions } from '@studio/api/datasets/filesetParquetRows';
import { CreateCustomizationStart } from '@studio/components/CreateCustomizationStart';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import type { Mock } from 'vitest';

// The fileset read is covered in filesetParquetRows.test.ts; here it only has to produce
// rows (or fail) so the provisioning flow around it can be exercised.
vi.mock('@studio/api/datasets/filesetParquetRows', () => ({
  filesetRowsQueryOptions: vi.fn(),
}));

const rowsOptions = vi.mocked(filesetRowsQueryOptions);

/** A BIRD-SQL row — the shape every shipped recipe's converter reads. */
const HF_ROW = {
  schema: 'CREATE TABLE t (id INT);',
  question: 'q',
  evidence: 'hint',
  SQL: 'SELECT 1',
};

/** Enough rows for the largest partition split any shipped recipe asks for. */
const ROW_SUPPLY = Math.max(
  ...CUSTOMIZATION_TEMPLATES.map((t) => t.dataset.trainingRowCount + t.dataset.validationRowCount)
);

const serveRows = (rows: () => Promise<Record<string, unknown>[]>) => {
  rowsOptions.mockImplementation(() => ({ queryKey: ['test-rows'], queryFn: rows }) as never);
};

const serveDefaultRows = () =>
  serveRows(() => Promise.resolve(Array.from({ length: ROW_SUPPLY }, () => ({ ...HF_ROW }))));

const renderStart = (onContinue: Mock = vi.fn()) => {
  render(
    <TestProviders>
      <CreateCustomizationStart workspace={DEFAULT_WORKSPACE} onContinue={onContinue} />
    </TestProviders>
  );
};

const continueButton = () => screen.getByRole('button', { name: /continue/i });

const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;

/** 409s the named fileset on create, and serves `storage` when it is fetched back. */
const nameAlreadyTaken = (name: string, storage: Record<string, unknown>) => [
  http.post(FILESETS_URL, async ({ request }) => {
    const body = (await request.json()) as { name: string };
    if (body.name !== name) return HttpResponse.json({ name: body.name });
    return new HttpResponse(null, { status: 409 });
  }),
  http.get(`${FILESETS_URL}/:name`, ({ params }) =>
    HttpResponse.json({ name: params.name, workspace: DEFAULT_WORKSPACE, storage })
  ),
];

const provisionSelectedTemplate = async (onContinue: Mock) => {
  const user = userEvent.setup();
  renderStart(onContinue);
  await user.click(screen.getByText('Start from a template'));
  await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
  await user.click(continueButton());
};

describe('CreateCustomizationStart', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseNavigate(vi.fn());
    mockUseParams({ [ROUTE_PARAMS.workspace]: DEFAULT_WORKSPACE });
    serveDefaultRows();
  });

  it('offers a way in for each start option', () => {
    renderStart();
    expect(screen.getByText('Start from a template')).toBeInTheDocument();
    expect(screen.getByText('Build from scratch')).toBeInTheDocument();
  });

  it('keeps Continue disabled until an option is picked', async () => {
    const user = userEvent.setup();
    renderStart();
    expect(screen.queryByRole('button', { name: /continue/i })).not.toBeInTheDocument();

    await user.click(screen.getByText('Build from scratch'));
    expect(continueButton()).toBeEnabled();
  });

  it('hands "from scratch" over without any form values', async () => {
    const user = userEvent.setup();
    const onContinue = vi.fn();
    renderStart(onContinue);

    await user.click(screen.getByText('Build from scratch'));
    await user.click(continueButton());

    expect(onContinue).toHaveBeenCalledWith({ optionId: 'scratch' });
  });

  describe('templates', () => {
    it('shows a card per recipe once the option is picked', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Start from a template'));
      for (const template of CUSTOMIZATION_TEMPLATES) {
        expect(screen.getByText(template.title)).toBeInTheDocument();
      }
    });

    /** Picking the option is not picking a recipe — Continue has nothing to act on yet. */
    it('will not continue until a recipe is selected', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Start from a template'));
      expect(continueButton()).toBeDisabled();

      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      expect(continueButton()).toBeEnabled();
    });

    it('provisions the recipe and hands over the form values it produced', async () => {
      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      await waitFor(
        () =>
          expect(onContinue).toHaveBeenCalledWith(
            expect.objectContaining({
              optionId: 'template',
              initialValues: expect.objectContaining({ backend: 'automodel' }),
            })
          ),
        { timeout: 10_000 }
      );
    });

    /** The source fileset points at HuggingFace; the converted JSONL goes somewhere writable. */
    it('registers the dataset repo as an external fileset before reading it', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      await waitFor(() =>
        expect(rowsOptions).toHaveBeenCalledWith(
          expect.objectContaining({
            workspace: DEFAULT_WORKSPACE,
            filesetName: dataset.sourceFilesetName,
          })
        )
      );
      expect(dataset.sourceFilesetName).not.toBe(dataset.name);
    });

    it('reports a failed dataset read and does not continue', async () => {
      serveRows(() => Promise.reject(new Error('No dataset file matched the recipe pattern.')));
      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      expect(await screen.findByText(/No dataset file matched/i)).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    /**
     * A 409 only says the name is taken, not that it is ours. Left unchecked, the pattern
     * matches nothing in a stranger's fileset and the error never mentions ownership.
     */
    it('refuses to reuse a source fileset pointing at a different repo', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      server.use(
        ...nameAlreadyTaken(dataset.sourceFilesetName, {
          type: 'huggingface',
          repo_id: 'someone/else',
        })
      );
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      expect(await screen.findByText(/points at "someone\/else"/i)).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    /** External storage rejects writes, so the uploads below would fail opaquely. */
    it('refuses to upload into a converted fileset backed by external storage', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      server.use(
        ...nameAlreadyTaken(dataset.name, { type: 'huggingface', repo_id: dataset.hfRepoId })
      );
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      expect(await screen.findByText(/cannot be written to/i)).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    /**
     * Provisioning takes long enough that the cards stay on screen behind a disabled
     * Continue. Changing the selection mid-flight used to leave the finished setup handing
     * the form a recipe the user had moved off.
     *
     * The mocked read would otherwise resolve before a click could land, so the response is
     * held open to make the in-flight window real rather than a race.
     */
    it('ignores clicks on the option cards while setup is running', async () => {
      let releaseRows: () => void = () => {};
      const held = new Promise<void>((resolve) => {
        releaseRows = resolve;
      });
      serveRows(async () => {
        await held;
        return Array.from({ length: ROW_SUPPLY }, () => ({ ...HF_ROW }));
      });

      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      // Setup is now parked on the held read. The cards are still mounted and clickable.
      await user.click(screen.getByText('Build from scratch'));
      expect(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title)).toBeInTheDocument();

      releaseRows();
      await waitFor(
        () =>
          expect(onContinue).toHaveBeenCalledWith(
            expect.objectContaining({ optionId: 'template' })
          ),
        { timeout: 10_000 }
      );
      expect(onContinue).not.toHaveBeenCalledWith({ optionId: 'scratch' });
    });
  });
});
