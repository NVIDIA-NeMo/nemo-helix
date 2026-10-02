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

  it('offers every way in at once', () => {
    renderStart();
    expect(screen.getByText('Build from scratch')).toBeInTheDocument();
    for (const template of CUSTOMIZATION_TEMPLATES) {
      expect(screen.getByText(template.title)).toBeInTheDocument();
    }
  });

  it('keeps Continue disabled until something is picked', async () => {
    const user = userEvent.setup();
    renderStart();
    expect(continueButton()).toBeDisabled();

    await user.click(screen.getByText('Build from scratch'));
    expect(continueButton()).toBeEnabled();
  });

  it('opens on the template rung, and waits for a recipe', async () => {
    const user = userEvent.setup();
    renderStart();

    expect(screen.getByRole('radio', { name: 'Start from a template' })).toBeChecked();
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();
    expect(screen.getByText('Pick a recipe to continue.')).toBeInTheDocument();

    await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
    expect(screen.getByRole('button', { name: /continue/i })).toBeEnabled();
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
    it('arms Continue as soon as a recipe is picked', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      expect(continueButton()).toBeEnabled();
    });

    it('provisions the recipe and hands over the form values it produced', async () => {
      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

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

  describe('saved templates', () => {
    const SAVED = {
      id: 'tpl-1',
      name: 'my-sft-recipe',
      workspace: DEFAULT_WORKSPACE,
      backend: 'automodel',
      description: 'Saved from last week',
      config: {
        model: `${DEFAULT_WORKSPACE}/base-model`,
        dataset: { training: `${DEFAULT_WORKSPACE}/data` },
        training: { training_type: 'sft', finetuning_type: 'lora', max_seq_length: 2048 },
        parallelism: { num_nodes: 1 },
      },
    };

    const serveSaved = (templates: unknown[]) =>
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/customization/v2/workspaces/:workspace/job-templates`,
          () => HttpResponse.json({ data: templates, object: 'list' })
        )
      );

    it('offers saved templates beside the curated recipes', async () => {
      serveSaved([SAVED]);
      renderStart();

      expect(await screen.findByText('my-sft-recipe')).toBeInTheDocument();
      expect(screen.getByText('Saved templates')).toBeInTheDocument();
      // An addition, not a swap.
      expect(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title)).toBeInTheDocument();
    });

    it('hands a saved template over without provisioning anything', async () => {
      serveSaved([SAVED]);
      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

      await user.click(await screen.findByText('my-sft-recipe'));
      await user.click(screen.getByRole('button', { name: /continue/i }));

      await waitFor(() =>
        expect(onContinue).toHaveBeenCalledWith(expect.objectContaining({ optionId: 'template' }))
      );
      const [[selection]] = onContinue.mock.calls;
      expect(selection.initialValues.automodel.model).toBe(`${DEFAULT_WORKSPACE}/base-model`);
    });

    it('offers Delete only once a saved template is picked', async () => {
      serveSaved([SAVED]);
      const user = userEvent.setup();
      renderStart();

      await screen.findByText('my-sft-recipe');
      expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();

      await user.click(screen.getByText('my-sft-recipe'));

      expect(await screen.findByRole('button', { name: /delete/i })).toBeInTheDocument();
    });

    it('leaves the curated recipes undeletable', async () => {
      serveSaved([SAVED]);
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));

      // Curated recipes are code, not entities.
      expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
    });

    it('leaves out a template whose backend the form has no arm for', async () => {
      // Served alongside a good one: waiting for that proves the list arrived, so the
      // absence below is the filter working rather than the fetch not having landed.
      serveSaved([SAVED, { ...SAVED, id: 'tpl-2', name: 'broken-one', backend: 'something-else' }]);
      renderStart();

      expect(await screen.findByText('my-sft-recipe')).toBeInTheDocument();
      expect(screen.queryByText('broken-one')).not.toBeInTheDocument();
    });

    it('reaches templates past the first page', async () => {
      // The endpoint defaults to 20 per page, so a 21st template would otherwise be
      // unreachable — neither selectable nor deletable.
      const onPage = {
        1: { ...SAVED, name: 'page-one' },
        2: { ...SAVED, id: 'tpl-2', name: 'page-two' },
      };
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/customization/v2/workspaces/:workspace/job-templates`,
          ({ request }) => {
            const page = Number(new URL(request.url).searchParams.get('page') ?? 1);
            return HttpResponse.json({
              data: page <= 2 ? [onPage[page as 1 | 2]] : [],
              pagination: {
                page,
                page_size: 1,
                current_page_size: 1,
                total_pages: 2,
                total_results: 2,
              },
            });
          }
        )
      );
      renderStart();

      expect(await screen.findByText('page-two')).toBeInTheDocument();
      expect(screen.getByText('page-one')).toBeInTheDocument();
    });
  });
});
