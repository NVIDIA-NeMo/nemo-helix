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
import { MemoryRouter } from 'react-router';
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

/**
 * The conflict banner and the reuse note both link to the fileset they name, so these
 * renders need a router — outside `TestProviders`, because the note is a toast and the
 * mock toaster renders it from there rather than from the component's own subtree.
 */
const renderStart = (onContinue: Mock = vi.fn()) =>
  render(
    <MemoryRouter>
      <TestProviders>
        <CreateCustomizationStart workspace={DEFAULT_WORKSPACE} onContinue={onContinue} />
      </TestProviders>
    </MemoryRouter>
  );

const continueButton = () => screen.getByRole('button', { name: /continue/i });

const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;

interface FakeFileset {
  purpose: string;
  storage: Record<string, unknown>;
  files: Map<string, string>;
}

/** A converted-rows file holding exactly what every recipe's `convertRow` emits. */
const convertedJsonl = (rows = 3) =>
  Array.from({ length: rows }, () => JSON.stringify({ prompt: 'p', completion: 'c' })).join('\n') +
  '\n';

/**
 * An in-memory files service: creates 409 on a name already held, reads 404 when nothing
 * holds it, and uploads land where a later read can see them.
 *
 * The default handler in `mocks/handlers/filesets` answers every read with a generic
 * fileset, which cannot express "this name is free" — and free-vs-taken is the whole
 * subject of these tests. Modelling the store also lets a recipe be run twice against one
 * workspace and have the second run see what the first left behind.
 */
const useFilesetStore = (seed: Record<string, Partial<FakeFileset>> = {}) => {
  const filesets = new Map<string, FakeFileset>(
    Object.entries(seed).map(([name, fileset]) => [
      name,
      {
        purpose: fileset.purpose ?? 'dataset',
        storage: fileset.storage ?? { type: 'local' },
        files: fileset.files ?? new Map(),
      },
    ])
  );

  const asOutput = (name: string, fileset: FakeFileset) => ({
    id: `${DEFAULT_WORKSPACE}/${name}`,
    name,
    workspace: DEFAULT_WORKSPACE,
    description: '',
    purpose: fileset.purpose,
    storage: fileset.storage,
    metadata: {},
    custom_fields: {},
    project: DEFAULT_WORKSPACE,
    created_at: '2026-10-07T00:00:00Z',
    updated_at: '2026-10-07T00:00:00Z',
  });

  server.use(
    http.post(FILESETS_URL, async ({ request }) => {
      const body = (await request.json()) as {
        name: string;
        purpose?: string;
        storage?: Record<string, unknown>;
      };
      if (filesets.has(body.name)) return new HttpResponse(null, { status: 409 });
      const created: FakeFileset = {
        purpose: body.purpose ?? 'dataset',
        storage: body.storage ?? { type: 'local' },
        files: new Map(),
      };
      filesets.set(body.name, created);
      return HttpResponse.json(asOutput(body.name, created));
    }),
    http.get(`${FILESETS_URL}/:name`, ({ params }) => {
      const name = String(params.name);
      const fileset = filesets.get(name);
      if (!fileset) return new HttpResponse(null, { status: 404 });
      return HttpResponse.json(asOutput(name, fileset));
    }),
    http.delete(`${FILESETS_URL}/:name`, ({ params }) => {
      const name = String(params.name);
      const fileset = filesets.get(name);
      if (!fileset) return new HttpResponse(null, { status: 404 });
      filesets.delete(name);
      return HttpResponse.json(asOutput(name, fileset));
    }),
    http.get(`${FILESETS_URL}/:name/files`, ({ params }) => {
      const fileset = filesets.get(String(params.name));
      if (!fileset) return new HttpResponse(null, { status: 404 });
      return HttpResponse.json({
        data: [...fileset.files].map(([path, body]) => ({
          file_ref: path,
          file_url: path,
          path,
          size: body.length,
        })),
      });
    }),
    http.put(`${FILESETS_URL}/:name/-/:path`, async ({ params, request }) => {
      const fileset = filesets.get(String(params.name));
      if (!fileset) return new HttpResponse(null, { status: 404 });
      const path = decodeURIComponent(String(params.path));
      fileset.files.set(path, await request.text());
      return HttpResponse.json({ path });
    }),
    http.get(`${FILESETS_URL}/:name/-/:path`, ({ params }) => {
      const fileset = filesets.get(String(params.name));
      const body = fileset?.files.get(decodeURIComponent(String(params.path)));
      if (body === undefined) return new HttpResponse(null, { status: 404 });
      return new HttpResponse(body);
    })
  );

  return filesets;
};

/** A fileset that setup should accept as its own finished work. */
const reusableDataset = (storage: Record<string, unknown> = { type: 'local' }): FakeFileset => ({
  purpose: 'dataset',
  storage,
  files: new Map([
    ['training.jsonl', convertedJsonl()],
    ['validation.jsonl', convertedJsonl()],
  ]),
});

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
    // Start every test from an empty workspace. Tests that need something already in
    // place call `useFilesetStore` again with a seed, which takes precedence.
    useFilesetStore();
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

    /**
     * Setup runs long enough that walking away part-way through is realistic, and nothing
     * stops it. The promise resolves either way and `onContinue` navigates, so a finished
     * setup must not pull a user who has already left back to the form.
     */
    it('does not continue when the picker unmounts part-way through setup', async () => {
      let releaseRows: () => void = () => {};
      serveRows(
        () =>
          new Promise((resolve) => {
            releaseRows = () => resolve(Array.from({ length: ROW_SUPPLY }, () => ({ ...HF_ROW })));
          })
      );

      // The uploads are the last thing setup does before handing values over, so seeing
      // both is proof the run carried on to completion after the unmount.
      let uploads = 0;
      server.use(
        http.put(`${FILESETS_URL}/:name/-/:path`, () => {
          uploads += 1;
          return HttpResponse.json({ path: 'ok' });
        })
      );

      const user = userEvent.setup();
      const onContinue = vi.fn();
      const { unmount } = renderStart(onContinue);

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      // Leave only once setup is genuinely in flight, blocked on the dataset read.
      await waitFor(() => expect(rowsOptions).toHaveBeenCalled());
      unmount();
      releaseRows();

      await waitFor(() => expect(uploads).toBe(2), { timeout: 10_000 });
      expect(onContinue).not.toHaveBeenCalled();
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
      useFilesetStore({
        [dataset.sourceFilesetName]: {
          storage: { type: 'huggingface', repo_id: 'someone/else' },
        },
      });
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      expect(await screen.findByText(/points at "someone\/else"/i)).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    /** External storage rejects writes, so the uploads would fail opaquely. */
    it('refuses to upload into a converted fileset backed by external storage', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({
        [dataset.name]: { storage: { type: 'huggingface', repo_id: dataset.hfRepoId } },
      });
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      expect(await screen.findByText(/does not accept uploads/i)).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    /**
     * The reported dead end. A fileset created with no explicit storage lands on the
     * deployment's default backend, which is S3 on a hosted install — so the recipe was
     * rejecting the very fileset its own previous run had created, and every user after
     * the first in a shared workspace was blocked.
     */
    it('reuses its own S3-backed fileset rather than calling S3 unwritable', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({ [dataset.name]: reusableDataset({ type: 's3', bucket: 'nmp' }) });
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      await waitFor(() =>
        expect(onContinue).toHaveBeenCalledWith(expect.objectContaining({ optionId: 'template' }))
      );
      expect(screen.queryByText(/does not accept uploads/i)).not.toBeInTheDocument();
    });

    it('reuses a compatible fileset without downloading the dataset again', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({ [dataset.name]: reusableDataset() });
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      await waitFor(() => expect(onContinue).toHaveBeenCalled());
      const [[selection]] = onContinue.mock.calls;
      expect(selection.initialValues.automodel.dataset.training).toBe(
        `${DEFAULT_WORKSPACE}/${dataset.name}`
      );
      // The download is the expensive part, and reuse exists to skip it.
      expect(rowsOptions).not.toHaveBeenCalled();
    });

    /**
     * Which fileset the job trains on is the point of the recipe, so a silent reuse is
     * not good enough — and it has to survive the hand-over, which navigates away.
     */
    it('says which existing fileset it reused', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({ [dataset.name]: reusableDataset() });

      await provisionSelectedTemplate(vi.fn());

      const note = await screen.findByText(/reused the existing dataset/i);
      expect(note).toHaveTextContent(`${DEFAULT_WORKSPACE}/${dataset.name}`);
      expect(
        screen.getByRole('link', { name: `${DEFAULT_WORKSPACE}/${dataset.name}` })
      ).toBeInTheDocument();
    });

    /** Running the same recipe twice in the same workspace has to work both times. */
    it('succeeds on a second run in the same workspace', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      const filesets = useFilesetStore();

      const first = vi.fn();
      const { unmount } = renderStart(first);
      const user = userEvent.setup();
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());
      await waitFor(() => expect(first).toHaveBeenCalled(), { timeout: 10_000 });
      expect(filesets.get(dataset.name)?.files.has('training.jsonl')).toBe(true);
      unmount();

      const second = vi.fn();
      await provisionSelectedTemplate(second);
      await waitFor(() =>
        expect(second).toHaveBeenCalledWith(expect.objectContaining({ optionId: 'template' }))
      );
      expect(screen.queryByText(/already exists/i)).not.toBeInTheDocument();
    });

    it('treats a half-written fileset as a conflict rather than reusing it', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({
        [dataset.name]: {
          files: new Map([['training.jsonl', convertedJsonl()]]),
        },
      });
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      expect(await screen.findByText(/missing validation.jsonl/i)).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    it('treats a fileset holding other rows as a conflict rather than reusing it', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({
        [dataset.name]: {
          files: new Map([
            ['training.jsonl', `${JSON.stringify({ text: 'something else' })}\n`],
            ['validation.jsonl', convertedJsonl()],
          ]),
        },
      });
      const onContinue = vi.fn();

      await provisionSelectedTemplate(onContinue);

      expect(await screen.findByText(/not the prompt\/completion pairs/i)).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    /** Waiting out a multi-megabyte download only to be told it cannot land is the dead end. */
    it('reports a conflict before downloading anything', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({
        [dataset.name]: { storage: { type: 'huggingface', repo_id: dataset.hfRepoId } },
      });

      await provisionSelectedTemplate(vi.fn());

      expect(await screen.findByText(/does not accept uploads/i)).toBeInTheDocument();
      expect(rowsOptions).not.toHaveBeenCalled();
    });

    it('offers a way out of a conflict, and links to what is in the way', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      useFilesetStore({
        [dataset.name]: { storage: { type: 'huggingface', repo_id: dataset.hfRepoId } },
      });

      await provisionSelectedTemplate(vi.fn());

      await screen.findByText(/does not accept uploads/i);
      expect(
        screen.getByRole('link', { name: `${DEFAULT_WORKSPACE}/${dataset.name}` })
      ).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /open fileset/i })).toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /create under a different name/i })
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /replace it/i })).toBeInTheDocument();
    });

    it('builds the dataset under a free name when asked to', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      const filesets = useFilesetStore({
        [dataset.name]: { storage: { type: 'huggingface', repo_id: dataset.hfRepoId } },
      });
      const onContinue = vi.fn();
      const user = userEvent.setup();

      await provisionSelectedTemplate(onContinue);
      await user.click(
        await screen.findByRole('button', { name: /create under a different name/i })
      );

      await waitFor(() => expect(onContinue).toHaveBeenCalled(), { timeout: 10_000 });
      const [[selection]] = onContinue.mock.calls;
      expect(selection.initialValues.automodel.dataset.training).toBe(
        `${DEFAULT_WORKSPACE}/${dataset.name}-2`
      );
      // The fileset that was in the way is left exactly as it was found.
      expect(filesets.get(dataset.name)?.storage).toEqual({
        type: 'huggingface',
        repo_id: dataset.hfRepoId,
      });
    });

    it('rebuilds the dataset in place when the user confirms a replace', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      const filesets = useFilesetStore({
        [dataset.name]: { storage: { type: 'huggingface', repo_id: dataset.hfRepoId } },
      });
      const onContinue = vi.fn();
      const user = userEvent.setup();

      await provisionSelectedTemplate(onContinue);
      await user.click(await screen.findByRole('button', { name: /replace it/i }));
      await user.click(await screen.findByRole('button', { name: /delete and rebuild/i }));

      await waitFor(() => expect(onContinue).toHaveBeenCalled(), { timeout: 10_000 });
      expect(filesets.get(dataset.name)?.storage).toEqual({ type: 'local' });
      expect(filesets.get(dataset.name)?.files.has('validation.jsonl')).toBe(true);
    });

    it('does not delete anything until the replace is confirmed', async () => {
      const { dataset } = CUSTOMIZATION_TEMPLATES[0];
      const filesets = useFilesetStore({
        [dataset.name]: { storage: { type: 'huggingface', repo_id: dataset.hfRepoId } },
      });
      const user = userEvent.setup();

      await provisionSelectedTemplate(vi.fn());
      await user.click(await screen.findByRole('button', { name: /replace it/i }));
      await user.click(await screen.findByRole('button', { name: /cancel/i }));

      expect(filesets.get(dataset.name)?.storage).toEqual({
        type: 'huggingface',
        repo_id: dataset.hfRepoId,
      });
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
