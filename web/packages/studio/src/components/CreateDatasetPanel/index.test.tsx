// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { CreateDatasetPanel } from '@studio/components/CreateDatasetPanel';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { render, screen } from '@studio/tests/util/render';
import { fireEvent, waitFor, within } from '@testing-library/react';
import userEvent, { type UserEvent } from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';
import { useState } from 'react';

// DOM contract (see the panel implementation): tabs HuggingFace/NGC/GitHub/S3/Upload; textboxes
// "HuggingFace URL" | "NGC Catalog URL" | "GitHub URL" | "Location", "Name", "Description"; secret
// comboboxes "Token secret" | "API key secret" | "Access key ID secret" | "Secret access key secret";
// buttons "Create Dataset" ("Creating…" while pending), "Cancel", "Choose files", "Choose folder";
// hidden <input type=file> test ids create-dataset-files-input / create-dataset-folder-input.

const WS = 'default';
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const SECRETS_URL = `${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets`;

const HF_URL = 'https://huggingface.co/datasets/tatsu-lab/alpaca';
const GITHUB_URL = 'https://github.com/tatsu-lab/stanford_alpaca';
const NGC_URL =
  'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemotron-personas/resources/nemotron-personas-dataset-en_us/versions/0.0.2';

const URL_ERRORS = {
  hf: 'Enter a full HuggingFace repository URL',
  ngc: 'Enter an NGC catalog URL with an organization, team, and asset',
  invalidHf: 'Invalid URL source. Please enter a valid HuggingFace URL',
};
const NAME_REQUIRED = 'Enter a name for this Dataset';

const filesetRecord = (name: string, purpose = 'dataset'): FilesetOutput =>
  ({
    id: `fs-${name}`,
    name,
    workspace: WS,
    description: '',
    purpose,
    storage: { type: 'local' },
    metadata: {},
    custom_fields: {},
    project: WS,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  }) as FilesetOutput;

type Body = Record<string, unknown> & { storage?: Record<string, unknown> };

/** Records create POST bodies and upload PUT paths; returns the created record. */
function recordRequests() {
  const posts: Body[] = [];
  const puts: string[] = [];
  server.use(
    http.post(FILESETS_URL, async ({ request }) => {
      const body = (await request.json()) as Body;
      posts.push(body);
      return HttpResponse.json(filesetRecord(body.name as string));
    }),
    http.put(`${FILESETS_URL}/:name/-/*`, ({ request }) => {
      puts.push(decodeURIComponent(new URL(request.url).pathname.split('/-/')[1]));
      return new HttpResponse(null, { status: 200 });
    })
  );
  return { posts, puts };
}

/** MSW answer for the `filter[name]` uniqueness check: `existing` names are taken (any purpose). */
const existingFilesets = (...existing: FilesetOutput[]) =>
  http.get(FILESETS_URL, ({ request }) => {
    const name = new URL(request.url).searchParams.get('filter[name]');
    const data = existing.filter((fs) => fs.name === name);
    return HttpResponse.json({
      data,
      pagination: {
        page: 1,
        page_size: 1,
        current_page_size: data.length,
        total_pages: 1,
        total_results: data.length,
      },
    });
  });

const noSecrets = () =>
  http.get(SECRETS_URL, () =>
    HttpResponse.json({
      data: [],
      pagination: {
        page: 1,
        page_size: 20,
        current_page_size: 0,
        total_pages: 1,
        total_results: 0,
      },
    })
  );

// Hosts the panel like a page would, so tests can close and reopen it.
function Host({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (f: FilesetOutput) => void;
}) {
  const [open, setOpen] = useState(true);
  return (
    <>
      <button onClick={() => setOpen(true)}>Open host panel</button>
      <CreateDatasetPanel
        open={open}
        workspace={WS}
        onClose={() => {
          onClose();
          setOpen(false);
        }}
        onCreated={onCreated}
      />
    </>
  );
}

function setup() {
  const onClose = vi.fn();
  const onCreated = vi.fn();
  const user = userEvent.setup();
  render(<Host onClose={onClose} onCreated={onCreated} />);
  return { user, onClose, onCreated };
}

const field = (name: string) => screen.getByRole('textbox', { name });
const createButton = () => screen.getByRole('button', { name: 'Create Dataset' });

const openTab = (user: UserEvent, name: string) => user.click(screen.getByRole('tab', { name }));

async function paste(user: UserEvent, name: string, text: string) {
  const input = field(name);
  await user.clear(input);
  await user.click(input);
  await user.paste(text);
}

async function pickSecret(user: UserEvent, combobox: string, secret: string) {
  await user.click(await screen.findByRole('combobox', { name: combobox }));
  await user.click(await screen.findByRole('option', { name: secret }));
}

const makeFile = (relativePath: string, contents = 'x') => {
  const file = new File([contents], relativePath.split('/').pop() ?? relativePath);
  if (relativePath.includes('/')) {
    Object.defineProperty(file, 'webkitRelativePath', { value: relativePath });
  }
  return file;
};

const selectFiles = (testId: string, ...paths: string[]) =>
  fireEvent.change(screen.getByTestId(testId), {
    target: { files: paths.map((p) => makeFile(p)) },
  });

describe('CreateDatasetPanel', () => {
  beforeEach(() => {
    server.use(existingFilesets(filesetRecord('existing-generic', 'generic')));
  });

  it('validates an untouched HuggingFace tab, rejects bare ids, and times URL errors', async () => {
    const { posts } = recordRequests();
    const { user, onCreated } = setup();

    // Tab order and default tab (spec 2)
    const tabs = await screen.findAllByRole('tab');
    expect(tabs.map((t) => t.textContent)).toEqual([
      'HuggingFace',
      'NGC',
      'GitHub',
      'S3',
      'Upload',
    ]);
    expect(screen.getByRole('tab', { name: 'HuggingFace' })).toHaveAttribute(
      'aria-selected',
      'true'
    );

    // Untouched: no errors, Create available (11.1 / 11.2)
    expect(screen.queryByText(NAME_REQUIRED)).not.toBeInTheDocument();
    expect(createButton()).toBeEnabled();

    // Submit untouched: both errors, no POST (spec 14)
    await user.click(createButton());
    expect(await screen.findByText(NAME_REQUIRED)).toBeInTheDocument();
    expect(screen.getByText(URL_ERRORS.hf)).toBeInTheDocument();

    // Bare owner/name and scheme-less URLs are rejected after blur
    await user.type(field('HuggingFace URL'), 'tatsu-lab/alpaca');
    await user.tab();
    expect(await screen.findByText(URL_ERRORS.invalidHf)).toBeInTheDocument();
    await user.clear(field('HuggingFace URL'));
    await user.type(field('HuggingFace URL'), 'huggingface.co/datasets/tatsu-lab/alpaca');
    await user.tab();
    expect(await screen.findByText(URL_ERRORS.invalidHf)).toBeInTheDocument();

    await user.click(createButton());
    expect(posts).toHaveLength(0);
    expect(onCreated).not.toHaveBeenCalled();
  });

  it('suppresses the URL error while typing but validates a paste immediately', async () => {
    const { user } = setup();
    const url = await screen.findByRole('textbox', { name: 'HuggingFace URL' });

    await user.type(url, 'tatsu-lab/alpaca');
    expect(screen.queryByText(URL_ERRORS.invalidHf)).not.toBeInTheDocument();
    await user.tab();
    expect(await screen.findByText(URL_ERRORS.invalidHf)).toBeInTheDocument();

    // Fresh draft path: a pasted invalid value errors without a blur
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Open host panel' }));
    await paste(user, 'HuggingFace URL', 'tatsu-lab/alpaca');
    expect(await screen.findByText(URL_ERRORS.invalidHf)).toBeInTheDocument();
  });

  it('infers from a pasted HF URL, handles wrong-tab links, revision reset, manual names, and submits', async () => {
    const { posts } = recordRequests();
    const { user, onCreated, onClose } = setup();
    await screen.findByRole('tab', { name: 'HuggingFace' });

    // Paste infers the name (spec 14)
    await paste(user, 'HuggingFace URL', HF_URL);
    await waitFor(() => expect(field('Name')).toHaveValue('alpaca'));

    // GitHub URL pasted into HuggingFace: link switches tabs without copying the URL
    await paste(user, 'HuggingFace URL', GITHUB_URL);
    expect(
      await screen.findByText(/Trying to import from GitHub\? Switch to GitHub tab\./)
    ).toBeInTheDocument();
    await user.click(screen.getByText('GitHub', { selector: 'a, button' }));
    expect(screen.getByRole('tab', { name: 'GitHub' })).toHaveAttribute('aria-selected', 'true');
    expect(field('GitHub URL')).toHaveValue('');

    // Back to HuggingFace: the draft is preserved. A revision-specific URL, then the root URL.
    await openTab(user, 'HuggingFace');
    expect(field('HuggingFace URL')).toHaveValue(GITHUB_URL);
    await paste(user, 'HuggingFace URL', `${HF_URL}/tree/dev`);
    await paste(user, 'HuggingFace URL', HF_URL);

    // A manual name survives later URL changes (nameTouched)
    await user.clear(field('Name'));
    await user.type(field('Name'), 'zz-my-set');
    await paste(user, 'HuggingFace URL', `${HF_URL}/tree/dev`);
    await paste(user, 'HuggingFace URL', HF_URL);
    expect(field('Name')).toHaveValue('zz-my-set');

    // Advanced starts collapsed, yet its default is submitted
    expect(screen.queryByRole('textbox', { name: 'Endpoint' })).not.toBeInTheDocument();

    await user.click(createButton());
    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));
    expect(posts).toHaveLength(1);
    expect(posts[0]).toMatchObject({
      name: 'zz-my-set',
      purpose: 'dataset',
      storage: {
        type: 'huggingface',
        repo_id: 'tatsu-lab/alpaca',
        repo_type: 'dataset',
        revision: 'main', // not the stale `dev`
        endpoint: 'https://huggingface.co',
      },
    });
    expect(posts[0]).not.toHaveProperty('cache');
    expect(JSON.stringify(posts[0])).not.toMatch(/hf_ref|token_secret/);
    expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ name: 'zz-my-set' }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it('shows the exact name errors on blur, flags duplicates across purposes, and gates Create', async () => {
    const { user } = setup();
    await screen.findByRole('tab', { name: 'HuggingFace' });
    await paste(user, 'HuggingFace URL', HF_URL);

    const cases: [string, string][] = [
      ['a', 'Names must be at least 2 characters'],
      ['1data', 'Names must start with a lowercase letter'],
      ['Data', 'Names must start with a lowercase letter'],
      ['data--train', "Names can't contain two hyphens in a row"],
      ['data-', "Names can't end with a hyphen"],
      ['a'.repeat(64), 'Names must be 63 characters or fewer'],
    ];
    for (const [name, message] of cases) {
      await user.clear(field('Name'));
      await user.type(field('Name'), name);
      await user.tab(); // blur displays the error
      expect(await screen.findByText(message)).toBeInTheDocument();
      expect(createButton()).toBeDisabled(); // manually edited and invalid
    }

    // Duplicate of a generic-purpose fileset in the same workspace
    await user.clear(field('Name'));
    await user.type(field('Name'), 'existing-generic');
    await user.tab();
    expect(
      await screen.findByText(`A Dataset named existing-generic already exists in ${WS}`)
    ).toBeInTheDocument();
    expect(createButton()).toBeDisabled();

    await user.clear(field('Name'));
    await user.type(field('Name'), 'fresh-name');
    await waitFor(() => expect(createButton()).toBeEnabled());
  });

  it('preserves drafts across tabs, validates only the active tab, and submits NGC with its secret', async () => {
    const { posts } = recordRequests();
    const { user, onCreated } = setup();
    await screen.findByRole('tab', { name: 'HuggingFace' });

    await user.type(field('Description'), 'hf draft description');
    await openTab(user, 'NGC');

    // NGC URL without a team: NGC URL error, no POST (HuggingFace is incomplete but irrelevant)
    await paste(
      user,
      'NGC Catalog URL',
      'https://catalog.ngc.nvidia.com/orgs/nvidia/resources/example'
    );
    await user.click(createButton());
    expect(await screen.findByText(URL_ERRORS.ngc)).toBeInTheDocument();
    expect(screen.queryByText(URL_ERRORS.hf)).not.toBeInTheDocument();
    expect(posts).toHaveLength(0);

    // Drafts survive switching away and back
    await openTab(user, 'HuggingFace');
    expect(field('Description')).toHaveValue('hf draft description');
    await openTab(user, 'NGC');

    // Valid URL: name inferred, secret required
    await paste(user, 'NGC Catalog URL', NGC_URL);
    await waitFor(() => expect(field('Name')).toHaveValue('nemotron-personas-dataset-en-us'));
    await user.click(createButton());
    expect(await screen.findByText('Select api key secret to continue.')).toBeInTheDocument();
    expect(posts).toHaveLength(0);

    await pickSecret(user, 'API key secret', 'huggingface-token');
    await user.click(createButton());
    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));
    expect(posts[0]).toMatchObject({
      name: 'nemotron-personas-dataset-en-us',
      description: '',
      purpose: 'dataset',
      storage: {
        type: 'ngc',
        org: 'nvidia',
        team: 'nemotron-personas',
        target: 'nemotron-personas-dataset-en_us',
        target_type: 'resource',
        version: '0.0.2',
        api_key_secret: 'huggingface-token',
        host: 'https://api.ngc.nvidia.com',
      },
    });
    expect(posts[0]).not.toHaveProperty('cache');
  });

  it('requires both S3 secrets, has no SDK-auth checkbox, and submits secret refs with use_sdk_auth false', async () => {
    const { posts } = recordRequests();
    const { user, onCreated } = setup();
    await openTab(user, 'S3');

    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Endpoint URL' })).not.toBeInTheDocument(); // Advanced collapsed

    await paste(user, 'Location', 's3://my-bucket/data/train/');
    await waitFor(() => expect(field('Name')).toHaveValue('train'));
    await user.click(createButton());
    expect(await screen.findByText('Select access key id secret to continue.')).toBeInTheDocument();
    expect(screen.getByText('Select secret access key secret to continue.')).toBeInTheDocument();
    expect(posts).toHaveLength(0);

    await pickSecret(user, 'Access key ID secret', 'openai-api-key');
    await pickSecret(user, 'Secret access key secret', 'anthropic-api-key');
    await user.click(createButton());
    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));
    expect(posts[0]).toMatchObject({
      name: 'train',
      purpose: 'dataset',
      storage: {
        type: 's3',
        bucket: 'my-bucket',
        prefix: 'data/train',
        use_sdk_auth: false,
        access_key_id_secret: 'openai-api-key',
        secret_access_key_secret: 'anthropic-api-key',
      },
    });
  });

  it('disables a required secret selector when the workspace has no secrets', async () => {
    server.use(noSecrets());
    const { posts } = recordRequests();
    const { user } = setup();
    await openTab(user, 'NGC');
    await paste(user, 'NGC Catalog URL', NGC_URL);

    const selector = await screen.findByRole('combobox', { name: 'API key secret' });
    await waitFor(() => expect(selector).toBeDisabled());
    expect(screen.getByText('No secrets available')).toBeInTheDocument();
    expect(screen.getByText(/Create one with .*nemo secrets create <name>/)).toBeInTheDocument();

    await user.click(createButton());
    expect(
      await screen.findByText('Create a secret in this workspace, then select it to continue.')
    ).toBeInTheDocument();
    expect(posts).toHaveLength(0);
  });

  it('creates a public GitHub dataset without a token, showing Creating… while pending', async () => {
    const { posts } = recordRequests();
    server.use(
      http.post(FILESETS_URL, async ({ request }) => {
        const body = (await request.json()) as Body;
        posts.push(body);
        await delay(300);
        return HttpResponse.json(filesetRecord(body.name as string));
      })
    );
    const { user, onCreated } = setup();
    await openTab(user, 'GitHub');
    await paste(user, 'GitHub URL', GITHUB_URL);
    await waitFor(() => expect(field('Name')).toHaveValue('stanford-alpaca'));

    await user.click(createButton());
    expect(await screen.findByRole('button', { name: 'Creating…' })).toBeDisabled();
    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));
    expect(posts).toHaveLength(1);
    expect(posts[0]).toMatchObject({
      name: 'stanford-alpaca',
      purpose: 'dataset',
      storage: {
        type: 'github',
        owner: 'tatsu-lab',
        repo: 'stanford_alpaca',
        revision: 'HEAD',
        api_base_url: 'https://api.github.com',
      },
    });
    expect(posts[0].storage).not.toHaveProperty('token_secret');
    expect(posts[0]).not.toHaveProperty('cache');
  });

  it('validates Upload selection, preserves folder paths, omits storage and uploads to relative paths', async () => {
    const { posts, puts } = recordRequests();
    const { user, onCreated } = setup();
    await openTab(user, 'Upload');

    expect(screen.getByText(/platform default storage/i)).toBeInTheDocument();
    await user.click(createButton());
    expect(
      await screen.findByText('Choose at least one file or a folder to upload.')
    ).toBeInTheDocument();
    expect(posts).toHaveLength(0);

    // A loose file with an invalid inferred name errors immediately (no blur), then a valid one infers
    selectFiles('create-dataset-files-input', '1data.jsonl');
    expect(await screen.findByText('Names must start with a lowercase letter')).toBeInTheDocument();
    selectFiles('create-dataset-files-input', 'data.final.jsonl');
    await waitFor(() => expect(field('Name')).toHaveValue('data.final'));

    // Folder selection: nested paths kept, folder name wins, pickers hidden behind the folder card
    selectFiles(
      'create-dataset-folder-input',
      'my-data/train/a.jsonl',
      'my-data/b.jsonl',
      'my-data/train/deep/c.jsonl'
    );
    await waitFor(() => expect(field('Name')).toHaveValue('my-data'));
    expect(screen.getByText(/my-data\/train\/a\.jsonl/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Choose files' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Choose folder' })).not.toBeInTheDocument();

    await user.click(createButton());
    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));
    expect(posts).toHaveLength(1);
    expect(posts[0]).toMatchObject({ name: 'my-data', purpose: 'dataset' });
    expect(posts[0]).not.toHaveProperty('storage');
    expect(posts[0]).not.toHaveProperty('cache');
    expect(puts).toEqual(
      expect.arrayContaining([
        'my-data/train/a.jsonl',
        'my-data/b.jsonl',
        'my-data/train/deep/c.jsonl',
      ])
    );
  });

  it('keeps the panel on outside clicks, and Cancel then reopen clears every draft', async () => {
    const { user, onClose } = setup();
    await screen.findByRole('tab', { name: 'HuggingFace' });

    await paste(user, 'HuggingFace URL', HF_URL);
    await user.type(field('Description'), 'discard me');
    await openTab(user, 'S3');
    await paste(user, 'Location', 's3://my-bucket/data');

    await user.click(document.body); // outside click
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('tab', { name: 'S3' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole('tab', { name: 'S3' })).not.toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: 'Open host panel' }));
    expect(await screen.findByRole('tab', { name: 'HuggingFace' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(field('HuggingFace URL')).toHaveValue('');
    expect(field('Name')).toHaveValue('');
    expect(field('Description')).toHaveValue('');
    await openTab(user, 'S3');
    expect(field('Location')).toHaveValue('');
  });

  it('shows a 409 from create as the name error and keeps the panel open', async () => {
    server.use(
      http.post(FILESETS_URL, () =>
        HttpResponse.json({ detail: 'Fileset already exists' }, { status: 409 })
      )
    );
    const { user, onCreated, onClose } = setup();
    await screen.findByRole('tab', { name: 'HuggingFace' });
    await paste(user, 'HuggingFace URL', HF_URL);
    await waitFor(() => expect(field('Name')).toHaveValue('alpaca'));

    await user.click(createButton());
    expect(
      await screen.findByText(`A Dataset named alpaca already exists in ${WS}`)
    ).toBeInTheDocument();
    expect(onCreated).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    expect(within(document.body).getByRole('tab', { name: 'HuggingFace' })).toBeInTheDocument();
  });
});
