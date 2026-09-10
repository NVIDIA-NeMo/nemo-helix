// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { NewAgentModal } from '@studio/routes/agents/AgentsListRoute/NewAgentModal';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { getAgentsListRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const FILESET_URL = `${FILESETS_URL}/:name`;
const UPLOAD_URL = `${FILESET_URL}/-/*`;
const AGENTS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`;
const AGENT_URL = `${AGENTS_URL}/:name`;

const FABRIC_YAML = `config_format: nemo-agents-spec-v1
name: calc
description: Adds numbers
`;

const makeFile = (relativePath: string, contents: string): File => {
  const file = new File([contents], relativePath.split('/').pop() ?? relativePath, {
    type: 'text/plain',
  });
  Object.defineProperty(file, 'webkitRelativePath', { value: relativePath });
  return file;
};

/** Blocks its reads until released, so selections can resolve out of order. */
const deferredFile = (relativePath: string, contents: string) => {
  const file = makeFile(relativePath, contents);
  const bytes = new TextEncoder().encode(contents);
  let release = () => {};
  let read = false;
  const open = new Promise<void>((resolve) => {
    release = resolve;
  });

  Object.defineProperty(file, 'arrayBuffer', {
    value: async () => {
      await open;
      read = true;
      return bytes.buffer;
    },
  });
  Object.defineProperty(file, 'text', {
    value: async () => {
      await open;
      read = true;
      return contents;
    },
  });

  return { file, release: () => release(), read: () => read };
};

const DEFAULT_FILES = [
  makeFile('calc-agent/agent.yaml', FABRIC_YAML),
  makeFile('calc-agent/mcps/calculator.py', 'print(1)\n'),
];

interface Scenario {
  filesetExists?: boolean;
  agentExists?: boolean;
}

const mockPlatform = ({ filesetExists = false, agentExists = false }: Scenario = {}) => {
  const uploaded: string[] = [];
  const created: { name?: string }[] = [];
  const filesets: { storage?: unknown }[] = [];
  const deleted: string[] = [];

  server.use(
    http.get(FILESET_URL, ({ params }) =>
      filesetExists
        ? HttpResponse.json({ name: params['name'], workspace })
        : HttpResponse.json({ detail: 'not found' }, { status: 404 })
    ),
    http.get(AGENT_URL, ({ params }) =>
      agentExists
        ? HttpResponse.json({ name: params['name'], workspace })
        : HttpResponse.json({ detail: 'not found' }, { status: 404 })
    ),
    http.delete(FILESET_URL, ({ params }) => {
      deleted.push(String(params['name']));
      return HttpResponse.json({ name: 'deleted' });
    }),
    http.post(FILESETS_URL, async ({ request }) => {
      const body = (await request.json()) as { storage?: unknown };
      filesets.push(body);
      return HttpResponse.json(body);
    }),
    http.put(UPLOAD_URL, ({ request }) => {
      uploaded.push(decodeURIComponent(new URL(request.url).pathname.split('/-/')[1] ?? ''));
      return HttpResponse.json({ path: 'ok' });
    }),
    http.post(AGENTS_URL, async ({ request }) => {
      const body = (await request.json()) as { name?: string };
      created.push(body);
      return HttpResponse.json({ ...body, workspace }, { status: 201 });
    })
  );

  return { uploaded, created, filesets, deleted };
};

const renderModal = () =>
  renderRoute(undefined, {
    history: getAgentsListRoute(workspace),
    routes: [
      {
        path: ROUTES.workspace.agentsList,
        element: <NewAgentModal open onClose={vi.fn()} workspace={workspace} />,
      },
      { path: ROUTES.workspace.agentDetail, element: <div>Agent detail page</div> },
    ],
  });

/** The prompt tab is the default, so upload tests have to switch before the form exists. */
const openUploadTab = async (dialog: HTMLElement) => {
  fireEvent.click(within(dialog).getByRole('tab', { name: 'Upload agent' }));
  await screen.findByTestId('agent-directory-input');
};

const openGitHubTab = async (dialog: HTMLElement) => {
  fireEvent.click(within(dialog).getByRole('tab', { name: 'GitHub repository' }));
  await within(dialog).findByRole('textbox', { name: 'Repository' });
};

const pickDirectory = (dialog: HTMLElement, files: File[] = DEFAULT_FILES) => {
  fireEvent.change(within(dialog).getByTestId('agent-directory-input'), { target: { files } });
};

const submit = async (dialog: HTMLElement, user: ReturnType<typeof userEvent.setup>) => {
  await user.click(within(dialog).getByRole('button', { name: /^(Create|Replace and create)$/ }));
};

describe('NewAgentModal coding agent prompt tab', () => {
  it('opens on the prompt, so an agent already in a repository needs no upload', async () => {
    mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');

    expect(within(dialog).getByRole('tab', { name: 'Coding agent prompt' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(within(dialog).queryByTestId('agent-directory-input')).not.toBeInTheDocument();
  });

  it('copies the integration prompt, left open for the name nothing has assigned yet', async () => {
    // `userEvent.setup()` stubs `navigator.clipboard`, so the prompt is readable back from it.
    const user = userEvent.setup();
    mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');
    // The prompt editor is a lazily imported chunk, so it can miss the default find timeout.
    await user.click(
      await within(dialog).findByRole('button', { name: 'Copy to clipboard' }, { timeout: 10_000 })
    );

    const prompt = await navigator.clipboard.readText();
    expect(prompt).toContain('nemo-agent-config');
    expect(prompt).toContain(workspace);
    // The list page creates nothing up front, so the prompt has to ask for the agent name.
    expect(prompt).toContain('<agent-name>');
  });

  it('offers Close rather than Create, since the prompt has nothing to submit', async () => {
    mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');

    expect(within(dialog).getByRole('button', { name: 'Close' })).toBeInTheDocument();
    expect(within(dialog).queryByRole('button', { name: 'Create' })).not.toBeInTheDocument();
  });

  it('leaves an upload failure behind when the user switches back to the prompt', async () => {
    const user = userEvent.setup();
    mockPlatform({ filesetExists: true, agentExists: true });

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog);
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());
    await submit(dialog, user);
    expect(await within(dialog).findByText(/already owns the fileset/)).toBeInTheDocument();

    await user.click(within(dialog).getByRole('tab', { name: 'Coding agent prompt' }));

    await waitFor(() =>
      expect(within(dialog).queryByText(/already owns the fileset/)).not.toBeInTheDocument()
    );
  });
});

describe('NewAgentModal upload tab', () => {
  it('uploads the picked directory, then creates the agent', async () => {
    const user = userEvent.setup();
    const { uploaded, created } = mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog);
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());

    await submit(dialog, user);

    await waitFor(() => expect(created).toHaveLength(1));
    expect([...uploaded].sort()).toEqual(['agent.yaml', 'mcps/calculator.py']);
    expect(created[0]?.name).toBe('calc');
  });

  it('shows why an owned name is refused instead of a generic failure', async () => {
    const user = userEvent.setup();
    const { created } = mockPlatform({ filesetExists: true, agentExists: true });

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog);
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());

    await submit(dialog, user);

    expect(await within(dialog).findByText(/already owns the fileset/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Create' })).toBeInTheDocument();
    expect(created).toHaveLength(0);
  });

  it('offers to replace an orphaned fileset, and replaces it on the next submit', async () => {
    const user = userEvent.setup();
    const { created } = mockPlatform({ filesetExists: true });

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog);
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());

    await submit(dialog, user);
    expect(await within(dialog).findByText(/no agent owns it/)).toBeInTheDocument();

    const replace = await within(dialog).findByRole('button', { name: 'Replace and create' });
    await user.click(replace);

    await waitFor(() => expect(created).toHaveLength(1));
  });

  it('rejects a directory with no agent.yaml at the top level', async () => {
    mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog, [makeFile('calc-agent/mcps/calculator.py', 'print(1)\n')]);

    expect(await within(dialog).findByText(/No agent\.yaml/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('ignores a slower selection that a newer one replaced', async () => {
    mockPlatform();
    const gate = deferredFile(
      'slow-agent/agent.yaml',
      'config_format: nemo-agents-spec-v1\nname: slow\n'
    );

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog, [gate.file]);
    pickDirectory(dialog, DEFAULT_FILES);
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());

    gate.release();
    await waitFor(() => expect(gate.read()).toBe(true));

    expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument();
    expect(within(dialog).getByText(/calc-agent — 2 files/)).toBeInTheDocument();
  });

  it('cannot submit the previous directory while a new one is validated', async () => {
    mockPlatform();
    const gate = deferredFile(
      'slow-agent/agent.yaml',
      'config_format: nemo-agents-spec-v1\nname: slow\n'
    );

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog);
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());
    expect(within(dialog).getByRole('button', { name: 'Create' })).toBeEnabled();

    pickDirectory(dialog, [gate.file]);

    await waitFor(() =>
      expect(within(dialog).getByRole('button', { name: 'Create' })).toBeDisabled()
    );

    gate.release();
    await waitFor(() => expect(within(dialog).getByDisplayValue('slow')).toBeInTheDocument());
    expect(within(dialog).getByRole('button', { name: 'Create' })).toBeEnabled();
  });

  it('clears the previous failure when a new directory is picked', async () => {
    const user = userEvent.setup();
    mockPlatform({ filesetExists: true, agentExists: true });

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog);
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());
    await submit(dialog, user);
    expect(await within(dialog).findByText(/already owns the fileset/)).toBeInTheDocument();

    pickDirectory(dialog, DEFAULT_FILES);

    await waitFor(() =>
      expect(within(dialog).queryByText(/already owns the fileset/)).not.toBeInTheDocument()
    );
  });

  it('rejects a directory holding a file that is not text', async () => {
    mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);
    pickDirectory(dialog, [
      makeFile('calc-agent/agent.yaml', FABRIC_YAML),
      new File([new Uint8Array([0xff, 0xfe, 0x00])], 'logo.bin'),
    ]);

    expect(await within(dialog).findByText(/is not a text file/)).toBeInTheDocument();
  });
});

describe('NewAgentModal oversized pick', () => {
  it('rejects a directory far larger than an agent, naming the count', async () => {
    mockPlatform();
    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);

    // A real accidental pick was 880k files; only length is read before the guard fires.
    fireEvent.change(within(dialog).getByTestId('agent-directory-input'), {
      target: { files: { length: 880_000 } },
    });

    expect(await within(dialog).findByText(/880,000 files/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Create' })).toBeDisabled();
  });
});

describe('NewAgentModal GitHub import', () => {
  const typeRepo = async (
    dialog: HTMLElement,
    user: ReturnType<typeof userEvent.setup>,
    spec = 'github.com/owner/repo'
  ) => {
    await user.type(within(dialog).getByRole('textbox', { name: 'Repository' }), spec);
    await user.tab();
  };

  it('does not let a typed repository submit from the upload tab', async () => {
    const user = userEvent.setup();
    mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openGitHubTab(dialog);
    await typeRepo(dialog, user);
    await waitFor(() =>
      expect(within(dialog).getByRole('button', { name: 'Create' })).toBeEnabled()
    );

    await openUploadTab(dialog);

    expect(within(dialog).getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('backs the spec fileset with the repository instead of uploading files', async () => {
    const user = userEvent.setup();
    const { uploaded, created, filesets } = mockPlatform();
    server.use(http.get(UPLOAD_URL, () => HttpResponse.text(FABRIC_YAML)));

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openGitHubTab(dialog);
    await typeRepo(dialog, user, 'github.com/owner/repo@v2#agents/calc');
    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());
    await submit(dialog, user);

    await waitFor(() => expect(created).toHaveLength(1));
    expect(uploaded).toHaveLength(0);
    expect(filesets[0]?.storage).toEqual({
      type: 'github',
      owner: 'owner',
      repo: 'repo',
      revision: 'v2',
      path: 'agents/calc',
    });
  });

  it('leaves the repository import for the new agent, not sitting on the open modal', async () => {
    const user = userEvent.setup();
    mockPlatform();
    server.use(http.get(UPLOAD_URL, () => HttpResponse.text(FABRIC_YAML)));

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openGitHubTab(dialog);
    await typeRepo(dialog, user);
    await waitFor(() => expect(within(dialog).getByDisplayValue('repo')).toBeInTheDocument());
    await submit(dialog, user);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(await screen.findByText(/Agent "repo" created/)).toBeInTheDocument();
  });

  it('names the agent after the repository so the fileset name is settled up front', async () => {
    const user = userEvent.setup();
    mockPlatform();
    server.use(http.get(UPLOAD_URL, () => HttpResponse.text(FABRIC_YAML)));

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openGitHubTab(dialog);
    await typeRepo(dialog, user, 'github.com/owner/my-repo');

    await waitFor(() => expect(within(dialog).getByDisplayValue('my-repo')).toBeInTheDocument());
  });

  it('rolls the fileset back when the repository has no agent.yaml', async () => {
    const user = userEvent.setup();
    const { created, deleted } = mockPlatform();
    server.use(
      http.get(UPLOAD_URL, () => HttpResponse.json({ detail: 'not found' }, { status: 404 }))
    );

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openGitHubTab(dialog);
    await typeRepo(dialog, user);
    await waitFor(() => expect(within(dialog).getByDisplayValue('repo')).toBeInTheDocument());

    await submit(dialog, user);

    expect(await within(dialog).findByText(/Could not read agent\.yaml/)).toBeInTheDocument();
    expect(created).toHaveLength(0);
    await waitFor(() => expect(deleted).toContain(agentSpecFilesetName('repo')));
  });
});

describe('NewAgentModal folder drop', () => {
  const dirEntry = (name: string, fullPath: string, children: FileSystemEntry[]) =>
    ({
      name,
      fullPath,
      isFile: false,
      isDirectory: true,
      createReader: () => {
        let drained = false;
        return {
          readEntries: (resolve: (entries: FileSystemEntry[]) => void) => {
            resolve(drained ? [] : children);
            drained = true;
          },
        };
      },
    }) as unknown as FileSystemEntry;

  const fileEntry = (name: string, fullPath: string, contents: string) =>
    ({
      name,
      fullPath,
      isFile: true,
      isDirectory: false,
      file: (resolve: (file: File) => void) => resolve(new File([contents], name)),
    }) as unknown as FileSystemEntry;

  it('accepts a dropped folder, walking it into nested paths', async () => {
    const user = userEvent.setup();
    const { uploaded, created } = mockPlatform();

    renderModal();
    const dialog = await screen.findByRole('dialog');
    await openUploadTab(dialog);

    const root = dirEntry('calc-agent', '/calc-agent', [
      fileEntry('agent.yaml', '/calc-agent/agent.yaml', FABRIC_YAML),
      dirEntry('mcps', '/calc-agent/mcps', [
        fileEntry('calculator.py', '/calc-agent/mcps/calculator.py', 'print(1)\n'),
      ]),
    ]);

    fireEvent.drop(within(dialog).getByTestId('agent-directory-dropzone'), {
      dataTransfer: { items: [{ kind: 'file', webkitGetAsEntry: () => root }], files: [] },
    });

    await waitFor(() => expect(within(dialog).getByDisplayValue('calc')).toBeInTheDocument());
    await submit(dialog, user);

    await waitFor(() => expect(created).toHaveLength(1));
    expect([...uploaded].sort()).toEqual(['agent.yaml', 'mcps/calculator.py']);
  });
});
