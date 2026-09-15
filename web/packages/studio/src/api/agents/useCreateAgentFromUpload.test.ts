// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateAgent, agentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesRetrieveFileset,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import {
  AgentSpecFilesetConflictError,
  AgentSpecFilesetOrphanError,
  createAgentFromUpload,
} from '@studio/api/agents/useCreateAgentFromUpload';
import type { UploadAgentEntry } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';

vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  agentsCreateAgent: vi.fn(),
  agentsGetAgent: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesRetrieveFileset: vi.fn(),
  filesCreateFileset: vi.fn(),
  filesUploadFile: vi.fn(),
  filesDeleteFileset: vi.fn(),
}));

const FABRIC_YAML = 'config_format: nemo-agents-spec-v1\nname: calc\ndescription: Adds numbers\n';

const entryFor = (path: string, contents: string): UploadAgentEntry => ({
  path,
  file: new File([contents], path.split('/').pop() ?? path),
});

const entries = (): UploadAgentEntry[] => [
  entryFor('agent.yaml', FABRIC_YAML),
  entryFor('mcps/calculator.py', 'print(1)\n'),
];

const params = () => ({ workspace: 'ws', name: 'calc', entries: entries() });

const httpError = (status: number): Error =>
  Object.assign(new Error(`HTTP ${status}`), { response: { status } });

const filesetMissing = () => vi.mocked(filesRetrieveFileset).mockRejectedValue(httpError(404));
const filesetExists = () =>
  vi.mocked(filesRetrieveFileset).mockResolvedValue({ name: 'calc-ethos' } as never);
const agentMissing = () => vi.mocked(agentsGetAgent).mockRejectedValue(httpError(404));
const agentExists = () => vi.mocked(agentsGetAgent).mockResolvedValue({ name: 'calc' } as never);

beforeEach(() => {
  filesetMissing();
  agentMissing();
  vi.mocked(filesCreateFileset).mockResolvedValue({ name: 'calc-ethos' } as never);
  vi.mocked(filesUploadFile).mockResolvedValue({ path: 'agent.yaml' } as never);
  vi.mocked(filesDeleteFileset).mockResolvedValue(undefined as never);
  vi.mocked(agentsCreateAgent).mockResolvedValue({ name: 'calc' } as never);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('createAgentFromUpload', () => {
  it('uploads every file before creating the agent', async () => {
    const order: string[] = [];
    vi.mocked(filesUploadFile).mockImplementation(async (_ws, _fs, path) => {
      order.push(`upload:${path}`);
      return { path } as never;
    });
    vi.mocked(agentsCreateAgent).mockImplementation(async () => {
      order.push('createAgent');
      return { name: 'calc' } as never;
    });

    await createAgentFromUpload(params());

    // Uploads run concurrently, so only their completion before the create is guaranteed.
    expect(order.at(-1)).toBe('createAgent');
    expect(order.slice(0, -1).sort()).toEqual(['upload:agent.yaml', 'upload:mcps/calculator.py']);
    expect(agentsCreateAgent).toHaveBeenCalledWith('ws', {
      name: 'calc',
      description: 'Adds numbers',
      config: expect.objectContaining({ config_format: 'nemo-agents-spec-v1' }),
      config_format: 'nemo-agents-spec-v1',
    });
  });

  it('refuses a fileset that an existing agent owns', async () => {
    filesetExists();
    agentExists();

    await expect(createAgentFromUpload(params())).rejects.toThrow(AgentSpecFilesetConflictError);
    expect(filesCreateFileset).not.toHaveBeenCalled();
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('asks before replacing a fileset that no agent owns', async () => {
    filesetExists();

    await expect(createAgentFromUpload(params())).rejects.toThrow(AgentSpecFilesetOrphanError);
    expect(filesDeleteFileset).not.toHaveBeenCalled();
    expect(agentsCreateAgent).not.toHaveBeenCalled();
  });

  it('replaces an orphaned fileset once confirmed', async () => {
    filesetExists();

    await createAgentFromUpload({ ...params(), replaceOrphanedFileset: true });

    expect(filesDeleteFileset).toHaveBeenCalledWith('ws', 'calc-ethos');
    expect(filesCreateFileset).toHaveBeenCalledWith(
      'ws',
      expect.objectContaining({ name: 'calc-ethos' })
    );
    expect(agentsCreateAgent).toHaveBeenCalled();
  });

  it('does not replace an owned fileset even when confirmed', async () => {
    filesetExists();
    agentExists();

    await expect(
      createAgentFromUpload({ ...params(), replaceOrphanedFileset: true })
    ).rejects.toThrow(AgentSpecFilesetConflictError);
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('deletes the fileset when an upload fails', async () => {
    vi.mocked(filesUploadFile)
      .mockResolvedValueOnce({ path: 'agent.yaml' } as never)
      .mockRejectedValueOnce(new Error('network down'));

    await expect(createAgentFromUpload(params())).rejects.toThrow('network down');
    expect(filesDeleteFileset).toHaveBeenCalledWith('ws', 'calc-ethos');
  });

  it('deletes the fileset when creating the agent fails', async () => {
    vi.mocked(agentsCreateAgent).mockRejectedValue(new Error('409 conflict'));

    await expect(createAgentFromUpload(params())).rejects.toThrow('409 conflict');
    expect(filesDeleteFileset).toHaveBeenCalledWith('ws', 'calc-ethos');
  });

  it('does not delete a fileset it failed to create', async () => {
    vi.mocked(filesCreateFileset).mockRejectedValue(httpError(409));

    await expect(createAgentFromUpload(params())).rejects.toThrow('HTTP 409');
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('does not claim the name when the fileset lookup fails for any reason but absence', async () => {
    vi.mocked(filesRetrieveFileset).mockRejectedValue(httpError(503));

    await expect(createAgentFromUpload(params())).rejects.toThrow('HTTP 503');
    expect(filesCreateFileset).not.toHaveBeenCalled();
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('does not replace a fileset when the agent lookup fails for any reason but absence', async () => {
    filesetExists();
    vi.mocked(agentsGetAgent).mockRejectedValue(httpError(503));

    await expect(
      createAgentFromUpload({ ...params(), replaceOrphanedFileset: true })
    ).rejects.toThrow('HTTP 503');
    expect(filesDeleteFileset).not.toHaveBeenCalled();
    expect(agentsCreateAgent).not.toHaveBeenCalled();
  });

  it('rejects a non-Fabric config before touching anything', async () => {
    const natEntries = [entryFor('agent.yaml', 'config_format: nat-workflow-v1\n')];

    await expect(
      createAgentFromUpload({ workspace: 'ws', name: 'calc', entries: natEntries })
    ).rejects.toThrow(/config_format/);
    expect(filesRetrieveFileset).not.toHaveBeenCalled();
    expect(filesCreateFileset).not.toHaveBeenCalled();
    expect(agentsCreateAgent).not.toHaveBeenCalled();
  });
});
