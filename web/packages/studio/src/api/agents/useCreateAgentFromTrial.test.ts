// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateAgent, agentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import { AgentSpecFilesetOrphanError } from '@studio/api/agents/agentSpecFileset';
import { createAgentFromTrial } from '@studio/api/agents/useCreateAgentFromTrial';
import { copyFilesetFiles } from '@studio/api/files/uploadFilesetEntries';

vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  agentsCreateAgent: vi.fn(),
  agentsGetAgent: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesRetrieveFileset: vi.fn(),
  filesCreateFileset: vi.fn(),
  filesDeleteFileset: vi.fn(),
}));

vi.mock('@studio/api/files/uploadFilesetEntries', () => ({ copyFilesetFiles: vi.fn() }));

const httpError = (status: number): Error =>
  Object.assign(new Error(`HTTP ${status}`), { response: { status } });

const GITHUB_STORAGE = { type: 'github', owner: 'nvidia', repo: 'agents', revision: 'abc123' };

const mockFilesets = (filesets: Record<string, Record<string, unknown>>) =>
  vi.mocked(filesRetrieveFileset).mockImplementation(async (workspace, name) => {
    const fileset = filesets[`${workspace}/${name}`];
    if (!fileset) throw httpError(404);
    return { workspace, name, ...fileset } as never;
  });

const params = (replaceOrphanedFileset = false) => ({
  workspace: 'ws',
  name: 'hermes-trial-4',
  source: {
    workspace: 'src',
    name: 'hermes',
    description: 'Hermes',
    config_format: 'nemo-agents-spec-v1',
  },
  config: { name: 'hermes-trial-4' },
  replaceOrphanedFileset,
});

beforeEach(() => {
  vi.mocked(agentsGetAgent).mockRejectedValue(httpError(404));
  vi.mocked(agentsCreateAgent).mockResolvedValue({ name: 'hermes-trial-4' } as never);
  vi.mocked(filesCreateFileset).mockResolvedValue({} as never);
  vi.mocked(filesDeleteFileset).mockResolvedValue(undefined as never);
  vi.mocked(copyFilesetFiles).mockResolvedValue();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('createAgentFromTrial', () => {
  it('copies an uploaded spec fileset into a new fileset for the new agent', async () => {
    mockFilesets({ 'src/hermes-ethos': { storage: { type: 'local', path: '/data' } } });

    await createAgentFromTrial(params());

    expect(filesCreateFileset).toHaveBeenCalledWith('ws', {
      name: 'hermes-trial-4-ethos',
      description: 'Agent spec for hermes-trial-4, copied from src/hermes',
      storage: undefined,
    });
    expect(copyFilesetFiles).toHaveBeenCalledWith(
      { workspace: 'src', name: 'hermes-ethos' },
      { workspace: 'ws', name: 'hermes-trial-4-ethos' }
    );
    expect(agentsCreateAgent).toHaveBeenCalledWith('ws', {
      name: 'hermes-trial-4',
      description: 'Hermes',
      config: { name: 'hermes-trial-4' },
      config_format: 'nemo-agents-spec-v1',
    });
  });

  it('points the new fileset at the same GitHub revision', async () => {
    mockFilesets({ 'src/hermes-ethos': { storage: GITHUB_STORAGE } });

    await createAgentFromTrial(params());

    expect(filesCreateFileset).toHaveBeenCalledWith(
      'ws',
      expect.objectContaining({ storage: GITHUB_STORAGE })
    );
    expect(copyFilesetFiles).not.toHaveBeenCalled();
  });

  it('creates only the agent when the source has no spec fileset', async () => {
    mockFilesets({});

    await createAgentFromTrial(params());

    expect(filesCreateFileset).not.toHaveBeenCalled();
    expect(agentsCreateAgent).toHaveBeenCalled();
  });

  it('refuses an orphaned fileset under the new name unless told to replace it', async () => {
    mockFilesets({ 'ws/hermes-trial-4-ethos': { storage: { type: 'local', path: '/old' } } });

    await expect(createAgentFromTrial(params())).rejects.toBeInstanceOf(
      AgentSpecFilesetOrphanError
    );
    expect(agentsCreateAgent).not.toHaveBeenCalled();

    await createAgentFromTrial(params(true));
    expect(filesDeleteFileset).toHaveBeenCalledWith('ws', 'hermes-trial-4-ethos');
    expect(agentsCreateAgent).toHaveBeenCalled();
  });

  it('rolls back the copied fileset when the agent is not created', async () => {
    mockFilesets({ 'src/hermes-ethos': { storage: { type: 'local', path: '/data' } } });
    vi.mocked(agentsCreateAgent).mockRejectedValue(new Error('invalid config'));

    await expect(createAgentFromTrial(params())).rejects.toThrow('invalid config');
    expect(filesDeleteFileset).toHaveBeenCalledWith('ws', 'hermes-trial-4-ethos');
  });
});
