// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateOptimizeJob } from '@nemo/sdk/generated/agents/agents';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import {
  launchOptimizeStudy,
  optimizeBundleFilesetName,
} from '@studio/api/agents/useLaunchOptimizeStudy';

vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  agentsCreateOptimizeJob: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesCreateFileset: vi.fn(),
  filesUploadFile: vi.fn(),
  filesDeleteFileset: vi.fn(),
}));

const entryFor = (path: string) => ({ path, file: new File(['x'], path) });

const params = () => ({
  workspace: 'ws',
  agentName: 'hermes',
  entries: [entryFor('optimize.yaml'), entryFor('dataset.json')],
  optimizeConfig: 'optimize.yaml',
});

beforeEach(() => {
  vi.spyOn(Date, 'now').mockReturnValue(1_700_000_000_000);
  vi.mocked(filesCreateFileset).mockResolvedValue({} as never);
  vi.mocked(filesUploadFile).mockResolvedValue({} as never);
  vi.mocked(filesDeleteFileset).mockResolvedValue(undefined as never);
  vi.mocked(agentsCreateOptimizeJob).mockResolvedValue({ name: 'study-1' } as never);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe('launchOptimizeStudy', () => {
  it('stages the bundle, then submits a study scoped to the agent', async () => {
    const filesetName = optimizeBundleFilesetName('hermes', 1_700_000_000_000);

    await expect(launchOptimizeStudy(params())).resolves.toEqual({ name: 'study-1' });

    expect(filesCreateFileset).toHaveBeenCalledWith(
      'ws',
      expect.objectContaining({ name: filesetName })
    );
    expect(
      vi
        .mocked(filesUploadFile)
        .mock.calls.map((call) => call[2])
        .sort()
    ).toEqual(['dataset.json', 'optimize.yaml']);
    expect(agentsCreateOptimizeJob).toHaveBeenCalledWith('ws', {
      spec: {
        optimize_config: 'optimize.yaml',
        optimize_config_fileset: `ws/${filesetName}`,
        agent: 'hermes',
        workspace: 'ws',
      },
      custom_fields: { studio_bundle_fileset: filesetName },
    });
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('removes the staged fileset when the submit fails', async () => {
    vi.mocked(agentsCreateOptimizeJob).mockRejectedValue(new Error('boom'));

    await expect(launchOptimizeStudy(params())).rejects.toThrow('boom');

    expect(filesDeleteFileset).toHaveBeenCalledWith(
      'ws',
      optimizeBundleFilesetName('hermes', 1_700_000_000_000)
    );
  });

  it('leaves nothing to roll back when the fileset cannot be created', async () => {
    vi.mocked(filesCreateFileset).mockRejectedValue(new Error('conflict'));

    await expect(launchOptimizeStudy(params())).rejects.toThrow('conflict');

    expect(filesUploadFile).not.toHaveBeenCalled();
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });
});
