// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateOptimizeJob } from '@nemo/sdk/generated/agents/agents';
import type { OptimizeJob } from '@nemo/sdk/generated/agents/schema/OptimizeJob';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesRetrieveFileset,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import {
  deleteStudioBundleFileset,
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
  filesRetrieveFileset: vi.fn(),
}));

const entryFor = (path: string) => ({ path, file: new File(['x'], path) });

const params = () => ({
  workspace: 'ws',
  agentName: 'hermes',
  entries: [entryFor('optimize.yaml'), entryFor('dataset.json')],
  optimizeConfig: 'optimize.yaml',
});

/** What the files service returns for a bundle Studio staged. */
const stagedBundle = (
  customFields: Record<string, unknown> = { studio_optimize_bundle: 'hermes' }
) => ({ name: 'hermes-optimize-abc', custom_fields: customFields }) as unknown as FilesetOutput;

beforeEach(() => {
  vi.spyOn(Date, 'now').mockReturnValue(1_700_000_000_000);
  vi.mocked(filesCreateFileset).mockResolvedValue({} as never);
  vi.mocked(filesUploadFile).mockResolvedValue({} as never);
  vi.mocked(filesDeleteFileset).mockResolvedValue(undefined as never);
  vi.mocked(filesRetrieveFileset).mockResolvedValue(stagedBundle() as never);
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
      // The stamp is what lets the delete flow tell this bundle from a fileset it must not touch.
      expect.objectContaining({
        name: filesetName,
        custom_fields: { studio_optimize_bundle: 'hermes' },
      })
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

describe('deleteStudioBundleFileset', () => {
  const study = (customFields?: Record<string, unknown>) =>
    ({
      name: 'study-1',
      spec: { optimize_config: 'optimize.yaml', optimize_config_fileset: 'ws/hermes-optimize-abc' },
      custom_fields: customFields,
    }) as unknown as OptimizeJob;

  /** A study pointing at the bundle Studio staged for it. */
  const ownBundle = () => study({ studio_bundle_fileset: 'hermes-optimize-abc' });

  const notFound = () => Object.assign(new Error('gone'), { status: 404 });

  it('deletes the bundle the study staged', async () => {
    await expect(deleteStudioBundleFileset('ws', ownBundle())).resolves.toBeUndefined();

    expect(filesDeleteFileset).toHaveBeenCalledWith('ws', 'hermes-optimize-abc');
  });

  it('ignores a study that claims no bundle', async () => {
    await expect(deleteStudioBundleFileset('ws', study())).resolves.toBeUndefined();

    expect(filesRetrieveFileset).not.toHaveBeenCalled();
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('ignores a marker naming a fileset the study does not run from', async () => {
    const aimedElsewhere = study({ studio_bundle_fileset: 'shared-eval-data' });

    await expect(deleteStudioBundleFileset('ws', aimedElsewhere)).resolves.toBeUndefined();

    expect(filesRetrieveFileset).not.toHaveBeenCalled();
    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('leaves a fileset Studio did not stage alone', async () => {
    vi.mocked(filesRetrieveFileset).mockResolvedValue(stagedBundle({}) as never);

    await expect(deleteStudioBundleFileset('ws', ownBundle())).resolves.toBeUndefined();

    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('counts an already-deleted bundle as cleaned up, so a half-finished delete can be retried', async () => {
    vi.mocked(filesDeleteFileset).mockRejectedValue(notFound());

    await expect(deleteStudioBundleFileset('ws', ownBundle())).resolves.toBeUndefined();
  });

  it('counts a bundle that no longer exists as cleaned up', async () => {
    vi.mocked(filesRetrieveFileset).mockRejectedValue(notFound());

    await expect(deleteStudioBundleFileset('ws', ownBundle())).resolves.toBeUndefined();

    expect(filesDeleteFileset).not.toHaveBeenCalled();
  });

  it('names the bundle it could not delete, so the caller can report it', async () => {
    vi.mocked(filesDeleteFileset).mockRejectedValue(new Error('boom'));

    await expect(deleteStudioBundleFileset('ws', ownBundle())).rejects.toThrow(
      /hermes-optimize-abc/
    );
  });
});
