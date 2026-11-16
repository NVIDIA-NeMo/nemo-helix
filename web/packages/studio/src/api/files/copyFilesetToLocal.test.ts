// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { copyFilesetToLocal } from '@studio/api/files/copyFilesetToLocal';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { http, HttpResponse } from 'msw';

const BASE = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;

const source: FilesetOutput = {
  id: '1',
  name: 'hf-data',
  workspace: 'ws',
  description: 'From HF',
  purpose: 'dataset',
  storage: { type: 'huggingface', repo_id: 'org/data' } as FilesetOutput['storage'],
  metadata: {},
  custom_fields: {},
  project: 'ws',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

const setup = ({ failUpload = false, failDelete = false } = {}) => {
  const created: unknown[] = [];
  const uploaded: string[] = [];
  const deleted: string[] = [];
  server.use(
    http.post(BASE, async ({ request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      created.push(body);
      return HttpResponse.json({ ...source, ...body, storage: { type: 'local', path: '/x' } });
    }),
    http.get(`${BASE}/:name/files`, () =>
      HttpResponse.json({
        data: ['a.jsonl', 'dir/b.jsonl'].map((path) => ({
          path,
          file_ref: path,
          file_url: path,
          size: 1,
        })),
      })
    ),
    http.get(`${BASE}/:name/-/*`, () => HttpResponse.text('row')),
    http.put(`${BASE}/:name/-/*`, ({ params }) => {
      if (failUpload) return HttpResponse.json({ detail: 'boom' }, { status: 500 });
      uploaded.push(String(params[0]));
      return HttpResponse.json({});
    }),
    http.delete(`${BASE}/:name`, ({ params }) => {
      if (failDelete) return HttpResponse.json({ detail: 'nope' }, { status: 500 });
      deleted.push(String(params.name));
      return new HttpResponse(null, { status: 200 });
    })
  );
  return { created, uploaded, deleted };
};

describe('copyFilesetToLocal', () => {
  it('creates a local fileset and copies every file with progress', async () => {
    const { created, uploaded, deleted } = setup();
    const onProgress = vi.fn();

    const result = await copyFilesetToLocal({ source, name: 'hf-data-copy', onProgress });

    expect(result.name).toBe('hf-data-copy');
    expect(created).toEqual([
      expect.objectContaining({ name: 'hf-data-copy', purpose: 'dataset', description: 'From HF' }),
    ]);
    expect(created[0]).not.toHaveProperty('storage');
    expect(uploaded.sort()).toEqual(['a.jsonl', 'dir/b.jsonl']);
    expect(onProgress).toHaveBeenLastCalledWith(2, 2);
    expect(deleted).toEqual([]);
  });

  it('deletes the new fileset when a file fails to copy', async () => {
    const { deleted } = setup({ failUpload: true });

    await expect(copyFilesetToLocal({ source, name: 'hf-data-copy' })).rejects.toBeDefined();
    expect(deleted).toEqual(['hf-data-copy']);
  });

  it('names the leftover fileset when the rollback also fails', async () => {
    setup({ failUpload: true, failDelete: true });

    await expect(copyFilesetToLocal({ source, name: 'hf-data-copy' })).rejects.toThrow(
      /ws\/hf-data-copy could not be removed/
    );
  });
});
