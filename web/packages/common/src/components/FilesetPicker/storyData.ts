// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetFileOutput, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { http, HttpResponse } from 'msw';

type StorageType = 'local' | 's3' | 'huggingface' | 'ngc' | 'github';

const storage = (type: StorageType, name: string): FilesetOutput['storage'] => {
  switch (type) {
    case 'local':
      return { type, path: name };
    case 's3':
      return { type, bucket: name };
    case 'huggingface':
      return { type, repo_id: `nvidia/${name}` };
    case 'ngc':
      return { type, org: 'nvidia', team: 'nemo', target: name, api_key_secret: 'ngc-key' };
    case 'github':
      return { type, owner: 'nvidia', repo: name };
  }
};

const fileset = (
  name: string,
  purpose: FilesetOutput['purpose'],
  storageType: StorageType
): FilesetOutput => ({
  id: `default/${name}`,
  name,
  workspace: 'default',
  description: '',
  purpose,
  storage: storage(storageType, name),
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2026-11-01T00:00:00Z',
  updated_at: '2026-11-01T00:00:00Z',
});

export const STORY_FILESETS: FilesetOutput[] = [
  fileset('sft-chat-v3', 'dataset', 'local'),
  fileset('sft-chat-v2', 'dataset', 'local'),
  fileset('support-transcripts-raw', 'dataset', 's3'),
  fileset('nemotron-personas', 'dataset', 'huggingface'),
  fileset('ngc-eval-suite', 'dataset', 'ngc'),
  fileset('grpo-reward-env', 'environment', 'local'),
  fileset('gym-env-package', 'generic', 'github'),
];

const file = (path: string, size: number): FilesetFileOutput => ({
  path,
  size,
  file_ref: path,
  file_url: path,
});

export const STORY_FILES: Record<string, FilesetFileOutput[]> = {
  'sft-chat-v3': [
    file('eval/questions.csv', 48_213),
    file('eval/questions.parquet', 21_904),
    file('train/chat-00001.jsonl', 4_812_334),
    file('README.md', 1_204),
  ],
  'sft-chat-v2': [file('train.jsonl', 2_381_002), file('validation.jsonl', 240_118)],
  'support-transcripts-raw': [],
};

export const listFilesetsHandler = (filesets: FilesetOutput[]) =>
  http.get('*/apis/files/v2/workspaces/:workspace/filesets', ({ request }) => {
    const params = new URL(request.url).searchParams;
    const exactName = params.get('filter[name]');
    const term = (params.get('filter[name][$like]') ?? '').replaceAll('%', '').toLowerCase();
    const purposes = [
      ...params.getAll('filter[purpose]'),
      ...params.getAll('filter[purpose][$in]'),
    ];
    const data = filesets.filter(
      (f) =>
        (exactName === null ? f.name.toLowerCase().includes(term) : f.name === exactName) &&
        (purposes.length === 0 || purposes.includes(f.purpose))
    );
    return HttpResponse.json({
      data,
      pagination: {
        page: 1,
        page_size: 20,
        current_page_size: data.length,
        total_pages: 1,
        total_results: data.length,
      },
    });
  });

export const listFilesetFilesHandler = (filesByFileset: Record<string, FilesetFileOutput[]>) =>
  http.get('*/apis/files/v2/workspaces/:workspace/filesets/:name/files', ({ params }) =>
    HttpResponse.json({ data: filesByFileset[String(params.name)] ?? [] })
  );
