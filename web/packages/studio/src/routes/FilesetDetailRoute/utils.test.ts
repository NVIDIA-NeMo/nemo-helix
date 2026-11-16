// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  FilesetOutput,
  HuggingfaceStorageConfig,
  LocalStorageConfig,
  NGCStorageConfig,
} from '@nemo/sdk/generated/platform/schema';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import { getModelSource, isRootReadme, parseReadme } from '@studio/routes/FilesetDetailRoute/utils';

const makeFileset = (storage: FilesetOutput['storage']): FilesetOutput => ({
  id: 'fs',
  name: 'my-model',
  workspace: 'ws',
  description: '',
  purpose: FilesetPurpose.model,
  storage,
  metadata: {},
  custom_fields: {},
  project: '',
  created_at: '',
  updated_at: '',
});

describe('isRootReadme', () => {
  it('matches case-insensitive readme.md / readme.markdown at the root', () => {
    expect(isRootReadme({ path: 'README.md' } as never)).toBe(true);
    expect(isRootReadme({ path: 'readme.markdown' } as never)).toBe(true);
  });

  it('rejects nested README files', () => {
    expect(isRootReadme({ path: 'docs/README.md' } as never)).toBe(false);
  });
});

describe('parseReadme', () => {
  it('returns content unchanged when there is no YAML frontmatter', () => {
    expect(parseReadme('# Hello world\nbody')).toEqual({ content: '# Hello world\nbody' });
  });

  it('parses YAML frontmatter and strips it from the markdown body', () => {
    const input = '---\nlicense: MIT\n---\n# Body';
    expect(parseReadme(input)).toEqual({
      content: '# Body',
      metadata: { license: 'MIT' },
    });
  });

  it('falls back to raw content when frontmatter never closes', () => {
    const input = '---\nlicense: MIT\nno-close';
    expect(parseReadme(input)).toEqual({ content: input });
  });
});

describe('getModelSource', () => {
  it('derives source from a Hugging Face repo_id', () => {
    const source = getModelSource(
      makeFileset({
        type: 'huggingface',
        repo_id: 'meta-llama/Llama-2-7b',
      } as HuggingfaceStorageConfig)
    );
    expect(source).toEqual({ path: 'meta-llama/Llama-2-7b', creatorSlug: 'meta' });
  });

  it('derives source from an NGC org/team/target', () => {
    const source = getModelSource(
      makeFileset({
        type: 'ngc',
        org: 'nvidia',
        team: 'nemo',
        target: 'llama-3.1-nemotron-nano-8b',
      } as NGCStorageConfig)
    );
    expect(source).toEqual({
      path: 'nvidia/nemo/llama-3.1-nemotron-nano-8b',
      creatorSlug: 'nvidia',
    });
  });

  it('falls back to workspace/name for local storage', () => {
    const source = getModelSource(
      makeFileset({
        type: 'local',
        path: '/some/path',
      } as LocalStorageConfig)
    );
    expect(source).toEqual({ path: 'ws/my-model', creatorSlug: 'ws' });
  });
});
