// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { modelsCreateDeploymentBodyNameRegExp } from '@nemo/sdk/generated/platform/zod/model-deployments';
import {
  HF_DERIVED_BASE_NAME_MAX_LEN,
  huggingFaceRepoIdToBaseName,
  huggingFaceSourceFilesetName,
} from '@studio/routes/DeploymentsListRoute/huggingFaceDeploymentArtifacts';

describe('huggingFaceRepoIdToBaseName', () => {
  it('joins org and repo with a hyphen and lowercases', () => {
    expect(huggingFaceRepoIdToBaseName('Qwen/Qwen2.5-7B-Instruct')).toBe(
      'qwen-qwen2.5-7b-instruct'
    );
  });

  it('keeps dots so version numbers stay readable', () => {
    expect(huggingFaceRepoIdToBaseName('Qwen/Qwen2.5-7B-Instruct')).toContain('2.5');
  });

  it('handles an org that already contains hyphens', () => {
    expect(huggingFaceRepoIdToBaseName('deepseek-ai/DeepSeek-V4-Flash')).toBe(
      'deepseek-ai-deepseek-v4-flash'
    );
  });

  it('prefixes repos that would otherwise start with a digit', () => {
    expect(huggingFaceRepoIdToBaseName('01-ai/Yi-Large')).toBe('m-01-ai-yi-large');
  });

  it('collapses consecutive hyphens, which the name regex forbids', () => {
    expect(huggingFaceRepoIdToBaseName('foo--bar/baz')).toBe('foo-bar-baz');
  });

  it('replaces characters outside the allowed class', () => {
    expect(huggingFaceRepoIdToBaseName('my org/weird!!name')).toBe('my-org-weird-name');
  });

  it('tolerates a bare repo with no org', () => {
    expect(huggingFaceRepoIdToBaseName('Qwen2.5-7B-Instruct')).toBe('qwen2.5-7b-instruct');
  });

  it('ignores surrounding whitespace and stray slashes', () => {
    expect(huggingFaceRepoIdToBaseName('  /Qwen/Qwen2.5-7B-Instruct/  ')).toBe(
      'qwen-qwen2.5-7b-instruct'
    );
  });

  it('truncates to the fileset-safe budget without leaving a trailing hyphen', () => {
    const derived = huggingFaceRepoIdToBaseName(`org/${'a-'.repeat(60)}`);
    expect(derived).not.toBeNull();
    expect(derived!.length).toBeLessThanOrEqual(HF_DERIVED_BASE_NAME_MAX_LEN);
    expect(derived!.endsWith('-')).toBe(false);
  });

  it('keeps the longest derived name — the fileset — inside the 63 character limit', () => {
    const derived = huggingFaceRepoIdToBaseName(`some-really-long-org/${'x'.repeat(80)}`)!;
    expect(huggingFaceSourceFilesetName(`${derived}-deployment`).length).toBeLessThanOrEqual(63);
  });

  it.each([
    'Qwen/Qwen2.5-7B-Instruct',
    'deepseek-ai/DeepSeek-V4-Flash',
    '01-ai/Yi-Large',
    'foo--bar/baz',
    'my org/weird!!name',
    'mistralai/Mistral-7B-Instruct-v0.3',
  ])('produces a name matching the deployment name regex for %s', (repoId) => {
    const derived = huggingFaceRepoIdToBaseName(repoId);
    expect(derived).not.toBeNull();
    expect(modelsCreateDeploymentBodyNameRegExp.test(derived!)).toBe(true);
  });

  it.each(['', '   ', '/', '---', '!!!'])('returns null for unusable input %s', (repoId) => {
    expect(huggingFaceRepoIdToBaseName(repoId)).toBeNull();
  });
});
