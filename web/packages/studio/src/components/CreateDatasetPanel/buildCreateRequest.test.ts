// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { buildCreateRequest } from '@studio/components/CreateDatasetPanel/buildCreateRequest';

describe('buildCreateRequest (spec 11.4, 5-8)', () => {
  it('HuggingFace: sends type, inferred fields, default endpoint, token ref; never UI url keys or cache', () => {
    const req = buildCreateRequest({
      tab: 'huggingface',
      name: 'alpaca',
      description: 'Alpaca data',
      fields: { repo_id: 'tatsu-lab/alpaca', repo_type: 'dataset', revision: 'main' },
      secrets: { token: 'hf-token' },
    });
    expect(req).toStrictEqual({
      name: 'alpaca',
      description: 'Alpaca data',
      purpose: 'dataset',
      storage: {
        type: 'huggingface',
        repo_id: 'tatsu-lab/alpaca',
        repo_type: 'dataset',
        revision: 'main',
        endpoint: 'https://huggingface.co',
        token_secret: 'hf-token',
      },
    });
    expect('cache' in req).toBe(false);
  });

  it('HuggingFace: omits empty token, uses advanced endpoint override', () => {
    const req = buildCreateRequest({
      tab: 'huggingface',
      name: 'alpaca',
      description: 'd',
      fields: { repo_id: 'a/b', repo_type: 'model', revision: 'v2' },
      secrets: { token: '' },
      advanced: { endpoint: 'https://hf.internal' },
    });
    expect(req.storage).toStrictEqual({
      type: 'huggingface',
      repo_id: 'a/b',
      repo_type: 'model',
      revision: 'v2',
      endpoint: 'https://hf.internal',
    });
    expect('cache' in req).toBe(false);
  });

  it('NGC: maps api key to api_key_secret, api host to host with default, omits absent version', () => {
    const withVersion = buildCreateRequest({
      tab: 'ngc',
      name: 'personas',
      description: 'd',
      fields: { org: 'nvidia', team: 'nemotron-personas', target: 'x', target_type: 'resource', version: '0.0.2' },
      secrets: { apiKey: 'ngc-key' },
    });
    expect(withVersion).toStrictEqual({
      name: 'personas',
      description: 'd',
      purpose: 'dataset',
      storage: {
        type: 'ngc',
        org: 'nvidia',
        team: 'nemotron-personas',
        target: 'x',
        target_type: 'resource',
        version: '0.0.2',
        api_key_secret: 'ngc-key',
        host: 'https://api.ngc.nvidia.com',
      },
    });

    const noVersion = buildCreateRequest({
      tab: 'ngc',
      name: 'personas',
      description: 'd',
      fields: { org: 'o', team: 't', target: 'x', target_type: 'model' },
      secrets: { apiKey: 'k' },
      advanced: { apiHost: 'https://api.stg.ngc' },
    });
    expect(noVersion.storage).toStrictEqual({
      type: 'ngc',
      org: 'o',
      team: 't',
      target: 'x',
      target_type: 'model',
      api_key_secret: 'k',
      host: 'https://api.stg.ngc',
    });
    expect('cache' in noVersion).toBe(false);
  });

  it('GitHub: sends owner/repo/revision, optional path only when present, default api_base_url, optional token', () => {
    const withPath = buildCreateRequest({
      tab: 'github',
      name: 'repo',
      description: 'd',
      fields: { owner: 'own', repo: 'repo', revision: 'dev', path: 'data/train' },
      secrets: { token: 'gh' },
    });
    expect(withPath.storage).toStrictEqual({
      type: 'github',
      owner: 'own',
      repo: 'repo',
      revision: 'dev',
      path: 'data/train',
      token_secret: 'gh',
      api_base_url: 'https://api.github.com',
    });

    const bare = buildCreateRequest({
      tab: 'github',
      name: 'repo',
      description: 'd',
      fields: { owner: 'own', repo: 'repo', revision: 'HEAD' },
      advanced: { apiBaseUrl: 'https://ghe.example/api/v3' },
    });
    expect(bare.storage).toStrictEqual({
      type: 'github',
      owner: 'own',
      repo: 'repo',
      revision: 'HEAD',
      api_base_url: 'https://ghe.example/api/v3',
    });
    expect('cache' in bare).toBe(false);
  });

  it('S3: always use_sdk_auth false with both secret refs, default signature v4, omits blank region/endpoint/prefix', () => {
    const minimal = buildCreateRequest({
      tab: 's3',
      name: 'bucket-data',
      description: 'd',
      fields: { bucket: 'my-bucket' },
      secrets: { accessKeyId: 'ak', secretAccessKey: 'sk' },
      advanced: { region: '', endpointUrl: '' },
    });
    expect(minimal).toStrictEqual({
      name: 'bucket-data',
      description: 'd',
      purpose: 'dataset',
      storage: {
        type: 's3',
        bucket: 'my-bucket',
        use_sdk_auth: false,
        access_key_id_secret: 'ak',
        secret_access_key_secret: 'sk',
        signature_version: 's3v4',
      },
    });

    const full = buildCreateRequest({
      tab: 's3',
      name: 'bucket-data',
      description: 'd',
      fields: { bucket: 'my-bucket', prefix: 'data/train' },
      secrets: { accessKeyId: 'ak', secretAccessKey: 'sk' },
      advanced: { region: 'us-west-2', endpointUrl: 'https://minio.local', signatureVersion: 's3' },
    });
    expect(full.storage).toStrictEqual({
      type: 's3',
      bucket: 'my-bucket',
      prefix: 'data/train',
      region: 'us-west-2',
      endpoint_url: 'https://minio.local',
      use_sdk_auth: false,
      access_key_id_secret: 'ak',
      secret_access_key_secret: 'sk',
      signature_version: 's3',
    });
    expect('cache' in full).toBe(false);
  });

  it('never leaks UI-only url keys into the request or storage', () => {
    const reqs = [
      buildCreateRequest({
        tab: 'huggingface',
        name: 'ab',
        description: 'd',
        fields: { repo_id: 'a/b', repo_type: 'dataset', revision: 'main' },
      }),
      buildCreateRequest({
        tab: 'github',
        name: 'ab',
        description: 'd',
        fields: { owner: 'a', repo: 'b', revision: 'HEAD' },
      }),
      buildCreateRequest({
        tab: 'ngc',
        name: 'ab',
        description: 'd',
        fields: { org: 'a', team: 'b', target: 'c', target_type: 'resource' },
        secrets: { apiKey: 'k' },
      }),
      buildCreateRequest({
        tab: 's3',
        name: 'ab',
        description: 'd',
        fields: { bucket: 'b' },
        secrets: { accessKeyId: 'a', secretAccessKey: 's' },
      }),
    ];
    for (const req of reqs) {
      for (const key of ['hf_ref', 'ngc_url', 'github_url', 's3_uri']) {
        expect(req).not.toHaveProperty(key);
        expect(req.storage).not.toHaveProperty(key);
      }
    }
  });

  it('Upload: purpose dataset, no storage key (not even undefined), no cache key', () => {
    const req = buildCreateRequest({ tab: 'upload', name: 'my-folder', description: 'files' });
    expect(req).toStrictEqual({ name: 'my-folder', description: 'files', purpose: 'dataset' });
    expect('storage' in req).toBe(false);
    expect('cache' in req).toBe(false);
  });
});
