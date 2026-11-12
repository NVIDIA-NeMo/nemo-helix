// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseSource } from '@studio/components/CreateDatasetPanel/parseSource';

const NGC_EXAMPLE =
  'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemotron-personas/resources/nemotron-personas-dataset-en_us/versions/0.0.2';

describe('parseSource precedence and classification (spec 4.1, 4.2)', () => {
  it('empty/whitespace-only input is empty on every tab', () => {
    for (const tab of ['huggingface', 'ngc', 'github', 's3'] as const) {
      expect(parseSource(tab, '').status).toBe('empty');
      expect(parseSource(tab, '   \t').status).toBe('empty');
    }
  });

  it('rejects bare ids, scheme-less URLs, unsupported schemes, userinfo, inner whitespace as invalid (spec 14)', () => {
    const hf = (raw: string) => parseSource('huggingface', raw).status;
    expect(hf('tatsu-lab/alpaca')).toBe('invalid');
    expect(hf('huggingface.co/datasets/tatsu-lab/alpaca')).toBe('invalid');
    expect(hf('ftp://huggingface.co/datasets/tatsu-lab/alpaca')).toBe('invalid');
    expect(hf('https://user:pw@huggingface.co/datasets/tatsu-lab/alpaca')).toBe('invalid');
    expect(hf('https://huggingface.co/datasets/tatsu lab/alpaca')).toBe('invalid');
    expect(hf('https://example.com/datasets/tatsu-lab/alpaca')).toBe('invalid');
    expect(parseSource('github', 'tatsu-lab/stanford_alpaca').status).toBe('invalid');
    expect(parseSource('s3', 'my-bucket').status).toBe('invalid');
    expect(parseSource('ngc', 'ngc.nvidia.com/orgs/a/teams/b/resources/c').status).toBe('invalid');
  });

  it('recognized source on another tab is wrong-tab with the actual tab, even when it would be incomplete there (spec 14)', () => {
    expect(parseSource('huggingface', 'https://github.com/tatsu-lab/stanford_alpaca')).toEqual({
      status: 'wrong-tab',
      actual: 'github',
    });
    expect(parseSource('huggingface', 'https://github.com/only-owner')).toEqual({
      status: 'wrong-tab',
      actual: 'github',
    });
    expect(parseSource('github', 'https://huggingface.co/datasets/tatsu-lab/alpaca')).toEqual({
      status: 'wrong-tab',
      actual: 'huggingface',
    });
    expect(parseSource('github', 'https://hf.co/tatsu-lab/alpaca')).toEqual({
      status: 'wrong-tab',
      actual: 'huggingface',
    });
    expect(parseSource('github', NGC_EXAMPLE)).toEqual({ status: 'wrong-tab', actual: 'ngc' });
    expect(parseSource('ngc', 's3://my-bucket/data')).toEqual({
      status: 'wrong-tab',
      actual: 's3',
    });
    expect(parseSource('huggingface', 's3://my-bucket/data')).toEqual({
      status: 'wrong-tab',
      actual: 's3',
    });
    expect(parseSource('s3', 'https://github.com/a/b')).toEqual({
      status: 'wrong-tab',
      actual: 'github',
    });
    expect(parseSource('s3', 'https://huggingface.co/datasets/a/b')).toEqual({
      status: 'wrong-tab',
      actual: 'huggingface',
    });
  });

  it('trims surrounding whitespace before parsing', () => {
    const r = parseSource('huggingface', '  https://huggingface.co/datasets/tatsu-lab/alpaca \n');
    expect(r.status).toBe('ok');
  });
});

describe('HuggingFace (spec 5, 14, 15)', () => {
  it('parses the spec 15 example', () => {
    expect(parseSource('huggingface', 'https://huggingface.co/datasets/tatsu-lab/alpaca')).toEqual({
      status: 'ok',
      fields: { repo_id: 'tatsu-lab/alpaca', repo_type: 'dataset', revision: 'main' },
      nameSeed: 'alpaca',
    });
  });

  it('infers repo_type from the path and accepts http, www, and hf.co hosts', () => {
    const fields = (raw: string) => {
      const r = parseSource('huggingface', raw);
      return r.status === 'ok' ? r.fields : r;
    };
    expect(fields('https://huggingface.co/spaces/own/app')).toEqual({
      repo_id: 'own/app',
      repo_type: 'space',
      revision: 'main',
    });
    expect(fields('https://huggingface.co/models/own/m')).toEqual({
      repo_id: 'own/m',
      repo_type: 'model',
      revision: 'main',
    });
    expect(fields('https://huggingface.co/own/m')).toEqual({
      repo_id: 'own/m',
      repo_type: 'model',
      revision: 'main',
    });
    expect(fields('http://www.huggingface.co/datasets/own/d')).toEqual({
      repo_id: 'own/d',
      repo_type: 'dataset',
      revision: 'main',
    });
    expect(fields('https://hf.co/datasets/own/d')).toEqual({
      repo_id: 'own/d',
      repo_type: 'dataset',
      revision: 'main',
    });
    expect(fields('https://www.hf.co/datasets/own/d')).toEqual({
      repo_id: 'own/d',
      repo_type: 'dataset',
      revision: 'main',
    });
  });

  it('takes a single segment after /tree/ as revision and falls back to main (spec 14 reset row)', () => {
    expect(parseSource('huggingface', 'https://huggingface.co/datasets/own/d/tree/v2')).toEqual({
      status: 'ok',
      fields: { repo_id: 'own/d', repo_type: 'dataset', revision: 'v2' },
      nameSeed: 'd',
    });
    // root URL of the same repo carries no stale revision
    const root = parseSource('huggingface', 'https://huggingface.co/datasets/own/d');
    expect(root.status === 'ok' && root.fields).toEqual({
      repo_id: 'own/d',
      repo_type: 'dataset',
      revision: 'main',
    });
  });

  it('is incomplete when owner or repo segment is missing', () => {
    expect(parseSource('huggingface', 'https://huggingface.co').status).toBe('incomplete');
    expect(parseSource('huggingface', 'https://huggingface.co/').status).toBe('incomplete');
    expect(parseSource('huggingface', 'https://huggingface.co/tatsu-lab').status).toBe(
      'incomplete'
    );
    expect(parseSource('huggingface', 'https://huggingface.co/datasets/tatsu-lab').status).toBe(
      'incomplete'
    );
  });
});

describe('NGC (spec 6, 15)', () => {
  it('parses the spec 15 example including versions suffix', () => {
    expect(parseSource('ngc', NGC_EXAMPLE)).toEqual({
      status: 'ok',
      fields: {
        org: 'nvidia',
        team: 'nemotron-personas',
        target: 'nemotron-personas-dataset-en_us',
        target_type: 'resource',
        version: '0.0.2',
      },
      nameSeed: 'nemotron-personas-dataset-en_us',
    });
  });

  it('accepts all four path structures, singular /version/, and ngc.nvidia.com hosts; omits version when absent', () => {
    const ok = (raw: string) => {
      const r = parseSource('ngc', raw);
      expect(r.status, raw).toBe('ok');
      return r.status === 'ok' ? r : undefined;
    };
    expect(ok('https://ngc.nvidia.com/orgs/o/teams/t/resources/x')?.fields).toEqual({
      org: 'o',
      team: 't',
      target: 'x',
      target_type: 'resource',
    });
    expect(ok('https://ngc.nvidia.com/orgs/o/teams/t/models/x')?.fields).toEqual({
      org: 'o',
      team: 't',
      target: 'x',
      target_type: 'model',
    });
    expect(ok('https://catalog.ngc.nvidia.com/orgs/o/t/resources/x')?.fields).toEqual({
      org: 'o',
      team: 't',
      target: 'x',
      target_type: 'resource',
    });
    expect(ok('https://catalog.ngc.nvidia.com/orgs/o/t/models/x/version/1.5')?.fields).toEqual({
      org: 'o',
      team: 't',
      target: 'x',
      target_type: 'model',
      version: '1.5',
    });
    expect(ok('https://ngc.nvidia.com/orgs/o/teams/t/resources/x')?.fields).not.toHaveProperty(
      'version'
    );
    expect(
      ok('https://ngc.nvidia.com/orgs/o/teams/t/resources/x/versions/2')?.fields
    ).toMatchObject({ version: '2' });
  });

  it('replacing a versioned URL with an unversioned one drops the version (spec 14)', () => {
    const r = parseSource('ngc', 'https://catalog.ngc.nvidia.com/orgs/o/teams/t/resources/x');
    expect(r.status === 'ok' && 'version' in r.fields).toBe(false);
  });

  it('rejects teamless and malformed paths as incomplete (spec 14, 6)', () => {
    for (const raw of [
      'https://ngc.nvidia.com/orgs/nvidia/resources/example',
      'https://ngc.nvidia.com/orgs/nvidia/models/example',
      'https://ngc.nvidia.com/orgs/nvidia/teams/resources/example',
      'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/resources/example',
      'https://ngc.nvidia.com/orgs/nvidia/teams/t/resources',
      'https://ngc.nvidia.com/orgs/nvidia',
      'https://ngc.nvidia.com/',
    ]) {
      expect(parseSource('ngc', raw).status, raw).toBe('incomplete');
    }
  });
});

describe('GitHub (spec 7, 15)', () => {
  it('parses the spec 15 example with HEAD revision and no path', () => {
    const r = parseSource('github', 'https://github.com/tatsu-lab/stanford_alpaca');
    expect(r).toEqual({
      status: 'ok',
      fields: { owner: 'tatsu-lab', repo: 'stanford_alpaca', revision: 'HEAD' },
      nameSeed: 'stanford_alpaca',
    });
    expect(r.status === 'ok' && 'path' in r.fields).toBe(false);
  });

  it('strips a trailing .git from the repo and the name seed; accepts www and trailing slash', () => {
    expect(parseSource('github', 'https://www.github.com/own/repo.git/')).toEqual({
      status: 'ok',
      fields: { owner: 'own', repo: 'repo', revision: 'HEAD' },
      nameSeed: 'repo',
    });
  });

  it('/tree/{revision}/{path...} supplies revision and directory path', () => {
    expect(parseSource('github', 'https://github.com/own/repo/tree/dev/data/train')).toEqual({
      status: 'ok',
      fields: { owner: 'own', repo: 'repo', revision: 'dev', path: 'data/train' },
      nameSeed: 'repo',
    });
    const noPath = parseSource('github', 'https://github.com/own/repo/tree/v1');
    expect(noPath.status === 'ok' && noPath.fields).toEqual({
      owner: 'own',
      repo: 'repo',
      revision: 'v1',
    });
    expect(noPath.status === 'ok' && 'path' in noPath.fields).toBe(false);
  });

  it('is incomplete without owner and repo', () => {
    expect(parseSource('github', 'https://github.com').status).toBe('incomplete');
    expect(parseSource('github', 'https://github.com/own').status).toBe('incomplete');
  });
});

describe('S3 (spec 8)', () => {
  const ok = (raw: string) => {
    const r = parseSource('s3', raw);
    expect(r.status, raw).toBe('ok');
    return r.status === 'ok' ? r : undefined;
  };

  it('s3:// URIs: bucket only seeds the name from the bucket; prefix seeds from last segment; trailing slash stripped', () => {
    const bucketOnly = ok('s3://my-bucket');
    expect(bucketOnly?.fields).toEqual({ bucket: 'my-bucket' });
    expect(bucketOnly?.fields).not.toHaveProperty('prefix');
    expect(bucketOnly?.nameSeed).toBe('my-bucket');

    expect(ok('s3://my-bucket/')?.fields).toEqual({ bucket: 'my-bucket' });

    const withPrefix = ok('s3://my-bucket/data/train/');
    expect(withPrefix?.fields).toEqual({ bucket: 'my-bucket', prefix: 'data/train' });
    expect(withPrefix?.nameSeed).toBe('train');
  });

  it('virtual-hosted URLs take bucket from host; region is NOT derived', () => {
    const r = ok('https://my-bucket.s3.us-west-2.amazonaws.com/data/train');
    expect(r?.fields).toEqual({ bucket: 'my-bucket', prefix: 'data/train' });
    expect(r?.fields).not.toHaveProperty('region');
    expect(ok('https://my-bucket.s3.amazonaws.com/x/')?.fields).toEqual({
      bucket: 'my-bucket',
      prefix: 'x',
    });
    expect(ok('https://my-bucket.s3.cn-north-1.amazonaws.com.cn/x')?.fields).toEqual({
      bucket: 'my-bucket',
      prefix: 'x',
    });
  });

  it('path-style URLs take bucket from the first path segment', () => {
    expect(ok('https://s3.us-west-2.amazonaws.com/my-bucket/data')?.fields).toEqual({
      bucket: 'my-bucket',
      prefix: 'data',
    });
    expect(ok('https://s3.amazonaws.com/my-bucket')?.fields).toEqual({ bucket: 'my-bucket' });
  });

  it('console URLs take bucket from /buckets/{bucket}; nonempty prefix query wins, empty is omitted', () => {
    const base = 'https://s3.console.aws.amazon.com/s3/buckets/my-bucket';
    expect(ok(`${base}?region=us-east-1&prefix=foo/bar/`)?.fields).toEqual({
      bucket: 'my-bucket',
      prefix: 'foo/bar',
    });
    expect(ok(`${base}?region=us-east-1`)?.fields).toEqual({ bucket: 'my-bucket' });
    expect(ok(`${base}?prefix=`)?.fields).toEqual({ bucket: 'my-bucket' });
    expect(ok(`${base}?region=us-east-1&prefix=foo/bar/`)?.fields).not.toHaveProperty('region');
  });

  it('is incomplete when recognized but no bucket is present', () => {
    expect(parseSource('s3', 'https://s3.us-west-2.amazonaws.com/').status).toBe('incomplete');
    expect(parseSource('s3', 'https://console.aws.amazon.com/s3/home').status).toBe('incomplete');
  });
});

describe('parseSource review regressions', () => {
  const s3Ok = (raw: string) => {
    const r = parseSource('s3', raw);
    expect(r.status, raw).toBe('ok');
    return r.status === 'ok' ? r : undefined;
  };

  it('S3 console: path-derived prefix, query override, and single percent-decoding (spec 8)', () => {
    const fromPath = s3Ok('https://s3.console.aws.amazon.com/s3/buckets/my-bucket/data/train/');
    expect(fromPath?.fields).toEqual({ bucket: 'my-bucket', prefix: 'data/train' });
    expect(fromPath?.nameSeed).toBe('train');

    expect(
      s3Ok('https://s3.console.aws.amazon.com/s3/buckets/my-bucket/data/train/?prefix=other/')
        ?.fields
    ).toEqual({
      bucket: 'my-bucket',
      prefix: 'other',
    });

    expect(s3Ok('https://s3.console.aws.amazon.com/s3/buckets/b?prefix=a%2520b/')?.fields).toEqual({
      bucket: 'b',
      prefix: 'a%20b',
    });
  });

  it('empty userinfo (`@`, `:@`) is still userinfo and invalid (spec 4.1)', () => {
    expect(parseSource('huggingface', 'https://@huggingface.co/datasets/a/b').status).toBe(
      'invalid'
    );
    expect(parseSource('huggingface', 'https://:@huggingface.co/datasets/a/b').status).toBe(
      'invalid'
    );
  });

  it('NGC: `resources` must not be treated as a team, even with models/target after it (spec 6, 4.2)', () => {
    expect(
      parseSource('ngc', 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/resources/models/x')
        .status
    ).toBe('incomplete');
  });

  it('S3: a host merely starting with `s3` is not an AWS S3 endpoint (spec 4.1)', () => {
    expect(parseSource('s3', 'https://bucket.s3evil.amazonaws.com/key').status).toBe('invalid');
    expect(parseSource('s3', 'https://s3evil.amazonaws.com/bucket/key').status).toBe('invalid');
  });
});
