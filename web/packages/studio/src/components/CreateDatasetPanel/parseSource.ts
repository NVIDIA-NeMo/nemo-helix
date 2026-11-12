// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Source: Linear ASTD-896 'Create Dataset — Form Validation and Behavior' handoff; section numbers in
// comments refer to it.
//
// Pure parser for the Create Dataset source URL input (spec 4-8). Classifies the input by host
// (so a URL pasted into the wrong tab is detected), then extracts the tab's inferred fields.
// Parsing never checks that the remote source exists.

export type SourceTab = 'huggingface' | 'ngc' | 'github' | 's3';

export interface HuggingFaceFields {
  repo_id: string;
  repo_type: 'dataset' | 'model' | 'space';
  revision: string;
}
export interface NgcFields {
  org: string;
  team: string;
  target: string;
  target_type: 'resource' | 'model';
  version?: string;
}
export interface GithubFields {
  owner: string;
  repo: string;
  revision: string;
  path?: string;
}
export interface S3Fields {
  bucket: string;
  prefix?: string;
}
export type SourceFields = HuggingFaceFields | NgcFields | GithubFields | S3Fields;

export type ParseResult =
  | { status: 'empty' }
  | { status: 'invalid' }
  | { status: 'wrong-tab'; actual: SourceTab }
  | { status: 'incomplete' }
  | { status: 'ok'; fields: SourceFields; nameSeed: string };

const HF_HOSTS = new Set(['huggingface.co', 'www.huggingface.co', 'hf.co', 'www.hf.co']);
const GITHUB_HOSTS = new Set(['github.com', 'www.github.com']);
// AWS S3 service label: s3, s3-<region>, s3.<region>, s3.dualstack.<region>; optionally .com.cn (spec 4.1).
const S3_SERVICE = String.raw`s3(?:[.-](?:dualstack\.)?[a-z0-9-]+)?\.amazonaws\.com(?:\.cn)?$`;
const S3_ENDPOINT = new RegExp(String.raw`(?:^|\.)${S3_SERVICE}`);
const S3_VIRTUAL_HOSTED = new RegExp(String.raw`^(.+)\.${S3_SERVICE}`);
const S3_CONSOLE_HOST = /(^|\.)console\.aws\.amazon\.com$/;

// Matches an authority that contains `@`, including the empty userinfo forms `@host` and `:@host`.
const HAS_USERINFO = /^[a-z][a-z0-9+.-]*:\/\/[^/?#]*@/i;

const INVALID: ParseResult = { status: 'invalid' };
const INCOMPLETE: ParseResult = { status: 'incomplete' };

function decode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

const segmentsOf = (path: string) => path.split('/').filter(Boolean).map(decode);

/** Which tab's source this URL belongs to (spec 4.1 recognized hosts), if any. */
function classify(url: URL): SourceTab | undefined {
  const host = url.hostname.toLowerCase();
  if (url.protocol === 's3:') return host ? 's3' : undefined;
  if (HF_HOSTS.has(host)) return 'huggingface';
  if (GITHUB_HOSTS.has(host)) return 'github';
  if (host === 'ngc.nvidia.com' || host.endsWith('.ngc.nvidia.com')) return 'ngc';
  if (S3_ENDPOINT.test(host)) return 's3';
  if (S3_CONSOLE_HOST.test(host) && (url.pathname === '/s3' || url.pathname.startsWith('/s3/')))
    return 's3';
  return undefined;
}

function parseHuggingFace(url: URL): ParseResult {
  let segs = segmentsOf(url.pathname);
  let repo_type: HuggingFaceFields['repo_type'] = 'model';
  if (segs[0] === 'datasets') repo_type = 'dataset';
  else if (segs[0] === 'spaces') repo_type = 'space';
  if (['datasets', 'spaces', 'models'].includes(segs[0])) segs = segs.slice(1);

  const [owner, repo] = segs;
  if (!owner || !repo) return INCOMPLETE;
  // Single segment only; branch names containing slashes are out of scope (spec 13).
  const revision = segs[2] === 'tree' && segs[3] ? segs[3] : 'main';
  return {
    status: 'ok',
    fields: { repo_id: `${owner}/${repo}`, repo_type, revision },
    nameSeed: repo,
  };
}

function parseNgc(url: URL): ParseResult {
  const segs = segmentsOf(url.pathname);
  if (segs[0] !== 'orgs') return INCOMPLETE;
  const org = segs[1];
  // `/orgs/{org}/teams/{team}/...` or `/orgs/{org}/{team}/...`
  const hasTeamsLiteral = segs[2] === 'teams';
  const team = segs[hasTeamsLiteral ? 3 : 2];
  const kindIndex = hasTeamsLiteral ? 4 : 3;
  const kind = segs[kindIndex];
  const target = segs[kindIndex + 1];
  // `resources`/`models` are never teams: this rejects /orgs/o/teams/resources/x and /orgs/o/teams/resources/models/x.
  if (!org || !team || !target || team === 'resources' || team === 'models') return INCOMPLETE;
  if (kind !== 'resources' && kind !== 'models') return INCOMPLETE;

  const version = ['version', 'versions'].includes(segs[kindIndex + 2])
    ? segs[kindIndex + 3]
    : undefined;
  const fields: NgcFields = {
    org,
    team,
    target,
    target_type: kind === 'resources' ? 'resource' : 'model',
  };
  if (version) fields.version = version;
  return { status: 'ok', fields, nameSeed: target };
}

function parseGithub(url: URL): ParseResult {
  const segs = segmentsOf(url.pathname);
  const owner = segs[0];
  const repo = segs[1]?.replace(/\.git$/, '');
  if (!owner || !repo) return INCOMPLETE;

  const fields: GithubFields = { owner, repo, revision: 'HEAD' };
  if (segs[2] === 'tree' && segs[3]) {
    fields.revision = segs[3];
    const path = segs.slice(4).join('/');
    if (path) fields.path = path;
  }
  return { status: 'ok', fields, nameSeed: repo };
}

function parseS3(url: URL): ParseResult {
  const host = url.hostname.toLowerCase();
  const segs = segmentsOf(url.pathname);
  let bucket: string | undefined;
  let prefixSegs: string[];

  if (url.protocol === 's3:') {
    bucket = decode(url.hostname);
    prefixSegs = segs;
  } else if (S3_CONSOLE_HOST.test(host)) {
    // Console URLs: bucket from /buckets/{bucket}; the path after it is the prefix, overridden by a nonempty
    // `prefix` query param (spec 8). searchParams is already decoded once, so it is not decoded again.
    const bucketIndex = segs.indexOf('buckets');
    bucket = bucketIndex === -1 ? undefined : segs[bucketIndex + 1];
    const queryPrefix = (url.searchParams.get('prefix') ?? '').split('/').filter(Boolean);
    prefixSegs = queryPrefix.length ? queryPrefix : segs.slice(bucketIndex + 2);
  } else {
    const virtualHosted = S3_VIRTUAL_HOSTED.exec(host);
    bucket = virtualHosted ? virtualHosted[1] : segs[0];
    prefixSegs = virtualHosted ? segs : segs.slice(1);
  }
  if (!bucket) return INCOMPLETE;

  const fields: S3Fields = { bucket };
  if (prefixSegs.length) fields.prefix = prefixSegs.join('/');
  return { status: 'ok', fields, nameSeed: prefixSegs.at(-1) ?? bucket };
}

const PARSERS: Record<SourceTab, (url: URL) => ParseResult> = {
  huggingface: parseHuggingFace,
  ngc: parseNgc,
  github: parseGithub,
  s3: parseS3,
};

/**
 * Parses `raw` for `tab`. Precedence (spec 4.2): empty, invalid, wrong-tab, then incomplete/ok.
 */
export function parseSource(tab: SourceTab, raw: string): ParseResult {
  const input = raw.trim();
  if (!input) return { status: 'empty' };
  if (/\s/.test(input)) return INVALID;

  let url: URL;
  try {
    url = new URL(input);
  } catch {
    return INVALID; // bare ids and scheme-less strings have no valid scheme
  }
  if (!['http:', 'https:', 's3:'].includes(url.protocol) || HAS_USERINFO.test(input))
    return INVALID;

  const actual = classify(url);
  if (!actual) return INVALID;
  if (actual !== tab) return { status: 'wrong-tab', actual };
  return PARSERS[tab](url);
}
