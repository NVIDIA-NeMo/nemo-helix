// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Builds the Create Fileset request body from a validated draft (spec 11.4). Sends only
// documented storage fields; UI-only URL inputs and `cache` are never included (cache policy is open, spec 12).

import { type CreateFilesetRequest, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import type {
  GithubFields,
  HuggingFaceFields,
  NgcFields,
  S3Fields,
  SourceFields,
  SourceTab,
} from '@studio/components/CreateDatasetPanel/parseSource';

export interface BuildCreateRequestInput {
  tab: SourceTab | 'upload';
  name: string;
  description: string;
  fields?: SourceFields;
  secrets?: { token?: string; apiKey?: string; accessKeyId?: string; secretAccessKey?: string };
  advanced?: {
    endpoint?: string;
    apiHost?: string;
    apiBaseUrl?: string;
    region?: string;
    endpointUrl?: string;
    signatureVersion?: string;
  };
}

/** `{ [key]: value }` for a nonempty value, otherwise `{}` so empty fields are omitted. */
function opt<K extends string>(key: K, value: string | undefined): { [P in K]?: string } {
  return (value ? { [key]: value } : {}) as { [P in K]?: string };
}

export function buildCreateRequest(input: BuildCreateRequestInput): CreateFilesetRequest {
  const { tab, name, description, secrets = {}, advanced = {} } = input;
  const base = { name, description, purpose: FilesetPurpose.dataset };
  if (tab === 'upload') return base; // platform default storage: omit `storage` (spec 10)

  // A blank/whitespace advanced override falls back to the default; with no default it is omitted.
  const adv = (value: string | undefined, fallback?: string) => (value?.trim() ? value : fallback);
  // ponytail: `fields` shape is guaranteed by the caller via parseSource for the same tab.
  const fields = input.fields;

  switch (tab) {
    case 'huggingface': {
      const f = fields as HuggingFaceFields;
      return {
        ...base,
        storage: {
          type: 'huggingface',
          repo_id: f.repo_id,
          repo_type: f.repo_type,
          revision: f.revision,
          endpoint: adv(advanced.endpoint, 'https://huggingface.co'),
          ...opt('token_secret', secrets.token),
        },
      };
    }
    case 'ngc': {
      const f = fields as NgcFields;
      return {
        ...base,
        storage: {
          type: 'ngc',
          org: f.org,
          team: f.team,
          target: f.target,
          target_type: f.target_type,
          ...opt('version', f.version),
          api_key_secret: secrets.apiKey ?? '',
          host: adv(advanced.apiHost, 'https://api.ngc.nvidia.com'),
        },
      };
    }
    case 'github': {
      const f = fields as GithubFields;
      return {
        ...base,
        storage: {
          type: 'github',
          owner: f.owner,
          repo: f.repo,
          revision: f.revision,
          ...opt('path', f.path),
          ...opt('token_secret', secrets.token),
          api_base_url: adv(advanced.apiBaseUrl, 'https://api.github.com'),
        },
      };
    }
    case 's3': {
      const f = fields as S3Fields;
      return {
        ...base,
        storage: {
          type: 's3',
          bucket: f.bucket,
          ...opt('prefix', f.prefix),
          ...opt('region', adv(advanced.region)),
          ...opt('endpoint_url', adv(advanced.endpointUrl)),
          use_sdk_auth: false,
          access_key_id_secret: secrets.accessKeyId ?? '',
          secret_access_key_secret: secrets.secretAccessKey ?? '',
          // Free text from the form; the service validates the s3v4|s3 enum (spec 13).
          signature_version: adv(advanced.signatureVersion, 's3v4') as 's3v4' | 's3',
        },
      };
    }
  }
}
