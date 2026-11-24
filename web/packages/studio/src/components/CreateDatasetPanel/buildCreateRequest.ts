// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Builds the Create Fileset request body from a validated draft (spec 11.4). Sends only
// documented storage fields; UI-only URL inputs and `cache` are never included (cache policy is open, spec 12).

import {
  type CreateFilesetRequest,
  FilesetPurpose,
  type S3StorageConfigSignatureVersion,
} from '@nemo/sdk/generated/platform/schema';
import type {
  GithubFields,
  HuggingFaceFields,
  NgcFields,
  S3Fields,
} from '@studio/components/CreateDatasetPanel/parseSource';
import { HF_DEFAULT_ENDPOINT, NGC_DEFAULT_HOST } from '@studio/util/storageConfigFromUrl';

const GITHUB_DEFAULT_API_BASE_URL = 'https://api.github.com';

interface Common {
  name: string;
  description: string;
}

// Discriminated on `tab`: each external tab carries only its own fields, secrets and advanced overrides.
export type BuildCreateRequestInput =
  | (Common & { tab: 'upload' })
  | (Common & {
      tab: 'huggingface';
      fields: HuggingFaceFields;
      secrets?: { token?: string };
      advanced?: { endpoint?: string };
    })
  | (Common & {
      tab: 'ngc';
      fields: NgcFields;
      secrets?: { apiKey?: string };
      advanced?: { apiHost?: string };
    })
  | (Common & {
      tab: 'github';
      fields: GithubFields;
      secrets?: { token?: string };
      advanced?: { apiBaseUrl?: string };
    })
  | (Common & {
      tab: 's3';
      fields: S3Fields;
      secrets?: { accessKeyId?: string; secretAccessKey?: string };
      advanced?: { region?: string; endpointUrl?: string; signatureVersion?: string };
    });

/** `{ [key]: value }` for a nonempty value, otherwise `{}` so empty or missing fields are omitted. */
function opt<K extends string>(key: K, value: string | undefined): { [P in K]?: string } {
  return (value ? { [key]: value } : {}) as { [P in K]?: string };
}

// A blank/whitespace advanced override falls back to the default; with no default it is omitted.
const adv = (value: string | undefined, fallback?: string) => (value?.trim() ? value : fallback);

export function buildCreateRequest(input: BuildCreateRequestInput): CreateFilesetRequest {
  const base = {
    name: input.name,
    description: input.description,
    purpose: FilesetPurpose.dataset,
  };

  switch (input.tab) {
    case 'upload':
      return base; // platform default storage: omit `storage` (spec 10)
    case 'huggingface': {
      const { fields: f, secrets, advanced } = input;
      return {
        ...base,
        storage: {
          type: 'huggingface',
          repo_id: f.repo_id,
          repo_type: f.repo_type,
          revision: f.revision,
          endpoint: adv(advanced?.endpoint, HF_DEFAULT_ENDPOINT),
          ...opt('token_secret', secrets?.token),
        },
      };
    }
    case 'ngc': {
      const { fields: f, secrets, advanced } = input;
      return {
        ...base,
        storage: {
          type: 'ngc',
          org: f.org,
          team: f.team,
          target: f.target,
          target_type: f.target_type,
          ...opt('version', f.version),
          // Required by the schema but absent when unselected; the panel blocks submit on a missing secret (spec 9).
          ...(opt('api_key_secret', secrets?.apiKey) as { api_key_secret: string }),
          host: adv(advanced?.apiHost, NGC_DEFAULT_HOST),
        },
      };
    }
    case 'github': {
      const { fields: f, secrets, advanced } = input;
      return {
        ...base,
        storage: {
          type: 'github',
          owner: f.owner,
          repo: f.repo,
          revision: f.revision,
          ...opt('path', f.path),
          ...opt('token_secret', secrets?.token),
          api_base_url: adv(advanced?.apiBaseUrl, GITHUB_DEFAULT_API_BASE_URL),
        },
      };
    }
    case 's3': {
      const { fields: f, secrets, advanced } = input;
      return {
        ...base,
        storage: {
          type: 's3',
          bucket: f.bucket,
          ...opt('prefix', f.prefix),
          ...opt('region', adv(advanced?.region)),
          ...opt('endpoint_url', adv(advanced?.endpointUrl)),
          use_sdk_auth: false,
          ...opt('access_key_id_secret', secrets?.accessKeyId),
          ...opt('secret_access_key_secret', secrets?.secretAccessKey),
          // Free text from the form, cast to the SDK enum type without checking; the service validates it (spec 13).
          signature_version: adv(
            advanced?.signatureVersion,
            's3v4'
          ) as S3StorageConfigSignatureVersion,
        },
      };
    }
  }
}
