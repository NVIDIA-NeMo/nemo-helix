// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tab definitions and user-facing copy for the Create Dataset panel. Section numbers refer to the
// Linear ASTD-896 'Create Dataset — Form Validation and Behavior' handoff.

import type { SourceTab } from '@studio/components/CreateDatasetPanel/parseSource';

export type TabId = SourceTab | 'upload';

/** Tab order (spec 2). The first entry is selected on open and after a reset. */
export const TAB_ORDER: readonly TabId[] = ['huggingface', 'ngc', 'github', 's3', 'upload'];

export const TAB_LABELS: Record<TabId, string> = {
  huggingface: 'HuggingFace',
  ngc: 'NGC',
  github: 'GitHub',
  s3: 'S3',
  upload: 'Upload',
};

/** Keys of the per-tab values that are not the URL: secret selections, region and advanced overrides. */
export type FieldKey =
  | 'token'
  | 'apiKey'
  | 'accessKeyId'
  | 'secretAccessKey'
  | 'region'
  | 'endpoint'
  | 'apiHost'
  | 'apiBaseUrl'
  | 'endpointUrl'
  | 'signatureVersion';

export type SecretFieldKey = 'token' | 'apiKey' | 'accessKeyId' | 'secretAccessKey';

export type BasicField =
  | {
      kind: 'secret';
      key: SecretFieldKey;
      label: string;
      /** Inline error when left empty at submit; a secret without one is optional (spec 9). */
      requiredMessage?: string;
    }
  | { kind: 'text'; key: FieldKey; label: string; placeholder?: string };

export interface AdvancedField {
  key: FieldKey;
  label: string;
  /** Shown as a placeholder: a blank value falls back to this default when the request is built. */
  placeholder: string;
}

export interface SourceTabConfig {
  urlLabel: string;
  urlPlaceholder: string;
  /** Shown for an empty input, or a recognized URL missing required parts (spec 4.2). */
  urlError: string;
  /** Source-specific fields shown between Name and Description. */
  basic: readonly BasicField[];
  /** Collapsed by default; their values still go into the request (spec 2). */
  advanced: readonly AdvancedField[];
}

export const SOURCE_TABS: Record<SourceTab, SourceTabConfig> = {
  huggingface: {
    urlLabel: 'HuggingFace URL',
    urlPlaceholder: 'https://huggingface.co/datasets/tatsu-lab/alpaca',
    urlError: 'Enter a full HuggingFace repository URL',
    basic: [{ kind: 'secret', key: 'token', label: 'Token secret' }],
    advanced: [{ key: 'endpoint', label: 'Endpoint', placeholder: 'https://huggingface.co' }],
  },
  ngc: {
    urlLabel: 'NGC Catalog URL',
    urlPlaceholder:
      'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemotron-personas/resources/nemotron-personas-dataset-en_us',
    urlError: 'Enter an NGC catalog URL with an organization, team, and asset',
    basic: [
      {
        kind: 'secret',
        key: 'apiKey',
        label: 'API key secret',
        requiredMessage: 'Select api key secret to continue.',
      },
    ],
    advanced: [{ key: 'apiHost', label: 'API host', placeholder: 'https://api.ngc.nvidia.com' }],
  },
  github: {
    urlLabel: 'GitHub URL',
    urlPlaceholder: 'https://github.com/tatsu-lab/stanford_alpaca',
    urlError: 'Enter a full GitHub repository URL',
    basic: [{ kind: 'secret', key: 'token', label: 'Token secret' }],
    advanced: [{ key: 'apiBaseUrl', label: 'API base URL', placeholder: 'https://api.github.com' }],
  },
  s3: {
    urlLabel: 'Location',
    urlPlaceholder: 's3://my-bucket/path/to/data',
    urlError: 'Enter an s3:// URI or an S3 console URL',
    basic: [
      { kind: 'text', key: 'region', label: 'Region' },
      {
        kind: 'secret',
        key: 'accessKeyId',
        label: 'Access key ID secret',
        requiredMessage: 'Select access key id secret to continue.',
      },
      {
        kind: 'secret',
        key: 'secretAccessKey',
        label: 'Secret access key secret',
        requiredMessage: 'Select secret access key secret to continue.',
      },
    ],
    advanced: [
      { key: 'endpointUrl', label: 'Endpoint URL', placeholder: 'https://s3.example.com' },
      { key: 'signatureVersion', label: 'Signature version', placeholder: 's3v4' },
    ],
  },
};

/** Replaces a required secret's own error when the workspace has no secrets to choose from (spec 9). */
export const NO_SECRETS_REQUIRED_MESSAGE =
  'Create a secret in this workspace, then select it to continue.';
export const NO_SECRETS_PLACEHOLDER = 'No secrets available';
export const NO_SECRETS_HELP =
  'No secrets in this workspace. Create one with `nemo secrets create <name>`, then return here.';
/** Shown instead of the empty-workspace copy when the lookup failed: that says nothing about the secrets. */
export const SECRETS_LOAD_ERROR_MESSAGE = "Couldn't load secrets.";

export const NO_FILES_MESSAGE = 'Choose at least one file or a folder to upload.';
export const NAME_HELP = 'Lowercase letters, digits, hyphens, and dots.';
export const getInvalidUrlMessage = (tab: SourceTab) =>
  `Invalid URL source. Please enter a valid ${TAB_LABELS[tab]} URL`;
