// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Validation of the active tab's draft (spec 3.2, 4.2, 9, 10, 11.3) and its translation into a
// Create Fileset request. Pure: the caller supplies what only the server or the UI knows.

import type { BuildCreateRequestInput } from '@studio/components/CreateDatasetPanel/buildCreateRequest';
import {
  getInvalidUrlMessage,
  NO_FILES_MESSAGE,
  NO_SECRETS_REQUIRED_MESSAGE,
  type SecretFieldKey,
  SOURCE_TABS,
  type TabId,
} from '@studio/components/CreateDatasetPanel/constants';
import { getDatasetNameError } from '@studio/components/CreateDatasetPanel/datasetName';
import {
  type Draft,
  draftEntries,
  type PanelState,
} from '@studio/components/CreateDatasetPanel/draft';
import {
  type ParseResult,
  parseSource,
  type SourceTab,
} from '@studio/components/CreateDatasetPanel/parseSource';

/** A wrong-source URL is a link to the right tab, so it cannot be a plain message (spec 4.2). */
export type SourceError =
  | { kind: 'message'; message: string }
  | { kind: 'wrong-tab'; actual: SourceTab };

export interface DraftErrors {
  name?: string;
  source?: SourceError;
  secrets: Partial<Record<SecretFieldKey, string>>;
  files?: string;
}

export interface ValidationContext {
  workspace: string;
  /** The name is already in use in the workspace. */
  nameTaken: boolean;
  /** The workspace's secret list loaded and is empty. */
  noSecrets: boolean;
}

function getSourceError(tab: SourceTab, parsed: ParseResult): SourceError | undefined {
  switch (parsed.status) {
    case 'ok':
      return undefined;
    case 'empty':
    case 'incomplete':
      return { kind: 'message', message: SOURCE_TABS[tab].urlError };
    case 'invalid':
      return { kind: 'message', message: getInvalidUrlMessage(tab) };
    case 'wrong-tab':
      return { kind: 'wrong-tab', actual: parsed.actual };
  }
}

export function validateDraft(tab: TabId, draft: Draft, ctx: ValidationContext): DraftErrors {
  const errors: DraftErrors = {
    name: getDatasetNameError(draft.name, { taken: ctx.nameTaken, workspace: ctx.workspace }),
    secrets: {},
  };

  if (tab === 'upload') {
    if (draftEntries(draft).length === 0) errors.files = NO_FILES_MESSAGE;
    return errors;
  }

  errors.source = getSourceError(tab, parseSource(tab, draft.source));
  for (const field of SOURCE_TABS[tab].basic) {
    if (field.kind !== 'secret' || !field.requiredMessage) continue;
    if (!draft.values[field.key]?.trim()) {
      errors.secrets[field.key] = ctx.noSecrets
        ? NO_SECRETS_REQUIRED_MESSAGE
        : field.requiredMessage;
    }
  }
  return errors;
}

export const hasErrors = (errors: DraftErrors): boolean =>
  Boolean(errors.name ?? errors.source ?? errors.files) || Object.keys(errors.secrets).length > 0;

/**
 * The errors to display (spec 11.1): name and URL errors follow their own interaction flags, while
 * secret and file errors appear only after a rejected submit. A rejected submit shows everything.
 */
export function visibleErrors(
  errors: DraftErrors,
  draft: Draft,
  submitErrors: PanelState['submitErrors']
): DraftErrors {
  return {
    name: draft.nameShown || submitErrors ? errors.name : undefined,
    source: draft.sourceShown || submitErrors ? errors.source : undefined,
    secrets: submitErrors ? errors.secrets : {},
    files: submitErrors ? errors.files : undefined,
  };
}

/** The request input for a draft that passed `validateDraft`. */
export function buildRequestInput(tab: TabId, draft: Draft): BuildCreateRequestInput | undefined {
  const common = { name: draft.name, description: draft.description };
  if (tab === 'upload') return { ...common, tab };

  const parsed = parseSource(tab, draft.source);
  if (parsed.status !== 'ok') return undefined;
  const { fields } = parsed;
  const v = draft.values;

  // Each parser yields its own tab's fields; the `in` checks narrow the union to that tab's type.
  switch (tab) {
    case 'huggingface':
      return 'repo_id' in fields
        ? {
            ...common,
            tab,
            fields,
            secrets: { token: v.token },
            advanced: { endpoint: v.endpoint },
          }
        : undefined;
    case 'ngc':
      return 'org' in fields
        ? {
            ...common,
            tab,
            fields,
            secrets: { apiKey: v.apiKey },
            advanced: { apiHost: v.apiHost },
          }
        : undefined;
    case 'github':
      return 'owner' in fields
        ? {
            ...common,
            tab,
            fields,
            secrets: { token: v.token },
            advanced: { apiBaseUrl: v.apiBaseUrl },
          }
        : undefined;
    case 's3':
      return 'bucket' in fields
        ? {
            ...common,
            tab,
            fields,
            secrets: { accessKeyId: v.accessKeyId, secretAccessKey: v.secretAccessKey },
            advanced: {
              region: v.region,
              endpointUrl: v.endpointUrl,
              signatureVersion: v.signatureVersion,
            },
          }
        : undefined;
  }
}
