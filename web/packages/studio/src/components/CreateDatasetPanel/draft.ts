// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pure state for the Create Dataset panel: one independent draft per tab (spec 2) and the
// reducer that implements name inference (spec 3.3) and the error-timing flags (spec 11.1).
// Nothing here renders or touches the network.

import type { FilesetEntry } from '@studio/api/files/types';
import {
  type FieldKey,
  TAB_ORDER,
  type TabId,
} from '@studio/components/CreateDatasetPanel/constants';
import { inferDatasetName } from '@studio/components/CreateDatasetPanel/datasetName';
import { parseSource } from '@studio/components/CreateDatasetPanel/parseSource';
import { uploadNameSeed } from '@studio/components/CreateDatasetPanel/uploadNameSeed';

export interface Draft {
  name: string;
  /** Set by the first manual edit; from then on source changes never rewrite the name (spec 3.3). */
  nameTouched: boolean;
  /** Whether the name error is on screen independent of a submit attempt (spec 11.1). */
  nameShown: boolean;
  description: string;
  /** The URL / location text as typed. Unused on the Upload tab. */
  source: string;
  /** Whether the URL error is on screen independent of a submit attempt (spec 11.1). */
  sourceShown: boolean;
  /** Secret selections, region and advanced overrides. A missing key is empty. */
  values: Partial<Record<FieldKey, string>>;
  /** Files picked directly. Each pick replaces the previous one. */
  looseFiles: readonly FilesetEntry[];
  /** Files of the picked folder, with their relative paths. Each pick replaces the previous one. */
  folderFiles: readonly FilesetEntry[];
}

export interface PanelState {
  tab: TabId;
  drafts: Record<TabId, Draft>;
  /** Shared across tabs: set by a rejected submit, cleared by any draft edit (spec 11.1). */
  submitErrors: boolean;
  /** Names the server rejected with a 409, which the name check may not know about yet. */
  takenNames: readonly string[];
}

export type PanelAction =
  | { type: 'tab-selected'; tab: TabId }
  | { type: 'source-changed'; value: string; pasted: boolean }
  | { type: 'source-blurred' }
  | { type: 'name-changed'; value: string }
  | { type: 'name-blurred' }
  | { type: 'description-changed'; value: string }
  | { type: 'value-changed'; key: FieldKey; value: string }
  | { type: 'files-picked'; entries: readonly FilesetEntry[] }
  | { type: 'folder-picked'; entries: readonly FilesetEntry[] }
  | { type: 'file-removed'; path: string }
  | { type: 'folder-removed' }
  | { type: 'submit-rejected' }
  | { type: 'name-taken'; name: string }
  | { type: 'reset' };

const emptyDraft = (): Draft => ({
  name: '',
  nameTouched: false,
  nameShown: false,
  description: '',
  source: '',
  sourceShown: false,
  values: {},
  looseFiles: [],
  folderFiles: [],
});

export const createInitialState = (): PanelState => ({
  tab: TAB_ORDER[0],
  drafts: {
    huggingface: emptyDraft(),
    ngc: emptyDraft(),
    github: emptyDraft(),
    s3: emptyDraft(),
    upload: emptyDraft(),
  },
  submitErrors: false,
  takenNames: [],
});

/** The files an Upload draft would send: loose files, then the folder's. */
export const draftEntries = (draft: Draft): FilesetEntry[] => [
  ...draft.looseFiles,
  ...draft.folderFiles,
];

/** Re-infers an untouched name from the staged files. Emptying the selection keeps the name (spec 3.3). */
function inferFromFiles(draft: Draft): Draft {
  const paths = draftEntries(draft).map((entry) => entry.path);
  if (draft.nameTouched || paths.length === 0) return draft;
  const name = inferDatasetName(uploadNameSeed(paths));
  // A nonempty inferred name shows its error at once; an empty one has nothing to flag yet.
  return { ...draft, name, nameShown: name !== '' };
}

function reduceDraft(tab: TabId, draft: Draft, action: PanelAction): Draft {
  switch (action.type) {
    case 'source-changed': {
      if (tab === 'upload') return draft; // no source URL on Upload
      const parsed = parseSource(tab, action.value);
      const next = { ...draft, source: action.value, sourceShown: action.pasted };
      if (draft.nameTouched) return next;
      // Every edit reparses the whole input, so a stale seed never survives (spec 4.3).
      return {
        ...next,
        name: inferDatasetName(parsed.status === 'ok' ? parsed.nameSeed : ''),
        nameShown: action.pasted,
      };
    }
    case 'source-blurred':
      return { ...draft, sourceShown: true, nameShown: draft.nameShown || !draft.nameTouched };
    case 'name-changed':
      return { ...draft, name: action.value, nameTouched: true, nameShown: false };
    case 'name-blurred':
      return draft.nameTouched ? { ...draft, nameShown: true } : draft;
    case 'description-changed':
      return { ...draft, description: action.value };
    case 'value-changed':
      return { ...draft, values: { ...draft.values, [action.key]: action.value } };
    case 'files-picked':
      return inferFromFiles({ ...draft, looseFiles: action.entries });
    case 'folder-picked':
      return inferFromFiles({ ...draft, folderFiles: action.entries });
    case 'file-removed':
      return inferFromFiles({
        ...draft,
        looseFiles: draft.looseFiles.filter((entry) => entry.path !== action.path),
      });
    case 'folder-removed':
      return inferFromFiles({ ...draft, folderFiles: [] });
    case 'name-taken':
      return draft.name === action.name ? { ...draft, nameShown: true } : draft;
    default:
      return draft;
  }
}

export function panelReducer(state: PanelState, action: PanelAction): PanelState {
  switch (action.type) {
    case 'reset':
      return createInitialState();
    case 'tab-selected':
      return { ...state, tab: action.tab };
    case 'submit-rejected':
      return { ...state, submitErrors: true };
    case 'name-taken':
      return {
        ...state,
        takenNames: [...state.takenNames, action.name],
        drafts: {
          ...state.drafts,
          [state.tab]: reduceDraft(state.tab, state.drafts[state.tab], action),
        },
      };
    default: {
      // Blurs only reveal errors; every other action is a draft edit and clears the submit flag.
      const isEdit = action.type !== 'source-blurred' && action.type !== 'name-blurred';
      return {
        ...state,
        submitErrors: isEdit ? false : state.submitErrors,
        drafts: {
          ...state.drafts,
          [state.tab]: reduceDraft(state.tab, state.drafts[state.tab], action),
        },
      };
    }
  }
}
