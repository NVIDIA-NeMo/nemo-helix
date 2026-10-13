// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { filesListFilesetFiles, filesRetrieveFileset } from '@nemo/sdk/generated/platform/files';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';

/**
 * Storage backends whose `upload` is actually implemented server-side (see
 * `services/core/files/src/nhx/core/files/app/backends/`). `huggingface`, `github` and
 * `ngc` all raise `NotImplementedError` on write.
 *
 * Both entries are legitimate homes for a template's converted rows: a fileset created
 * without an explicit `storage` lands on the deployment's `default_storage_config`, which
 * is S3 on a hosted install and local on a laptop. Treating everything but `local` as
 * read-only is what made the template reject the fileset its own previous run created.
 */
export const WRITABLE_STORAGE_TYPES: ReadonlySet<string> = new Set(['local', 's3']);

/** Paths setup uploads, and so the paths a reusable fileset must already hold. */
export const TEMPLATE_DATASET_FILES = ['training.jsonl', 'validation.jsonl'] as const;

/** Keys every template's `convertRow` emits. A fileset holding anything else is not ours. */
const REQUIRED_ROW_KEYS = ['prompt', 'completion'] as const;

export type TemplateConflictReason =
  | 'unwritable'
  | 'purpose'
  | 'incomplete'
  | 'schema'
  | 'source-repo';

/** Which of the template's two filesets collided, and so which name an override renames. */
export type TemplateFilesetTarget = 'source' | 'converted';

export interface TemplateFilesetConflict {
  workspace: string;
  name: string;
  /** `workspace/name`, as the rest of Studio refers to filesets. */
  filesetRef: string;
  target: TemplateFilesetTarget;
  reason: TemplateConflictReason;
  /**
   * Why this fileset cannot be reused, phrased to read after the fileset's own name. The
   * name is rendered as a link beside it rather than quoted in here, so the user can open
   * the thing the message is about.
   */
  message: string;
}

export type FilesetPreflight =
  /** Nothing holds the name; setup creates it. */
  | { kind: 'absent' }
  /** A compatible fileset is already in place; setup reuses it. */
  | { kind: 'reusable'; filesetRef: string }
  /** The name is taken by something setup must not write into or overwrite silently. */
  | { kind: 'conflict'; conflict: TemplateFilesetConflict };

const makeConflict =
  (workspace: string, name: string, target: TemplateFilesetTarget) =>
  (reason: TemplateConflictReason, message: string): FilesetPreflight => ({
    kind: 'conflict',
    conflict: { workspace, name, filesetRef: `${workspace}/${name}`, target, reason, message },
  });

/** Retrieves a fileset, or null when nothing holds the name. */
const retrieveIfPresent = async (workspace: string, name: string) => {
  try {
    return await filesRetrieveFileset(workspace, name);
  } catch (e) {
    if (isNotFoundError(e)) return null;
    throw e;
  }
};

/** Reads the first line of a JSONL head, or null when it cannot be read or parsed. */
const firstRow = (head: ArrayBuffer | null): Record<string, unknown> | null => {
  if (!head) return null;
  const text = new TextDecoder().decode(head);
  const newline = text.indexOf('\n');
  // A head that is all one line may have been cut mid-row, so it is not worth parsing.
  const line = newline === -1 ? null : text.slice(0, newline).trim();
  if (!line) return null;
  try {
    const parsed: unknown = JSON.parse(line);
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
};

/**
 * Checks the external fileset pointing at the recipe's HuggingFace repo.
 *
 * A 409 on create says only that the name is taken, not that it is taken by our repo.
 * Unlike the model filesets, a foreign dataset fileset fails silently: nothing matches
 * the file pattern and the user gets an error that never mentions ownership. It is
 * deliberately not repointed or deleted; it may be the user's own.
 */
export const preflightSourceFileset = async (
  workspace: string,
  name: string,
  hfRepoId: string
): Promise<FilesetPreflight> => {
  const existing = await retrieveIfPresent(workspace, name);
  if (!existing) return { kind: 'absent' };

  const conflict = makeConflict(workspace, name, 'source');
  const { storage } = existing;
  if (storage.type && storage.type !== 'huggingface') {
    return conflict('source-repo', `already exists on ${storage.type} storage, not "${hfRepoId}".`);
  }
  if (storage.type === 'huggingface' && storage.repo_id !== hfRepoId) {
    return conflict(
      'source-repo',
      `already exists and points at "${storage.repo_id}", not "${hfRepoId}".`
    );
  }
  return { kind: 'reusable', filesetRef: `${workspace}/${name}` };
};

export interface ConvertedPreflightArgs {
  workspace: string;
  /** Name of the converted-rows fileset the template wants to write. */
  name: string;
  /** Fetches the first bytes of a file in the fileset; resolves null on any read failure. */
  readHead: (path: string) => Promise<ArrayBuffer | null>;
}

/**
 * Decides, before anything is downloaded, whether the template can write its converted
 * rows to `workspace/name` — and whether it needs to at all.
 *
 * Everything here is cheap: a fileset read, a file listing, and at most one ranged read of
 * the first few KB. The dataset download it guards is tens of megabytes, so a user who
 * cannot succeed finds out in a second rather than after the transfer.
 */
export const preflightConvertedFileset = async ({
  workspace,
  name,
  readHead,
}: ConvertedPreflightArgs): Promise<FilesetPreflight> => {
  const existing = await retrieveIfPresent(workspace, name);
  if (!existing) return { kind: 'absent' };

  const conflict = makeConflict(workspace, name, 'converted');

  // The discriminator is optional in the generated schema. When it is absent there is
  // nothing to rule the fileset out on, so the content checks below decide instead.
  const storageType = existing.storage.type;
  if (storageType && !WRITABLE_STORAGE_TYPES.has(storageType)) {
    return conflict(
      'unwritable',
      `already exists on ${storageType} storage, which does not accept uploads.`
    );
  }

  if (existing.purpose !== FilesetPurpose.dataset) {
    return conflict(
      'purpose',
      `already exists, but holds ${existing.purpose} files rather than a training dataset.`
    );
  }

  const listed = await filesListFilesetFiles(workspace, name);
  const sizeByPath = new Map(listed.data.map((file) => [file.path, file.size]));
  const missing = TEMPLATE_DATASET_FILES.filter((path) => !sizeByPath.get(path));
  if (missing.length > 0) {
    return conflict(
      'incomplete',
      `already exists but is missing ${missing.join(' and ')}, so it is not a complete training set.`
    );
  }

  // The listing proves the splits are there and non-empty; this proves they hold the
  // prompt/completion pairs the recipe trains on rather than some other JSONL.
  //
  // A head that cannot be read or parsed is deliberately not a conflict. The files exist,
  // are non-empty and sit on writable storage, and failing here would turn a transient
  // read error into the dead end this check exists to remove.
  const row = firstRow(await readHead(TEMPLATE_DATASET_FILES[0]));
  if (row && !REQUIRED_ROW_KEYS.every((key) => typeof row[key] === 'string')) {
    return conflict(
      'schema',
      'already exists, but its rows are not the prompt/completion pairs this recipe trains on.'
    );
  }

  return { kind: 'reusable', filesetRef: `${workspace}/${name}` };
};

/**
 * First free `name-2`, `name-3`, … so "create under a different name" lands somewhere
 * predictable instead of a random suffix the user then has to recognise.
 *
 * Falls back to the last candidate rather than throwing: the create that follows reports a
 * collision far better than a bare "ran out of names" would.
 */
export const findAvailableFilesetName = async (
  workspace: string,
  base: string,
  limit = 20
): Promise<string> => {
  let candidate = base;
  for (let suffix = 2; suffix <= limit; suffix += 1) {
    candidate = `${base}-${suffix}`;
    if (!(await retrieveIfPresent(workspace, candidate))) return candidate;
  }
  return candidate;
};
