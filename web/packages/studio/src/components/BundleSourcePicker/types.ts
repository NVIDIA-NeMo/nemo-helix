// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BundleSource } from '@studio/api/files/types';
import type { Control } from 'react-hook-form';

/**
 * What a {@link BundleSourcePicker} is looking for in a bundle: the one file that drives it (an
 * optimize config, an eval config, ...) among the assets it references. Keep it referentially
 * stable — define it at module scope, or `useMemo` it when it closes over props.
 */
export interface BundleFileSpec<T> {
  /** Lower-case noun for the file, e.g. "optimize config". Shown in labels and messages. */
  label: string;
  /** What a valid file looks like, completing "expected …", e.g. "a YAML file with an optimizer section". */
  description: string;
  /** Plural noun for the candidates, completing "This fileset has no …", e.g. "YAML files". */
  candidateNoun: string;
  /** Which bundle paths could be the file. Only these are offered or read. */
  isCandidate: (path: string) => boolean;
  /** The parsed file, or undefined when this candidate is not the file after all. */
  parse: (text: string) => T | undefined;
  /** Problems that block using the file, given every path in its bundle (relative to the root). */
  validate?: (file: T, bundlePaths: ReadonlySet<string>) => string[];
}

export type BundleSourceKind = 'upload' | 'fileset';

/** A bundle and the file in it that parsed and passed validation. */
export interface BundleSelection<T> {
  source: BundleSource;
  /** The chosen file, relative to the bundle root. */
  path: string;
  file: T;
}

/** What each source reports, so the picker renders either one the same way. */
export interface BundleSourceSideState<T> {
  /** Paths that could be the file, relative to the bundle root. */
  candidatePaths: string[];
  path: string;
  setPath: (path: string) => void;
  /** Validation problems in the chosen file; there is no selection while there are any. */
  problems: string[];
  /** The bundle or the chosen file could not be read. */
  error?: string;
  /** Still reading the bundle or the chosen file. */
  isLoading: boolean;
  selection?: BundleSelection<T>;
  reset: () => void;
}

export interface UploadedBundleState<T> extends BundleSourceSideState<T> {
  /** "folder — 3 files, 12 KB", once a selection was read. */
  summary?: string;
  onFilesPicked: (files: File[]) => void;
  onItemsDropped: (items: DataTransferItem[]) => void;
}

export interface FilesetFormValues {
  fileset: string;
}

export interface FilesetBundleState<T> extends BundleSourceSideState<T> {
  control: Control<FilesetFormValues>;
  fileset: string;
  /** The chosen fileset holds no candidates at all. */
  hasNoCandidates: boolean;
}

export interface BundleSourceState<T> {
  spec: BundleFileSpec<T>;
  source: BundleSourceKind;
  setSource: (source: BundleSourceKind) => void;
  upload: UploadedBundleState<T>;
  fileset: FilesetBundleState<T>;
  /** The active source's state; read `active.selection` to act on the bundle. */
  active: BundleSourceSideState<T>;
  /** Clears both sources and returns to uploading, e.g. when a dialog closes. */
  reset: () => void;
}
