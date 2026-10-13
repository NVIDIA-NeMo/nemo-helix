// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import type { FilesetEntry } from '@studio/api/files/types';
import type {
  BundleFileSpec,
  UploadedBundleState,
} from '@studio/components/BundleSourcePicker/types';
import { MAX_PICKED_FILES } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type { PickedFile } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import {
  collectAgentEntries,
  pickedFromDataTransfer,
  pickedFromFileList,
  selectionRootName,
  totalEntryBytes,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { useMemo, useRef, useState } from 'react';

interface Bundle<T> {
  label: string;
  entries: FilesetEntry[];
  /** Every candidate that parsed, keyed by path. */
  files: Record<string, T>;
}

const readBundle = async <T>(picked: PickedFile[], spec: BundleFileSpec<T>): Promise<Bundle<T>> => {
  const entries = collectAgentEntries(picked);
  if (entries.length === 0) throw new Error('That selection has no uploadable files.');

  // Local reads are cheap, so every candidate is parsed up front and only real matches are offered.
  const parsed = await Promise.all(
    entries
      .filter((entry) => spec.isCandidate(entry.path))
      .map(async (entry) => [entry.path, spec.parse(await entry.file.text())] as const)
  );
  const files: Record<string, T> = {};
  parsed.forEach(([path, file]) => {
    if (file !== undefined) files[path] = file;
  });
  if (Object.keys(files).length === 0) {
    throw new Error(`No ${spec.label} in that selection: expected ${spec.description}.`);
  }

  return { label: selectionRootName(picked), entries, files };
};

/** A bundle picked from the user's machine, read and validated in the browser before upload. */
export const useUploadedBundle = <T>(spec: BundleFileSpec<T>): UploadedBundleState<T> => {
  const [bundle, setBundle] = useState<Bundle<T> | undefined>();
  const [path, setPath] = useState('');
  const [error, setError] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(false);

  const candidatePaths = useMemo(() => Object.keys(bundle?.files ?? {}).sort(), [bundle]);
  const file = bundle?.files[path];

  const problems = useMemo(() => {
    if (!bundle || file === undefined) return [];
    return spec.validate?.(file, new Set(bundle.entries.map((entry) => entry.path))) ?? [];
  }, [bundle, file, spec]);

  const summary = useMemo(() => {
    if (!bundle) return undefined;
    const count = bundle.entries.length;
    const size = `${count} ${count === 1 ? 'file' : 'files'}, ${Math.max(1, Math.round(totalEntryBytes(bundle.entries) / 1000))} KB`;
    return bundle.label ? `${bundle.label} — ${size}` : size;
  }, [bundle]);

  // Selection reads finish out of order, so the newest selection has to win.
  const selectionSeq = useRef(0);

  const reset = () => {
    selectionSeq.current += 1;
    setBundle(undefined);
    setPath('');
    setError(undefined);
    setIsLoading(false);
  };

  const acceptPicked = async (loadPicked: () => Promise<PickedFile[]> | PickedFile[]) => {
    reset();
    const selection = selectionSeq.current;
    setIsLoading(true);

    try {
      const picked = await loadPicked();
      if (picked.length > MAX_PICKED_FILES) {
        throw new Error(
          `That selection holds ${picked.length.toLocaleString()} files; a bundle should hold only the ${spec.label} and the assets it references.`
        );
      }
      const next = await readBundle(picked, spec);
      if (selection !== selectionSeq.current) return;
      const paths = Object.keys(next.files);
      setBundle(next);
      setPath(paths.length === 1 ? (paths[0] ?? '') : '');
    } catch (cause) {
      if (selection !== selectionSeq.current) return;
      setError(getErrorMessage(cause as Error) || 'Could not read that selection.');
    } finally {
      if (selection === selectionSeq.current) setIsLoading(false);
    }
  };

  return {
    candidatePaths,
    path,
    setPath,
    problems,
    error,
    isLoading,
    summary,
    selection:
      bundle && file !== undefined && problems.length === 0
        ? { source: { entries: bundle.entries }, path, file }
        : undefined,
    reset,
    onFilesPicked: (files) => void acceptPicked(() => pickedFromFileList(files)),
    onItemsDropped: (items) => void acceptPicked(() => pickedFromDataTransfer(items)),
  };
};
