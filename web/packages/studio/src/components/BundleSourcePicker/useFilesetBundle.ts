// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type {
  BundleFileSpec,
  FilesetBundleState,
  FilesetFormValues,
} from '@studio/components/BundleSourcePicker/types';
import { useFetchFileAsArrayBuffer } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { useQuery } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';

/**
 * A bundle that already lives in a fileset: the fileset is the bundle root. Every candidate is
 * offered, but only the chosen one is downloaded, then validated against the fileset's listing.
 */
export const useFilesetBundle = <T>(
  workspace: string,
  spec: BundleFileSpec<T>
): FilesetBundleState<T> => {
  const { control, reset: resetForm } = useForm<FilesetFormValues>({
    defaultValues: { fileset: '' },
  });
  const fileset = useWatch({ control, name: 'fileset' });
  const [pickedPath, setPickedPath] = useState('');

  const filesQuery = useFilesListFilesetFiles(workspace, fileset, undefined, {
    query: { enabled: !!fileset },
  });
  const filesetPaths = useMemo(
    () => (filesQuery.data?.data ?? []).map((file) => file.path),
    [filesQuery.data]
  );
  const candidatePaths = useMemo(
    () => filesetPaths.filter(spec.isCandidate).sort(),
    [filesetPaths, spec]
  );
  // A lone candidate is the only choice, so it is chosen without asking.
  const path = pickedPath || (candidatePaths.length === 1 ? (candidatePaths[0] ?? '') : '');

  const fetchFile = useFetchFileAsArrayBuffer();
  const textQuery = useQuery({
    queryKey: ['bundle-file-text', workspace, fileset, path],
    queryFn: async ({ signal }) =>
      new TextDecoder().decode(await fetchFile({ workspace, datasetName: fileset, path, signal })),
    enabled: !!fileset && !!path,
    // Dropped once deselected, so choosing it again validates the fileset's current contents.
    staleTime: Infinity,
    gcTime: 0,
  });

  const file = useMemo(
    () => (textQuery.data === undefined ? undefined : spec.parse(textQuery.data)),
    [textQuery.data, spec]
  );

  const problems = useMemo(() => {
    if (!textQuery.isSuccess) return [];
    if (file === undefined)
      return [`${path} is not a valid ${spec.label}: expected ${spec.description}.`];
    return spec.validate?.(file, new Set(filesetPaths)) ?? [];
  }, [textQuery.isSuccess, file, path, filesetPaths, spec]);

  const queryError = filesQuery.error ?? textQuery.error;

  return {
    control,
    fileset,
    hasNoCandidates: filesQuery.isSuccess && candidatePaths.length === 0,
    candidatePaths,
    path,
    setPath: setPickedPath,
    problems,
    error: queryError
      ? getErrorMessage(queryError as Error) || 'Could not read that fileset.'
      : undefined,
    isLoading: filesQuery.isFetching || textQuery.isFetching,
    selection:
      file !== undefined && problems.length === 0 ? { source: { fileset }, path, file } : undefined,
    reset: () => {
      resetForm();
      setPickedPath('');
    },
  };
};
