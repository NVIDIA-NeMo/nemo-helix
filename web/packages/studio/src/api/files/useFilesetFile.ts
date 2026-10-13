// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFetchFileAsArrayBuffer } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { useQuery, UseQueryResult } from '@tanstack/react-query';

interface UseFilesetFileParams {
  workspace: string;
  fileset: string;
  path: string;
  enabled?: boolean;
}

/**
 * Downloads a fileset file byte-for-byte as a `File`, off the main thread. The bytes are kept
 * only while this path stays selected, so choosing it again later downloads its current contents.
 */
export const useFilesetFile = ({
  workspace,
  fileset,
  path,
  enabled = true,
}: UseFilesetFileParams): UseQueryResult<File, Error> => {
  const fetchFile = useFetchFileAsArrayBuffer();

  return useQuery({
    queryKey: ['fileset-file', workspace, fileset, path],
    queryFn: async ({ signal }) => {
      const buffer = await fetchFile({ workspace, datasetName: fileset, path, signal });
      return new File([buffer], path.split('/').pop() ?? path);
    },
    enabled: enabled && !!fileset && !!path,
    staleTime: Infinity,
    gcTime: 0,
  });
};
