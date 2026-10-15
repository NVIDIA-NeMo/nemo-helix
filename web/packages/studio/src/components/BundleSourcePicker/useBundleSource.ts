// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  BundleFileSpec,
  BundleSourceKind,
  BundleSourceState,
} from '@studio/components/BundleSourcePicker/types';
import { useFilesetBundle } from '@studio/components/BundleSourcePicker/useFilesetBundle';
import { useUploadedBundle } from '@studio/components/BundleSourcePicker/useUploadedBundle';
import { useState } from 'react';

/**
 * State for {@link BundleSourcePicker}. The owner keeps it so it can read `active.selection`, which
 * is set only once the chosen file parses and passes `spec.validate`. Each source keeps its own
 * pick, so switching back and forth loses nothing.
 */
export const useBundleSource = <T>({
  workspace,
  spec,
}: {
  workspace: string;
  spec: BundleFileSpec<T>;
}): BundleSourceState<T> => {
  const [source, setSource] = useState<BundleSourceKind>('upload');
  const upload = useUploadedBundle(spec);
  const fileset = useFilesetBundle(workspace, spec);

  return {
    spec,
    source,
    setSource,
    upload,
    fileset,
    active: source === 'upload' ? upload : fileset,
    reset: () => {
      upload.reset();
      fileset.reset();
      setSource('upload');
    },
  };
};
