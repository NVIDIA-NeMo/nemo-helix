// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Compartment, type Extension } from '@codemirror/state';
import type { EditorView } from '@codemirror/view';
import { useCallback, useEffect, useMemo } from 'react';

export function useExtensionWithDependency(
  view: EditorView | null,
  extension: Extension,
  condition: boolean
) {
  const compartment = useMemo(() => new Compartment(), []);
  const extensionFactory = useCallback(() => (condition ? extension : []), [condition, extension]);
  const extensionMemoized = useMemo(
    () => compartment.of(extensionFactory()),
    [compartment, extensionFactory]
  );

  useEffect(() => {
    if (view) {
      view.dispatch({
        effects: compartment.reconfigure(extensionFactory()),
      });
    }
  }, [compartment, condition, extension, extensionFactory, view]);

  return extensionMemoized;
}
