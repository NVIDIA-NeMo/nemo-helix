// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Compartment, type Extension } from '@codemirror/state';
import type { EditorView } from '@codemirror/view';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { jsonlLinter, jsonLinter } from '@nemo/common/src/components/CodeEditor/linters/JsonLinter';
import { yamlLinter } from '@nemo/common/src/components/CodeEditor/linters/yaml';
import { useEffect, useMemo } from 'react';

export function useLinter(view: EditorView | null, contentType: ContentType, hideLinter: boolean) {
  const compartment = useMemo(() => new Compartment(), []);
  const extension = useMemo(
    () => compartment.of(getLinterExtension(contentType, hideLinter)),
    [compartment, contentType, hideLinter]
  );

  useEffect(() => {
    if (view) {
      view.dispatch({
        effects: compartment.reconfigure(getLinterExtension(contentType, hideLinter)),
      });
    }
  }, [compartment, contentType, hideLinter, view]);

  return extension;
}

export const getLinterExtension = (contentType: ContentType, hideLinter: boolean): Extension => {
  if (hideLinter) return [];
  switch (contentType) {
    case ContentType.YAML:
      return yamlLinter;
    case ContentType.JSON:
      return jsonLinter;
    case ContentType.JSONL:
      return jsonlLinter;
    case ContentType.TEXT:
    default:
      return [];
  }
};
