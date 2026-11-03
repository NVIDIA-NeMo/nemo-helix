// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { javascript } from '@codemirror/lang-javascript';
import { json } from '@codemirror/lang-json';
import { python } from '@codemirror/lang-python';
import { yaml } from '@codemirror/lang-yaml';
import { Compartment, type Extension } from '@codemirror/state';
import type { EditorView } from '@codemirror/view';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { useEffect, useMemo } from 'react';

export function useLanguageExtension(view: EditorView | null, contentType: ContentType) {
  const compartment = useMemo(() => new Compartment(), []);
  const extension = useMemo(
    () => compartment.of(getLanguageExtension(contentType)),
    [compartment, contentType]
  );

  useEffect(() => {
    if (view) {
      view.dispatch({
        effects: compartment.reconfigure(getLanguageExtension(contentType)),
      });
    }
  }, [compartment, contentType, view]);

  return extension;
}

const getLanguageExtension = (contentType: ContentType): Extension => {
  switch (contentType) {
    case ContentType.JAVASCRIPT:
      return javascript();
    case ContentType.PYTHON:
      return python();
    case ContentType.YAML:
      return yaml();
    case ContentType.JSON:
    case ContentType.JSONL:
      return json();
    case ContentType.TEXT:
    default:
      return [];
  }
};
