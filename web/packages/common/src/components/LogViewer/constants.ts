// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CodeSnippetLanguage } from '@nvidia/foundations-react-core';
import type { BasicSetupOptions } from '@uiw/react-codemirror';

export const LOG_LANGUAGE: CodeSnippetLanguage = 'shell';

/** Read-only: keep selection and search; drop every editing aid and the gutter. */
export const LOG_VIEWER_BASIC_SETUP: BasicSetupOptions = {
  // Its navigation keys move an invisible cursor; without it the browser scrolls.
  defaultKeymap: false,
  // Control characters would render as red placeholders (ANSI codes are stripped anyway).
  highlightSpecialChars: false,
  // Each sync would be an undo step holding the replaced text, and Mod-u (which ignores
  // readOnly) could revert the document behind LogEditor's back.
  history: false,
  historyKeymap: false,
  lineNumbers: false,
  foldGutter: false,
  drawSelection: false,
  highlightActiveLine: false,
  highlightActiveLineGutter: false,
  dropCursor: false,
  indentOnInput: false,
  bracketMatching: false,
  closeBrackets: false,
  autocompletion: false,
  highlightSelectionMatches: true,
  searchKeymap: true,
};
