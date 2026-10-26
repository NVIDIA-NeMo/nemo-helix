// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EditorView } from '@codemirror/view';
import { useExtensionWithDependency } from '@nemo/common/src/components/CodeEditor/extensions/useExtensionWithDependency';
import {
  LOG_LANGUAGE,
  LOG_VIEWER_BASIC_SETUP,
} from '@nemo/common/src/components/LogViewer/constants';
import {
  pinToBottom,
  stickToBottom,
} from '@nemo/common/src/components/LogViewer/extensions/stickToBottom';
import { useShikiHighlighting } from '@nemo/common/src/components/LogViewer/extensions/useShikiHighlighting';
import { useTheme } from '@nvidia/foundations-react-core';
import CodeMirror from '@uiw/react-codemirror';
import classNames from 'classnames';
import { FC, useEffect, useMemo, useRef, useState } from 'react';

// Focusable for keyboard scrolling and Cmd+F search without making the editor editable.
const FOCUSABLE_CONTENT = EditorView.contentAttributes.of({ tabindex: '0' });

interface LogEditorProps {
  text: string;
  /** Changing this swaps the content out wholesale and re-pins to the bottom (e.g. tail → all). */
  resetKey: unknown;
  wrapLines: boolean;
  fillHeight: boolean;
  scrollLocked: boolean;
}

/** Owns the editor's document so new log lines are appended, keeping the reader's scroll position. */
export const LogEditor: FC<LogEditorProps> = ({
  text,
  resetKey,
  wrapLines,
  fillHeight,
  scrollLocked,
}) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [view, setView] = useState<EditorView | null>(null);
  // CodeMirror's starting document, and never changed after: @uiw/react-codemirror replaces
  // the whole document whenever `value` changes, so the effect below syncs it instead.
  const [initialText] = useState(text);
  // What the document was last synced from, so the next update can tell an append apart.
  const syncedRef = useRef({ text, resetKey });

  // Colors come from Shiki; this only flags dark mode for the search panel and matches.
  const editorTheme = useMemo(() => EditorView.theme({}, { dark: isDark }), [isDark]);
  const highlightingExtension = useShikiHighlighting(LOG_LANGUAGE, isDark ? 'dark' : 'light');
  const lineWrappingExtension = useExtensionWithDependency(
    view,
    EditorView.lineWrapping,
    wrapLines
  );
  const extensions = useMemo(
    () => [FOCUSABLE_CONTENT, stickToBottom, highlightingExtension, lineWrappingExtension],
    [highlightingExtension, lineWrappingExtension]
  );

  useEffect(() => {
    if (!view) return;
    const synced = syncedRef.current;
    if (synced.text === text && synced.resetKey === resetKey) return;

    const isReset = synced.resetKey !== resetKey;
    // Appending leaves the lines above untouched, so CodeMirror keeps the reader where they
    // are. Insert at the document's end rather than at the old text's length: CodeMirror
    // normalizes \r\n, so the two can differ.
    const appends = !isReset && text.startsWith(synced.text);
    view.dispatch({
      changes: appends
        ? { from: view.state.doc.length, insert: text.slice(synced.text.length) }
        : { from: 0, to: view.state.doc.length, insert: text },
      effects: isReset ? pinToBottom.of(null) : [],
    });
    syncedRef.current = { text, resetKey };
  }, [view, text, resetKey]);

  return (
    <CodeMirror
      className={classNames('min-h-0 min-w-0 flex-1', {
        'log-viewer--scroll-locked': scrollLocked,
      })}
      value={initialText}
      readOnly
      editable={false}
      basicSetup={LOG_VIEWER_BASIC_SETUP}
      extensions={extensions}
      theme={editorTheme}
      height={fillHeight ? '100%' : undefined}
      maxHeight={fillHeight ? undefined : 'calc(var(--nv-code-snippet-rows) * 1lh)'}
      onCreateEditor={setView}
      data-testid="log-viewer-code"
    />
  );
};
