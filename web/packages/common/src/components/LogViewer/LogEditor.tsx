// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChangeSpec } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import { useExtensionWithDependency } from '@nemo/common/src/components/CodeEditor/extensions/useExtensionWithDependency';
import {
  LOG_LANGUAGE,
  LOG_VIEWER_BASIC_SETUP,
} from '@nemo/common/src/components/LogViewer/constants';
import { diffLogs } from '@nemo/common/src/components/LogViewer/diffLogs';
import {
  isPinnedToBottom,
  pinToBottom,
  stickToBottom,
} from '@nemo/common/src/components/LogViewer/extensions/stickToBottom';
import { useShikiHighlighting } from '@nemo/common/src/components/LogViewer/extensions/useShikiHighlighting';
import { formatLogEntryForDisplay, formatLogsForDisplay } from '@nemo/common/src/utils/logs';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import { useTheme } from '@nvidia/foundations-react-core';
import CodeMirror from '@uiw/react-codemirror';
import classNames from 'classnames';
import { FC, useEffect, useMemo, useRef, useState } from 'react';

// Focusable for keyboard scrolling and Cmd+F search without making the editor editable.
// The label names the textbox role CodeMirror gives the content.
const FOCUSABLE_CONTENT = EditorView.contentAttributes.of({ tabindex: '0', 'aria-label': 'Logs' });

// The display formatter normalizes \r, so entry line counts stay in step with CodeMirror's.
const countLines = (logs: readonly HelixJobLog[]): number =>
  logs.reduce((lines, log) => lines + formatLogEntryForDisplay(log).split('\n').length, 0);

interface LogEditorProps {
  logs: readonly HelixJobLog[];
  /** Changing this swaps the content out wholesale and re-pins to the bottom (e.g. tail → all). */
  resetKey: unknown;
  wrapLines: boolean;
  fillHeight: boolean;
  scrollLocked: boolean;
}

/** Owns the editor's document, applying head trims and tail appends as edits to keep scroll. */
export const LogEditor: FC<LogEditorProps> = ({
  logs,
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
  const [initialText] = useState(() => formatLogsForDisplay(logs));
  // What the document was last synced from, so the next update can be diffed against it.
  const syncedRef = useRef({ logs, resetKey });

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
    if (synced.logs === logs && synced.resetKey === resetKey) return;
    syncedRef.current = { logs, resetKey };

    const isReset = synced.resetKey !== resetKey;
    const diff = isReset ? null : diffLogs(synced.logs, logs);
    if (diff && diff.dropped === 0 && diff.added.length === 0) return;

    const { doc } = view.state;
    // A pinned reader has no place to keep, and with wrapping on a trim-plus-append edit
    // can leave it short of the bottom; a replacement can't.
    if (!diff || (diff.dropped > 0 && isPinnedToBottom(view))) {
      view.dispatch({
        changes: { from: 0, to: doc.length, insert: formatLogsForDisplay(logs) },
        effects: isReset ? pinToBottom.of(null) : [],
      });
      return;
    }

    // A full replace would map CodeMirror's scroll anchor to the top; editing only the edges
    // keeps the lines on screen in place.
    const changes: ChangeSpec[] = [];
    if (diff.dropped > 0) {
      // diffLogs only reports a trim when some entries survive, so this line exists.
      const firstKept = doc.line(countLines(synced.logs.slice(0, diff.dropped)) + 1);
      changes.push({ from: 0, to: firstKept.from });
    }
    if (diff.added.length > 0) {
      const separator = doc.length > 0 ? '\n' : '';
      changes.push({ from: doc.length, insert: separator + formatLogsForDisplay(diff.added) });
    }
    view.dispatch({ changes });
  }, [view, logs, resetKey]);

  return (
    <CodeMirror
      className={classNames('flex min-h-0 min-w-0 flex-1 flex-col', {
        'log-viewer--capped': !fillHeight,
        'log-viewer--scroll-locked': scrollLocked,
      })}
      value={initialText}
      readOnly
      editable={false}
      basicSetup={LOG_VIEWER_BASIC_SETUP}
      extensions={extensions}
      theme={editorTheme}
      height={fillHeight ? '100%' : undefined}
      onCreateEditor={setView}
      data-testid="log-viewer-code"
    />
  );
};
