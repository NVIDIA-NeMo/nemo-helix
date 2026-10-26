// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { foldGutter } from '@codemirror/language';
import { EditorView, lineNumbers } from '@codemirror/view';
import { BASIC_SETUP, ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { useExtensionWithDependency } from '@nemo/common/src/components/CodeEditor/extensions/useExtensionWithDependency';
import { useLanguageExtension } from '@nemo/common/src/components/CodeEditor/extensions/useLanguageExtension';
import { useLinter } from '@nemo/common/src/components/CodeEditor/extensions/useLinter';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { CopyGeneric } from '@nemo/icons/icons';
import { Flex, Button, useTheme, Stack } from '@nvidia/foundations-react-core';
import { githubLight, githubDark } from '@uiw/codemirror-theme-github';
import CodeMirror from '@uiw/react-codemirror';
import { type FC, type MouseEvent, type ReactNode, useState, useMemo, useCallback } from 'react';

import '@nemo/common/src/components/CodeEditor/styles.css';

export interface CodeEditorProps {
  content: string;
  readOnly?: boolean;
  onChange?: (newContent: string) => void;
  contentType?: ContentType;
  id?: string;
  slotLabel?: ReactNode;
  slotControls?: ReactNode;
  hideCopyButton?: boolean;
  hideLineNumbers?: boolean;
  hideFoldGutter?: boolean;
  enableLineWrapping?: boolean;
  hideLinter?: boolean;
  className?: string;
}

export const CodeEditor: FC<CodeEditorProps> = ({
  content,
  readOnly = false,
  contentType = ContentType.JSON,
  onChange,
  className,
  id,
  slotLabel,
  slotControls,
  hideCopyButton = false,
  hideLineNumbers = false,
  hideFoldGutter = false,
  enableLineWrapping = true,
  hideLinter = false,
}) => {
  const { theme } = useTheme();
  const toast = useToast();
  const [view, setView] = useState<EditorView | null>(null);

  const handleChange = (newContent: string) => {
    onChange?.(newContent);
  };

  const copyToClipboard = useCallback(
    async (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (!view) return;
      try {
        const text = view.state.doc.toString();
        await navigator.clipboard.writeText(text);
        toast.success('Copied to clipboard');
      } catch {
        toast.error('Failed to copy to clipboard');
      }
    },
    [toast, view]
  );

  const languageExtension = useLanguageExtension(view, contentType);
  const linterExtension = useLinter(view, contentType, !hideLinter);
  const lineNumbersExtension = useExtensionWithDependency(view, lineNumbers(), !hideLineNumbers);
  const foldGutterExtension = useExtensionWithDependency(view, foldGutter(), !hideFoldGutter);
  const lineWrappingExtension = useExtensionWithDependency(
    view,
    EditorView.lineWrapping,
    enableLineWrapping
  );

  const extensions = useMemo(
    () => [
      languageExtension,
      linterExtension,
      lineNumbersExtension,
      foldGutterExtension,
      lineWrappingExtension,
    ],
    [
      languageExtension,
      linterExtension,
      lineNumbersExtension,
      foldGutterExtension,
      lineWrappingExtension,
    ]
  );

  return (
    <Stack gap="density-sm" className={className}>
      {(slotLabel || !hideCopyButton || slotControls) && (
        <Flex justify="between" align="center" gap="density-sm" className="shrink-0">
          <Flex
            align="center"
            gap="density-sm"
            className="min-w-0"
            role="button"
            tabIndex={0}
            onClick={() => view?.focus()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                view?.focus();
              }
            }}
          >
            {slotLabel}
          </Flex>
          <Flex align="center" gap="density-sm">
            {slotControls}
            {!hideCopyButton && (
              <Button
                onClick={copyToClipboard}
                kind="tertiary"
                size="tiny"
                aria-label="Copy to clipboard"
              >
                <CopyGeneric />
              </Button>
            )}
          </Flex>
        </Flex>
      )}
      <CodeMirror
        id={id}
        data-testid="nv-code-editor-root"
        className="min-h-0"
        value={content}
        basicSetup={BASIC_SETUP}
        height="100%"
        theme={theme === 'dark' ? githubDark : githubLight}
        onCreateEditor={setView}
        extensions={extensions}
        readOnly={readOnly}
        onChange={handleChange}
      />
    </Stack>
  );
};
