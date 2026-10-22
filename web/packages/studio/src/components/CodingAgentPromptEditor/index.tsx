// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Button, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { CopyGeneric } from '@nvidia/react-gui-icons/icons';
import { type FC, lazy, Suspense, useState } from 'react';

const CodeEditor = lazy(() =>
  import('@nemo/common/src/components/CodeEditor').then((module) => ({
    default: module.CodeEditor,
  }))
);

const editorFallback = (
  <Flex align="center" justify="center" className="h-64">
    <Spinner size="medium" aria-label="Loading editor..." />
  </Flex>
);

interface CodingAgentPromptEditorProps {
  /** Starting point for the prompt. Editable before copying. */
  prompt: string;
  className?: string;
}

/**
 * A coding agent prompt, editable before it is copied — the prompts name an agent and a workspace,
 * and the user is the one who knows when those need adjusting before the handoff.
 */
export const CodingAgentPromptEditor: FC<CodingAgentPromptEditorProps> = ({
  prompt,
  className = 'h-64',
}) => {
  const { success, error } = useToast();
  const [content, setContent] = useState(prompt);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(content);
      success('Prompt copied — paste it into your coding agent.');
    } catch {
      error('Error copying prompt to clipboard');
    }
  };

  return (
    <Stack gap="2" className={className}>
      <Suspense fallback={editorFallback}>
        <CodeEditor
          content={content}
          contentType={ContentType.TEXT}
          onChange={setContent}
          className="h-full min-h-0"
          hideLineNumbers
          hideFoldGutter
          hideLinter
          hideCopyButton
          slotLabel={<Text kind="body/semibold/sm">Coding agent prompt</Text>}
          slotControls={
            <Button
              kind="tertiary"
              size="tiny"
              aria-label="CopyGeneric to clipboard"
              onClick={copyToClipboard}
            >
              <CopyGeneric size="16" className="text-brand" aria-hidden />
            </Button>
          }
        />
      </Suspense>
    </Stack>
  );
};
