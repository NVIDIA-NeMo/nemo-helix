// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { CodeSnippet, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface CodingAgentPromptEditorProps {
  /** The prompt to display and copy. */
  prompt: string;
  className?: string;
}

/**
 * A read-only coding agent prompt with a copy button. The prompts name an agent and a workspace;
 * the user adjusts those in their coding agent after pasting.
 */
export const CodingAgentPromptEditor: FC<CodingAgentPromptEditorProps> = ({
  prompt,
  className = 'h-64',
}) => {
  const { success } = useToast();

  return (
    <Stack className={className}>
      {/* KUI only caps a snippet's height via `rows`; fill the wrapper and scroll the code instead. */}
      <CodeSnippet
        value={prompt}
        language="text"
        kind="block"
        className="flex h-full min-h-0 flex-col"
        slotActions={<Text kind="body/semibold/sm">Coding agent prompt</Text>}
        onCopySuccess={() => success('Prompt copied — paste it into your coding agent.')}
        attributes={{
          CodeSnippetCode: {
            className:
              'min-h-0 flex-1 overflow-auto [&_code]:whitespace-pre-wrap [&_code]:break-words [&_pre]:whitespace-pre-wrap',
          },
        }}
      />
    </Stack>
  );
};
