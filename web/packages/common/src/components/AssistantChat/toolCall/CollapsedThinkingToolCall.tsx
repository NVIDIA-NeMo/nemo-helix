// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { splitCollapsedThinkingParagraphs } from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import { Text } from '@nvidia/foundations-react-core';
import { ChevronRight, IconClipboard } from '@nvidia/react-gui-icons/icons';

export const CollapsedThinkingToolCall = ({
  label = 'Earlier thinking',
  text,
}: {
  readonly label?: string;
  readonly text: string;
}) => {
  const paragraphs = splitCollapsedThinkingParagraphs(text);
  if (!paragraphs.length) return null;

  return (
    <Text asChild kind="body/regular/sm">
      <details
        className="group/thinking my-density-xs max-w-full text-gray-500 dark:text-gray-400"
        data-testid="assistant-collapsed-thinking"
      >
        <summary className="inline-flex cursor-pointer list-none items-center gap-density-xs marker:hidden">
          <ChevronRight
            aria-hidden
            className="size-3 shrink-0 transition-transform group-open/thinking:rotate-90"
          />
          <IconClipboard aria-hidden className="size-3.5 shrink-0" />
          <span>{label}</span>
        </summary>
        <div
          className="mt-density-xs space-y-density-xs border-l border-base pl-density-md text-secondary"
          data-testid="assistant-collapsed-thinking-content"
        >
          {paragraphs.map((paragraph, index) => (
            <p key={`${paragraph.slice(0, 24)}-${index}`} className="whitespace-pre-wrap">
              {paragraph}
            </p>
          ))}
        </div>
      </details>
    </Text>
  );
};
