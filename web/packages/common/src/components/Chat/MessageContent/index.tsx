// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { messageMarkdownComponents } from '@nemo/common/src/components/Chat/MessageContent/markdownComponents';
import { MarkdownDataViewTable } from '@nemo/common/src/components/Chat/MessageContent/MarkdownDataViewTable';
import { remarkNormalizeEmptyOrderedListMarkers } from '@nemo/common/src/components/Chat/MessageContent/remarkPlugin';
import {
  MessageTextSizeContext,
  useMessageTextStyle,
} from '@nemo/common/src/components/Chat/MessageContent/textSize';
import type { MessageContentProps } from '@nemo/common/src/components/Chat/MessageContent/types';
import { splitMessageWithLabels } from '@nemo/common/src/components/Chat/MessageContent/utils';
import { CodeDisplay } from '@nemo/common/src/components/CodeDisplay';
import { simpleHash } from '@nemo/common/src/utils/simpleHash';
import { Stack, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { decode } from 'html-entities';
import { type FC, type PropsWithChildren, useMemo } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

export type {
  MarkdownTableOptions,
  MessageContentProps,
} from '@nemo/common/src/components/Chat/MessageContent/types';

/**
 * This component takes a content string from a chat response and converts into a user readable
 * list of snippets using content-specific render types. Currently supports plaintext and code.
 */
export const MessageContent: FC<PropsWithChildren<MessageContentProps>> = ({
  content,
  markdownLinkComponent,
  markdownTableOptions,
  renderAsMarkdown = true,
  textSize = 'md',
}) => (
  <MessageTextSizeContext.Provider value={textSize}>
    <MessageSnippets
      content={content}
      markdownLinkComponent={markdownLinkComponent}
      markdownTableOptions={markdownTableOptions}
      renderAsMarkdown={renderAsMarkdown}
    />
  </MessageTextSizeContext.Provider>
);

const MessageSnippets: FC<Omit<MessageContentProps, 'textSize'>> = ({
  content,
  markdownLinkComponent,
  markdownTableOptions,
  renderAsMarkdown,
}) => {
  const textStyle = useMessageTextStyle();
  const snippets = useMemo(() => splitMessageWithLabels(content), [content]);
  const markdownComponents = useMemo<Components>(
    () => ({
      ...messageMarkdownComponents,
      a: markdownLinkComponent ?? messageMarkdownComponents.a,
      table: ({ children }) => (
        <MarkdownDataViewTable options={markdownTableOptions}>{children}</MarkdownDataViewTable>
      ),
    }),
    [markdownLinkComponent, markdownTableOptions]
  );

  return snippets.map((descriptor, index) => {
    const contentHash = simpleHash(descriptor.value);
    if (descriptor.type === 'plaintext') {
      return (
        <div
          className={cn(
            'max-w-full whitespace-normal font-normal text-primary',
            textStyle.className,
            // Code blocks split a reply into snippets; only the reply's opening element loses its
            // top margin, so a heading that follows a code block keeps its spacing.
            index === 0 && '[&>:first-child]:mt-0'
          )}
          data-testid="chat-message-content-text"
          key={`plaintext-${contentHash}`}
        >
          {renderAsMarkdown ? (
            <Markdown
              remarkPlugins={[remarkGfm, remarkNormalizeEmptyOrderedListMarkers]}
              components={markdownComponents}
            >
              {decode(descriptor.value)}
            </Markdown>
          ) : (
            <Text kind="mono/md" className="whitespace-pre-wrap">
              {decode(descriptor.value)}
            </Text>
          )}
        </div>
      );
    } else if (descriptor.type === 'code') {
      return (
        <Stack key={`code-${contentHash}`}>
          {renderAsMarkdown ? (
            <CodeDisplay data-testid="chat-message-content-text">{descriptor.value}</CodeDisplay>
          ) : (
            <Text kind="mono/md" className="whitespace-pre-wrap">
              {descriptor.value}
            </Text>
          )}
        </Stack>
      );
    }
  });
};
