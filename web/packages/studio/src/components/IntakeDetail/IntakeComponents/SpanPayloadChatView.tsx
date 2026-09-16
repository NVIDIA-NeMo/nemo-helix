// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type {
  ChatMessage,
  ChatToolCall,
} from '@studio/components/IntakeDetail/IntakeComponents/openaiChat';
import { PayloadPending } from '@studio/components/IntakeDetail/IntakeComponents/PayloadPending';
import { ChevronRight } from 'lucide-react';
import { type FC, lazy, type ReactNode, Suspense } from 'react';

// The same lazy chunk the markdown format uses; message bodies are markdown too.
const MarkdownContent = lazy(() =>
  import('@nemo/common/src/components/MarkdownContent').then((module) => ({
    default: module.MarkdownContent,
  }))
);

/** Roles whose turn is context rather than conversation, so it starts folded away. */
const COLLAPSED_ROLES = new Set(['system', 'developer', 'tool']);

const ROLE_LABELS: Record<string, string> = {
  system: 'System prompt',
  developer: 'Developer prompt',
  tool: 'Tool result',
  user: 'User',
  assistant: 'Assistant',
};

const roleLabel = (role: string) => ROLE_LABELS[role] ?? role;

interface DisclosureProps {
  summary: string;
  /** Rendered verbatim after the summary — a tool name is an identifier, not a label. */
  detail?: string;
  children: ReactNode;
}

/**
 * Native `<details>`, so a folded turn costs no state and still opens under
 * find-in-page. Nesting inside the accordion's own `<details>` is valid.
 */
const Disclosure: FC<DisclosureProps> = ({ summary, detail, children }) => (
  <details
    className="group rounded-md border border-base bg-surface-raised"
    data-testid="chat-turn-detail"
  >
    <summary className="flex cursor-pointer list-none items-center gap-density-xs px-density-md py-density-sm text-secondary [&::-webkit-details-marker]:hidden">
      <ChevronRight
        size={14}
        className="shrink-0 transition-transform group-open:rotate-90"
        aria-hidden
      />
      <Text kind="label/regular/xs" className="uppercase">
        {summary}
      </Text>
      {detail ? (
        <Text kind="label/regular/xs" className="truncate font-mono text-subtle">
          {detail}
        </Text>
      ) : null}
    </summary>
    <div className="border-t border-base px-density-md py-density-sm">{children}</div>
  </details>
);

const ToolCallBlock: FC<{ call: ChatToolCall }> = ({ call }) => (
  <Disclosure summary="Tool call" detail={call.name}>
    <pre className="overflow-x-auto text-xs whitespace-pre-wrap text-secondary">
      {call.arguments || '(no arguments)'}
    </pre>
  </Disclosure>
);

const MessageBody: FC<{ text: string }> = ({ text }) => (
  <div className="[&_pre]:whitespace-pre-wrap [&_table]:block [&_table]:overflow-x-auto">
    <MarkdownContent content={text} />
  </div>
);

const ChatTurn: FC<{ message: ChatMessage }> = ({ message }) => {
  const label = roleLabel(message.role);

  if (COLLAPSED_ROLES.has(message.role)) {
    return (
      <Disclosure summary={label}>
        <MessageBody text={message.content} />
      </Disclosure>
    );
  }

  const fromUser = message.role === 'user';

  return (
    <div className={`flex flex-col gap-density-xs ${fromUser ? 'items-end' : 'items-start'}`}>
      <Text kind="label/regular/xs" className="uppercase text-subtle">
        {label}
      </Text>
      <div
        className={`flex max-w-[85%] min-w-0 flex-col gap-density-sm rounded-lg border px-density-lg py-density-md ${
          fromUser ? 'border-strong bg-surface-sunken' : 'border-base bg-surface-raised'
        }`}
      >
        {message.reasoning ? (
          <Disclosure summary="Reasoning">
            <MessageBody text={message.reasoning} />
          </Disclosure>
        ) : null}
        {message.toolCalls?.map((call, index) => (
          <ToolCallBlock key={call.id ?? `${call.name}-${index}`} call={call} />
        ))}
        {message.content ? <MessageBody text={message.content} /> : null}
      </div>
    </div>
  );
};

interface SpanPayloadChatViewProps {
  messages: ChatMessage[];
}

/**
 * An OpenAI-compatible payload read as the conversation it describes. System,
 * tool, and reasoning turns start collapsed so the user and assistant exchange
 * is what the section shows.
 */
export const SpanPayloadChatView: FC<SpanPayloadChatViewProps> = ({ messages }) => (
  <Suspense fallback={<PayloadPending />}>
    <div
      className="flex max-h-[420px] flex-col gap-density-lg overflow-auto rounded-md border border-base bg-surface-base p-density-lg"
      data-testid="span-payload-chat"
    >
      {messages.map((message, index) => (
        <ChatTurn key={`${message.role}-${index}`} message={message} />
      ))}
    </div>
  </Suspense>
);
