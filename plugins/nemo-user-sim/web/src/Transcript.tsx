// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, CodeSnippet, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import type { ReactNode } from 'react';
import type { ChatMessage } from './types';

type ToolCall = NonNullable<ChatMessage['tool_calls']>[number];

/** Arguments and results arrive as JSON text; indent them when they parse. */
const pretty = (text: string | undefined) => {
  try {
    return JSON.stringify(JSON.parse(text ?? ''), null, 2);
  } catch {
    return text ?? '';
  }
};

const Label = ({ text, tag, real }: { text: string; tag: string; real: boolean }) => (
  <Flex align="center" gap="density-xs">
    <Text kind="body/semibold/xs" className="text-subtle">
      {text}
    </Text>
    <Tag kind="solid" color={real ? 'green' : 'purple'} density="compact" readOnly>
      {tag}
    </Tag>
  </Flex>
);

const Bubble = ({ right, label, text }: { right: boolean; label: ReactNode; text: string }) => (
  <Flex justify={right ? 'end' : 'start'}>
    {/* Inline width: Studio compiles only the Tailwind classes its own sources use. */}
    <Stack
      gap="density-xs"
      className={`rounded-lg p-3 ${right ? 'bg-surface-raised' : 'bg-surface-sunken'}`}
      style={{ maxWidth: '80%' }}
    >
      {label}
      <Text kind="body/regular/sm" className="whitespace-pre-wrap">
        {text}
      </Text>
    </Stack>
  </Flex>
);

const ToolCallCard = ({
  call,
  result,
  real,
}: {
  call: ToolCall;
  result: string | undefined;
  real: boolean;
}) => (
  <Card density="compact" style={{ maxWidth: '80%' }}>
    <Stack gap="density-sm">
      <Label text={call.function.name} tag="called by your agent" real />
      <Text kind="body/regular/xs" className="text-subtle">
        What it sent
      </Text>
      <CodeSnippet value={pretty(call.function.arguments)} language="json" copyButton="none" />
      <Label text="What came back" tag={real ? 'real result' : 'simulated result'} real={real} />
      {result === undefined ? (
        <Text kind="body/regular/xs" className="text-subtle">
          No result was recorded for this call.
        </Text>
      ) : (
        <CodeSnippet value={pretty(result)} language="json" copyButton="none" />
      )}
    </Stack>
  </Card>
);

/**
 * One conversation as a chat: the simulated user on the right, the agent on the left, and a card per tool
 * call with the arguments the agent sent and the result it got back: simulated, or real for its own tools.
 */
export const Transcript = ({
  messages,
  persona,
  agent,
  ownCalls = [],
}: {
  messages: ChatMessage[];
  persona: string;
  agent: string;
  /** Calls to the agent's own tools: their results are real. */
  ownCalls?: string[];
}) => {
  const results = new Map(
    messages
      .filter((message) => message.role === 'tool' && message.tool_call_id)
      .map((message) => [message.tool_call_id, message.content])
  );
  return (
    <Stack gap="density-sm">
      {messages.map((message, index) => {
        if (message.role === 'tool') {
          // A result with an ID sits on its call's card.
          return message.tool_call_id ? null : (
            <Bubble
              key={index}
              right={false}
              label={<Label text="Tool" tag="simulated result" real={false} />}
              text={message.content ?? ''}
            />
          );
        }
        const label =
          message.role === 'user' ? (
            <Label text={persona} tag="simulated user" real={false} />
          ) : message.role === 'system' ? (
            <Label text="System prompt" tag="set by the probe" real={false} />
          ) : (
            <Label text={agent} tag="your agent" real />
          );
        return (
          <Stack key={index} gap="density-sm">
            {message.content ? (
              <Bubble right={message.role === 'user'} label={label} text={message.content} />
            ) : null}
            {message.tool_calls?.map((call, callIndex) => (
              <ToolCallCard
                key={call.id ?? callIndex}
                call={call}
                result={call.id ? results.get(call.id) : undefined}
                real={!!call.id && ownCalls.includes(call.id)}
              />
            ))}
          </Stack>
        );
      })}
    </Stack>
  );
};
