// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadAssistantMessagePart } from '@assistant-ui/react';
import { logger } from '@nemo/common/src/utils/logger';
import {
  createAssistantThinkingPart,
  createAssistantToolCallPart,
  groupConsecutiveAssistantSubtleToolCalls,
} from '@studio/routes/agents/AssistantChatRoute/toolParts';

interface ServerSentEvent {
  event?: string;
  data: string;
}

interface ParsedSseChunk {
  events: ServerSentEvent[];
  rest: string;
}

interface AssistantContentPart {
  id?: unknown;
  input?: unknown;
  type?: unknown;
  text?: unknown;
  name?: unknown;
}

interface AssistantMessage {
  id?: unknown;
  content?: unknown;
}

interface AssistantStreamEvent {
  type?: unknown;
  message?: AssistantMessage;
}

export const parseSseChunk = (chunk: string): ParsedSseChunk => {
  const normalized = chunk.replace(/\r\n/g, '\n');
  const blocks = normalized.split('\n\n');
  const rest = blocks.pop() ?? '';

  return {
    rest,
    events: blocks
      .map((block) => {
        const lines = block.split('\n');
        let event: string | undefined;
        const dataLines: string[] = [];

        for (const line of lines) {
          if (line.startsWith('event:')) {
            event = line.slice('event:'.length).trim();
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
          }
        }

        return { event, data: dataLines.join('\n') };
      })
      .filter((event) => event.event || event.data),
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const getContentParts = (event: AssistantStreamEvent): AssistantContentPart[] => {
  const content = event.message?.content;
  if (!Array.isArray(content)) return [];
  return content.filter(isRecord);
};

export const getAssistantPartsFromAssistantEvent = (
  event: unknown
): readonly ThreadAssistantMessagePart[] => {
  if (!isRecord(event) || event.type !== 'assistant') return [];

  const parts = getContentParts(event);
  const message = event.message;
  const messageId =
    isRecord(message) && typeof message.id === 'string' && message.id ? message.id : 'message';
  const assistantParts = parts
    .map((part, index): ThreadAssistantMessagePart | undefined => {
      if (part.type === 'text' && typeof part.text === 'string') {
        return part.text ? { type: 'text', text: part.text } : undefined;
      }
      if (part.type === 'reasoning' && typeof part.text === 'string') {
        return part.text.trim()
          ? createAssistantThinkingPart(part.text, `assistant-thinking-${messageId}-${index}`)
          : undefined;
      }
      if (part.type === 'tool_use') {
        const toolName = typeof part.name === 'string' ? part.name : 'tool';

        const toolCallId =
          typeof part.id === 'string' && part.id
            ? part.id
            : `assistant-tool-${messageId}-${toolName}-${index}`;
        return createAssistantToolCallPart({
          input: part.input,
          toolCallId,
          toolName,
        });
      }
      return undefined;
    })
    .filter((part): part is ThreadAssistantMessagePart => part !== undefined);

  return groupConsecutiveAssistantSubtleToolCalls(assistantParts);
};

export const getAssistantTextFromAssistantEvent = (event: unknown): string => {
  const parts = getAssistantPartsFromAssistantEvent(event);
  return parts
    .map((part) => {
      if (part.type === 'text') return part.text;
      return '';
    })
    .join('');
};

export const parseJsonObject = (value: string): unknown => {
  if (!value) return undefined;
  try {
    return JSON.parse(value) as unknown;
  } catch (error) {
    logger.error('Failed to parse NeMo Assistant stream JSON', error);
    return undefined;
  }
};
