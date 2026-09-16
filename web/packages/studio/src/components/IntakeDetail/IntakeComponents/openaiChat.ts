// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** One tool invocation an assistant turn asked for. */
export interface ChatToolCall {
  id?: string;
  name: string;
  /** Argument JSON as the model emitted it, not re-parsed, so malformed arguments still show. */
  arguments: string;
}

export interface ChatMessage {
  /** `system`, `developer`, `user`, `assistant`, `tool` — or whatever the payload carried. */
  role: string;
  content: string;
  /** Provider-returned chain of thought (`reasoning_content`). */
  reasoning?: string;
  toolCalls?: ChatToolCall[];
  /** Ties a tool result back to the call that asked for it. */
  toolCallId?: string;
  finishReason?: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const nonEmptyString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value : undefined;

/** Content is a string, `null` for a tool-call-only turn, or an array of typed parts. */
const contentText = (content: unknown): string => {
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return '';
  }
  return content
    .map((part) => {
      if (typeof part === 'string') {
        return part;
      }
      if (!isRecord(part)) {
        return '';
      }
      // Image and audio parts carry no text; name the kind so the turn is not blank.
      return typeof part.text === 'string'
        ? part.text
        : typeof part.type === 'string'
          ? `[${part.type}]`
          : '';
    })
    .filter(Boolean)
    .join('\n\n');
};

const toToolCalls = (value: unknown): ChatToolCall[] | undefined => {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const calls = value.flatMap((call) => {
    if (!isRecord(call)) {
      return [];
    }
    // The OpenAI shape nests name/arguments under `function`; some gateways flatten them.
    const fn = isRecord(call.function) ? call.function : call;
    const name = nonEmptyString(fn.name);
    if (!name) {
      return [];
    }
    const args = fn.arguments;
    return [
      {
        id: nonEmptyString(call.id),
        name,
        arguments:
          typeof args === 'string'
            ? args
            : args === undefined || args === null
              ? ''
              : JSON.stringify(args, null, 2),
      },
    ];
  });
  return calls.length ? calls : undefined;
};

const buildMessage = (value: Record<string, unknown>, role: string): ChatMessage => ({
  role,
  content: contentText(value.content),
  reasoning: nonEmptyString(value.reasoning_content) ?? nonEmptyString(value.reasoning),
  toolCalls: toToolCalls(value.tool_calls),
  toolCallId: nonEmptyString(value.tool_call_id),
});

/** A request-side message, which always names its own role. */
const toMessage = (value: unknown): ChatMessage | null => {
  if (!isRecord(value)) {
    return null;
  }
  const role = nonEmptyString(value.role);
  return role ? buildMessage(value, role) : null;
};

/**
 * A response-side message, where the role is implied by position. Inferring it
 * would match almost any object, so this one has to carry something readable.
 */
const toResponseMessage = (value: unknown, finishReason?: string): ChatMessage | null => {
  if (!isRecord(value)) {
    return null;
  }
  const message = buildMessage(value, nonEmptyString(value.role) ?? 'assistant');
  if (!message.content && !message.reasoning && !message.toolCalls) {
    return null;
  }
  return { ...message, finishReason };
};

/** A transcript only, so one stray element rejects the whole array. */
const toTranscript = (value: unknown): ChatMessage[] | null => {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  const messages = value.map(toMessage);
  return messages.every((message) => message !== null) ? (messages as ChatMessage[]) : null;
};

const requestMessages = (root: Record<string, unknown>): ChatMessage[] | null => {
  // Some observers record the whole HTTP body, putting the request under `content`.
  const body = isRecord(root.content) ? root.content : root;
  return toTranscript(body.messages);
};

const responseMessages = (root: Record<string, unknown>): ChatMessage[] | null => {
  if (Array.isArray(root.choices)) {
    const choices = root.choices.flatMap((choice) => {
      if (!isRecord(choice)) {
        return [];
      }
      // `delta` is the streaming spelling of `message`.
      const message = toResponseMessage(
        choice.message ?? choice.delta,
        nonEmptyString(choice.finish_reason)
      );
      return message ? [message] : [];
    });
    if (choices.length) {
      return choices;
    }
  }
  const single = toResponseMessage(
    root.assistant_message ?? root.message,
    nonEmptyString(root.finish_reason)
  );
  return single ? [single] : null;
};

/**
 * The chat turns in an OpenAI-compatible payload, or `null` when it is not one.
 *
 * Covers a bare transcript, a request (`messages`, optionally wrapped in the
 * recorded body), and a response (`choices[].message`, `assistant_message`). A
 * payload holding both — a recorded request/response pair — reads as one
 * conversation ending in the reply.
 */
export const toChatMessages = (parsed: unknown): ChatMessage[] | null => {
  const transcript = toTranscript(parsed);
  if (transcript) {
    return transcript;
  }
  if (!isRecord(parsed)) {
    return null;
  }
  const request = requestMessages(parsed);
  const response = responseMessages(parsed);
  if (request && response) {
    return [...request, ...response];
  }
  return request ?? response;
};
