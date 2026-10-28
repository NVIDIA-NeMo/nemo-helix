// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AppendMessage,
  ImageMessagePart,
  MessageStatus,
  TextMessagePart,
  ThreadMessageLike,
} from '@assistant-ui/react';
import {
  humanizeChatError,
  type DescribeChatError,
} from '@nemo/common/src/components/AssistantChat/describeChatError';
import type {
  ChatCompletionContentPart,
  ChatCompletionMessageParam,
} from 'openai/resources/index.mjs';

/** Content parts a user message can carry into the runtime: text plus images. */
export type UserMessageContentPart = TextMessagePart | ImageMessagePart;

const supportsImagesSubstrings = ['vision', 'image', 'images'];
export const modelSupportsImageAttachments = (model?: string): boolean =>
  !!model && supportsImagesSubstrings.some((substring) => model.toLowerCase().includes(substring));

const createMessageId = (): string => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `assistant-chat-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
};

export const createTextMessage = (
  role: ThreadMessageLike['role'],
  text: string,
  status?: MessageStatus
): ThreadMessageLike => ({
  id: createMessageId(),
  role,
  content: [{ type: 'text', text }],
  status,
});

/**
 * The content and status of an assistant reply whose request failed: the explanation as the body
 * (see `humanizeChatError`) and the raw error in the status, where the error banner and the copy
 * action read it.
 */
export const getFailedReply = (
  errorMessage: string,
  describe?: DescribeChatError
): { parts: TextMessagePart[]; status: MessageStatus } => ({
  parts: [{ type: 'text', text: humanizeChatError(errorMessage, describe) }],
  status: { type: 'incomplete', reason: 'error', error: errorMessage },
});

const isTextPart = (part: unknown): part is TextMessagePart => {
  if (typeof part !== 'object' || part === null) return false;
  return 'type' in part && part.type === 'text' && 'text' in part && typeof part.text === 'string';
};

const isImagePart = (part: unknown): part is ImageMessagePart => {
  if (typeof part !== 'object' || part === null) return false;
  return (
    'type' in part &&
    part.type === 'image' &&
    'image' in part &&
    typeof (part as ImageMessagePart).image === 'string'
  );
};

export const getMessageText = (message: Pick<ThreadMessageLike, 'content'>): string => {
  if (typeof message.content === 'string') return message.content;
  return message.content
    .filter(isTextPart)
    .map((part) => part.text)
    .join('\n');
};

const collectImageParts = (parts: readonly unknown[]): ImageMessagePart[] =>
  parts.filter(isImagePart);

/**
 * Collects image parts from a message — both inline content parts and any
 * composer attachments. The `SimpleImageAttachmentAdapter` emits an image part
 * (a data URL) in each attachment's content on send, so the two sources are
 * unified here.
 */
export const getMessageImageParts = (message: {
  content: ThreadMessageLike['content'];
  attachments?: readonly { readonly content: readonly unknown[] }[];
}): ImageMessagePart[] => {
  const fromContent = Array.isArray(message.content) ? collectImageParts(message.content) : [];
  const fromAttachments = (message.attachments ?? []).flatMap((attachment) =>
    collectImageParts(attachment.content)
  );
  return [...fromContent, ...fromAttachments];
};

/**
 * Builds the stored thread-message content for a freshly composed user message:
 * the trimmed text (when present) followed by any image parts pulled from
 * composer attachments.
 */
export const getUserMessageContent = (message: AppendMessage): UserMessageContentPart[] => {
  const content: UserMessageContentPart[] = [];
  const text = getMessageText(message).trim();
  if (text) content.push({ type: 'text', text });
  content.push(...getMessageImageParts(message));
  return content;
};

export const appendMessageToThreadMessage = (message: AppendMessage): ThreadMessageLike => ({
  id: createMessageId(),
  role: message.role,
  content: message.content,
});

const getOpenAIMessage = (message: ThreadMessageLike): ChatCompletionMessageParam | undefined => {
  if (!['assistant', 'system', 'user'].includes(message.role)) return undefined;

  const text = getMessageText(message);

  if (message.role === 'user') {
    const imageParts = getMessageImageParts(message);
    if (imageParts.length > 0) {
      const content: ChatCompletionContentPart[] = [];
      if (text) content.push({ type: 'text', text });
      for (const part of imageParts) {
        content.push({ type: 'image_url', image_url: { url: part.image } });
      }
      return { role: 'user', content };
    }
  }

  if (!text) return undefined;

  return {
    role: message.role,
    content: text,
  } as ChatCompletionMessageParam;
};

export const getOpenAIMessages = (
  messages: readonly ThreadMessageLike[],
  systemPrompt?: string
): ChatCompletionMessageParam[] => {
  const openAIMessages = messages
    .map(getOpenAIMessage)
    .filter((message): message is ChatCompletionMessageParam => message !== undefined);

  if (!systemPrompt) return openAIMessages;

  const withoutSystem = openAIMessages.filter((message) => message.role !== 'system');
  return [{ role: 'system', content: systemPrompt }, ...withoutSystem];
};

const getMessageIndex = (
  messages: readonly ThreadMessageLike[],
  messageId: string | null | undefined
): number => {
  if (!messageId) return -1;

  const explicitIndex = messages.findIndex((message) => message.id === messageId);
  if (explicitIndex !== -1) return explicitIndex;

  const fallbackIndex = Number(messageId);
  return Number.isInteger(fallbackIndex) &&
    fallbackIndex >= 0 &&
    fallbackIndex < messages.length &&
    String(fallbackIndex) === messageId
    ? fallbackIndex
    : -1;
};

export const getEditedMessageIndex = (
  messages: readonly ThreadMessageLike[],
  message: AppendMessage
): number => {
  const sourceIndex = getMessageIndex(messages, message.sourceId);
  if (sourceIndex !== -1) return sourceIndex;

  if (message.parentId === null) return 0;

  const parentIndex = getMessageIndex(messages, message.parentId);
  return parentIndex === -1 ? -1 : parentIndex + 1;
};
