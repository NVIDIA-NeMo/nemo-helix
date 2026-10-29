// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AppendMessage, ThreadMessageLike } from '@assistant-ui/react';
import { describeChatError } from '@nemo/common/src/components/AssistantChat/describeChatError';
import {
  getFailedReply,
  getOpenAIMessages,
  getUserMessageContent,
  modelSupportsImageAttachments,
} from '@nemo/common/src/components/AssistantChat/messageUtils';

const createMessage = (role: ThreadMessageLike['role'], text: string): ThreadMessageLike => ({
  id: `${role}-${text}`,
  role,
  content: [{ type: 'text', text }],
});

const DATA_URL = 'data:image/png;base64,AAAA';

describe('getOpenAIMessages', () => {
  it('filters empty content for all OpenAI message roles', () => {
    const messages = [
      createMessage('system', ''),
      createMessage('user', ''),
      createMessage('assistant', ''),
      createMessage('user', 'Hello'),
    ];

    expect(getOpenAIMessages(messages)).toEqual([{ role: 'user', content: 'Hello' }]);
  });

  it('replaces existing system messages when a system prompt is provided', () => {
    const messages = [
      createMessage('system', 'Original system prompt'),
      createMessage('user', 'Hello'),
    ];

    expect(getOpenAIMessages(messages, 'Replacement system prompt')).toEqual([
      { role: 'system', content: 'Replacement system prompt' },
      { role: 'user', content: 'Hello' },
    ]);
  });

  it('forwards user images as OpenAI multimodal content alongside the text', () => {
    const messages: ThreadMessageLike[] = [
      {
        id: 'user-with-image',
        role: 'user',
        content: [
          { type: 'text', text: 'What is in this image?' },
          { type: 'image', image: DATA_URL },
        ],
      },
    ];

    expect(getOpenAIMessages(messages)).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is in this image?' },
          { type: 'image_url', image_url: { url: DATA_URL } },
        ],
      },
    ]);
  });

  it('sends an image-only user message without a text part', () => {
    const messages: ThreadMessageLike[] = [
      { id: 'image-only', role: 'user', content: [{ type: 'image', image: DATA_URL }] },
    ];

    expect(getOpenAIMessages(messages)).toEqual([
      { role: 'user', content: [{ type: 'image_url', image_url: { url: DATA_URL } }] },
    ]);
  });
});

describe('getUserMessageContent', () => {
  it('combines composed text with image parts from attachments', () => {
    const message = {
      role: 'user',
      content: [{ type: 'text', text: '  Describe this  ' }],
      attachments: [
        {
          id: 'a1',
          type: 'image',
          name: 'pic.png',
          status: { type: 'complete' },
          content: [{ type: 'image', image: DATA_URL }],
        },
      ],
    } as unknown as AppendMessage;

    expect(getUserMessageContent(message)).toEqual([
      { type: 'text', text: 'Describe this' },
      { type: 'image', image: DATA_URL },
    ]);
  });

  it('returns an empty array when there is neither text nor an attachment', () => {
    const message = {
      role: 'user',
      content: [{ type: 'text', text: '   ' }],
    } as unknown as AppendMessage;

    expect(getUserMessageContent(message)).toEqual([]);
  });
});

describe('modelSupportsImageAttachments', () => {
  it('matches the vision substring case-insensitively', () => {
    expect(modelSupportsImageAttachments('meta/llama-3.2-11b-vision-instruct')).toBe(true);
    expect(modelSupportsImageAttachments('Some-Vision-Model')).toBe(true);
  });

  it('rejects models without the vision substring or with no name', () => {
    expect(modelSupportsImageAttachments('meta/llama-3.1-8b-instruct')).toBe(false);
    expect(modelSupportsImageAttachments('')).toBe(false);
    expect(modelSupportsImageAttachments(undefined)).toBe(false);
  });
});

describe('getFailedReply', () => {
  it('keeps the raw error in the status and puts the explanation in the body', () => {
    const raw =
      "RuntimeError: internal error: APIStatusError: Error code: 429 - {'detail': 'slow down'}";

    const { parts, status } = getFailedReply(raw, describeChatError);

    expect(status).toEqual({ type: 'incomplete', reason: 'error', error: raw });
    expect(parts).toHaveLength(1);
    expect(parts[0].text).toMatch(/rate limiting/i);
  });

  it('puts the generic line in the body when nothing describes the error', () => {
    const { parts, status } = getFailedReply('Something nobody mapped');

    expect(parts[0].text).toBe('There was an error generating a response.');
    expect(status).toMatchObject({ error: 'Something nobody mapped' });
  });
});
