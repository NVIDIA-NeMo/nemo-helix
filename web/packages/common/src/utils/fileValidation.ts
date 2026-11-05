// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileFormatType, InputFileSchemaType } from '@nemo/common/src/types';
import { findMessagesArray, getFirstRow } from '@nemo/common/src/utils/file';
import { parseJsonOrJsonl } from '@nemo/common/src/utils/parseRecords';
import { ChatCompletionRole } from 'openai/resources/index.mjs';

export interface FileValidationResult {
  isValid: boolean;
  format: FileFormatType | null;
  error?: string;
}

export interface PromptCompletionDetectionResult {
  schemaType: InputFileSchemaType.COMPLETION;
  detectedFields: {
    prompt?: string;
    completion?: string;
  };
  isComplete: boolean;
  firstRow: Record<string, unknown>;
}

export interface MessagesDetectionResult {
  schemaType: InputFileSchemaType.CHAT_COMPLETION;
  detectedMessages: {
    user?: {
      index: number;
      selector: string;
    };
    assistant?: {
      index: number;
      selector: string;
    };
    system?: {
      index: number;
      selector: string;
    };
  };
  messagesKey: string;
  isComplete: boolean;
  firstRow: Record<string, unknown>;
}

export interface UnknownSchemaDetectionResult {
  schemaType: null;
  firstRow: Record<string, unknown>;
}

export type FileFormatDetectionResult =
  | MessagesDetectionResult
  | PromptCompletionDetectionResult
  | UnknownSchemaDetectionResult
  | null;

/**
 * Finds the first message in an array that matches the given role.
 * For user and assistant roles, also requires non-empty string content.
 * @param messagesArray The array of messages to search
 * @param role The role to search for
 * @returns An object with index and content if found, null otherwise
 */
export function findMessageByRole(
  messagesArray: Array<{ role?: string; content?: string }>,
  role: ChatCompletionRole
): { index: number; content: string } | null {
  for (let i = 0; i < messagesArray.length; i++) {
    const message = messagesArray[i];

    // Skip non-objects
    if (typeof message !== 'object' || message === null) {
      continue;
    }

    // Check if role matches
    if (message.role !== role) {
      continue;
    }

    // For user and assistant, require non-empty string content
    if (role === 'user' || role === 'assistant') {
      if (typeof message.content === 'string' && message.content.length > 0) {
        return { index: i, content: message.content };
      }
    } else {
      // For system and other roles, just return the first match
      return { index: i, content: message.content || '' };
    }
  }

  return null;
}

export async function validateFileFormat(file: File): Promise<FileValidationResult> {
  try {
    const text = await file.text();

    let items: unknown[];
    let format: FileFormatType;
    try {
      ({ records: items, format } = parseJsonOrJsonl(text));
    } catch {
      return { isValid: false, format: null, error: 'File is not valid JSON or JSONL' };
    }

    if (items.length === 0) {
      return { isValid: false, format: null, error: 'File contains no data' };
    }

    return { isValid: true, format };
  } catch (error) {
    return {
      isValid: false,
      format: null,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

export async function detectFileStructure(
  file: File,
  format: FileFormatType,
  targetMode?: 'online' | 'offline'
): Promise<FileFormatDetectionResult | null> {
  try {
    const firstItem = await getFirstRow(file, format);

    if (!firstItem) {
      return null;
    }
    // Check for messages format - search through all keys to find a messages array
    const messagesArrayResult = findMessagesArray(firstItem);

    if (messagesArrayResult) {
      const { key: messagesKey, value: messagesArray } = messagesArrayResult;
      const userResult = findMessageByRole(messagesArray, 'user');
      const assistantResult = findMessageByRole(messagesArray, 'assistant');
      const systemResult = findMessageByRole(messagesArray, 'system');

      const detectedMessages: MessagesDetectionResult['detectedMessages'] = {};

      if (userResult) {
        detectedMessages.user = {
          index: userResult.index,
          selector: `${messagesKey}[${userResult.index}].content`,
        };
      }

      if (assistantResult) {
        detectedMessages.assistant = {
          index: assistantResult.index,
          selector: `${messagesKey}[${assistantResult.index}].content`,
        };
      }

      if (systemResult) {
        detectedMessages.system = {
          index: systemResult.index,
          selector: `${messagesKey}[${systemResult.index}].content`,
        };
      }

      // Determine if the schema is complete
      // For online mode: both user and assistant present
      // For offline mode: user and assistant present, but never complete (need cached output key)
      const hasRequiredMessages = userResult !== null && assistantResult !== null;
      const isComplete = targetMode === 'offline' ? false : hasRequiredMessages;

      return {
        schemaType: InputFileSchemaType.CHAT_COMPLETION,
        detectedMessages,
        messagesKey,
        isComplete,
        firstRow: firstItem,
      };
    }

    // Check for prompt-completion format
    const promptKeys = ['prompt', 'question'];
    const completionKeys = ['completion', 'ideal_response', 'response', 'output', 'answer'];

    const promptKey = promptKeys.find((key) => firstItem[key] !== undefined);
    const completionKey = completionKeys.find((key) => firstItem[key] !== undefined);

    if (promptKey || completionKey) {
      // For online mode: both prompt and completion present
      // For offline mode: prompt and completion present, but never complete (need cached output key)
      const hasRequiredFields = !!(promptKey && completionKey);
      const isComplete = targetMode === 'offline' ? false : hasRequiredFields;

      return {
        schemaType: InputFileSchemaType.COMPLETION,
        detectedFields: {
          ...(promptKey && { prompt: promptKey }),
          ...(completionKey && { completion: completionKey }),
        },
        isComplete,
        firstRow: firstItem,
      };
    }

    // No schema detected but we have valid data - return unknown schema with firstRow
    return {
      schemaType: null,
      firstRow: firstItem,
    };
  } catch {
    // Parse error or other error - return null
    return null;
  }
}
