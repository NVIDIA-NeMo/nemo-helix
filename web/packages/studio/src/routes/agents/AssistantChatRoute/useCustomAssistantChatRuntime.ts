// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type AppendMessage,
  type MessageStatus,
  type ThreadAssistantMessagePart,
  type ThreadMessageLike,
  useExternalStoreRuntime,
} from '@assistant-ui/react';
import {
  CANCELLED_STATUS,
  COMPLETE_STATUS,
  RUNNING_STATUS,
} from '@nemo/common/src/components/AssistantChat/constants';
import {
  appendMessageToThreadMessage,
  createTextMessage,
  getMessageText,
} from '@nemo/common/src/components/AssistantChat/messageUtils';
import {
  createAssistantThinkingPart,
  getAssistantCompletedMessageParts,
  getAssistantThinkingPartText,
  groupConsecutiveAssistantSubtleToolCalls,
  isAssistantThinkingPart,
} from '@studio/routes/agents/AssistantChatRoute/toolParts';
import { useCallback, useRef, useState } from 'react';

export interface CustomAssistantRunContext {
  prompt: string;
  signal: AbortSignal;
  appendAssistantText: (text: string) => void;
  appendAssistantParts: (parts: readonly ThreadAssistantMessagePart[]) => void;
  prepareForUserInput: () => void;
  setAssistantText: (text: string) => void;
  isCurrentRun: () => boolean;
}

export interface CustomAssistantRunResult {
  content?: readonly ThreadAssistantMessagePart[];
  status?: MessageStatus;
  text?: string;
}

export type CustomAssistantBeforeRunResult = 'continue' | 'cancel';

export interface CustomAssistantBeforeRunContext {
  prompt: string;
  signal: AbortSignal;
  prepareForUserInput: () => void;
  isCurrentRun: () => boolean;
}

interface UseCustomAssistantChatRuntimeOptions {
  initialMessages?: readonly ThreadMessageLike[];
  onBeforeRun?: (
    context: CustomAssistantBeforeRunContext
  ) => Promise<CustomAssistantBeforeRunResult | void> | CustomAssistantBeforeRunResult | void;
  onRun: (context: CustomAssistantRunContext) => Promise<CustomAssistantRunResult | void>;
  onError?: (error: Error) => void;
}

interface CompleteAssistantMessageOptions {
  readonly collapseAssistantContent?: boolean;
}

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'AbortError';

const convertMessage = (message: ThreadMessageLike): ThreadMessageLike => message;

const mergeAssistantParts = (
  currentParts: readonly ThreadAssistantMessagePart[],
  nextParts: readonly ThreadAssistantMessagePart[]
): readonly ThreadAssistantMessagePart[] => {
  const merged = [...currentParts];

  for (const part of nextParts) {
    if (isAssistantThinkingPart(part)) {
      const previousThinkingPart = merged.at(-1);
      if (previousThinkingPart && isAssistantThinkingPart(previousThinkingPart)) {
        merged[merged.length - 1] = createAssistantThinkingPart(
          [getAssistantThinkingPartText(previousThinkingPart), getAssistantThinkingPartText(part)]
            .filter(Boolean)
            .join('\n\n'),
          previousThinkingPart.toolCallId
        );
        continue;
      }

      merged.push(part);
      continue;
    }

    if (part.type !== 'text') {
      merged.push(part);
      continue;
    }

    const previousPart = merged.at(-1);
    if (previousPart?.type === part.type) {
      merged[merged.length - 1] = {
        ...previousPart,
        text: previousPart.text + part.text,
      };
      continue;
    }

    merged.push(part);
  }

  return groupConsecutiveAssistantSubtleToolCalls(merged);
};

const getAssistantPartsText = (parts: readonly ThreadAssistantMessagePart[]): string =>
  parts
    .map((part) => {
      if (part.type === 'text') return part.text;
      return '';
    })
    .join('');

const getAssistantMessageParts = (
  message: ThreadMessageLike
): readonly ThreadAssistantMessagePart[] => {
  if (typeof message.content === 'string') return [{ type: 'text', text: message.content }];
  return message.content as readonly ThreadAssistantMessagePart[];
};

const hasVisibleAssistantContent = (content: ThreadMessageLike['content']): boolean => {
  if (typeof content === 'string') return content.trim().length > 0;

  return content.some((part) => {
    if (part.type === 'text') return part.text.trim().length > 0;
    return true;
  });
};

const completeAssistantContent = (
  content: ThreadMessageLike['content'],
  status: MessageStatus,
  elapsedMs?: number
): ThreadMessageLike['content'] => {
  if (status.type !== 'complete' || !Array.isArray(content)) return content;
  return getAssistantCompletedMessageParts(content, { elapsedMs });
};

export const useCustomAssistantChatRuntime = ({
  initialMessages = [],
  onBeforeRun,
  onRun,
  onError,
}: UseCustomAssistantChatRuntimeOptions) => {
  const [messages, setMessages] = useState<readonly ThreadMessageLike[]>(initialMessages);
  const [isRunning, setIsRunning] = useState(false);
  const messagesRef = useRef<readonly ThreadMessageLike[]>(initialMessages);
  const abortControllerRef = useRef<AbortController | null>(null);

  const setThreadMessages = useCallback((nextMessages: readonly ThreadMessageLike[]) => {
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
  }, []);

  const updateAssistantMessageContent = useCallback(
    (assistantMessageId: string, content: ThreadMessageLike['content'], status: MessageStatus) => {
      setThreadMessages(
        messagesRef.current.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                content,
                status,
              }
            : message
        )
      );
    },
    [setThreadMessages]
  );

  const updateAssistantMessageText = useCallback(
    (assistantMessageId: string, text: string, status: MessageStatus) => {
      updateAssistantMessageContent(assistantMessageId, [{ type: 'text', text }], status);
    },
    [updateAssistantMessageContent]
  );

  const completeAssistantMessageContent = useCallback(
    (assistantMessageId: string, content: ThreadMessageLike['content'], status: MessageStatus) => {
      if (!hasVisibleAssistantContent(content)) {
        setThreadMessages(
          messagesRef.current.filter((message) => message.id !== assistantMessageId)
        );
        return;
      }

      updateAssistantMessageContent(assistantMessageId, content, status);
    },
    [setThreadMessages, updateAssistantMessageContent]
  );

  const runCompletion = useCallback(
    async (conversationMessages: readonly ThreadMessageLike[]) => {
      const latestMessage = conversationMessages.at(-1);
      const prompt = latestMessage ? getMessageText(latestMessage).trim() : '';
      if (!prompt) return;

      abortControllerRef.current?.abort();
      const runController = new AbortController();
      abortControllerRef.current = runController;
      const isCurrentRun = () => abortControllerRef.current === runController;

      let assistantMessageId: string | null = null;
      let responseText = '';
      let responseContent: readonly ThreadAssistantMessagePart[] | undefined;
      let runStartedAt = Date.now();
      let runPausedAt: number | undefined;

      const resumeRunTimer = () => {
        if (runPausedAt === undefined) return;
        runStartedAt += Date.now() - runPausedAt;
        runPausedAt = undefined;
      };

      const createAssistantMessage = () => {
        resumeRunTimer();
        const assistantMessage = createTextMessage('assistant', '', RUNNING_STATUS);
        assistantMessageId = assistantMessage.id!;
        responseText = '';
        responseContent = undefined;
        setThreadMessages([...messagesRef.current, assistantMessage]);
      };

      const resumeLastAssistantMessage = (): boolean => {
        const lastMessage = messagesRef.current.at(-1);
        if (
          !lastMessage?.id ||
          lastMessage.role !== 'assistant' ||
          lastMessage.status?.type !== 'complete'
        ) {
          return false;
        }

        resumeRunTimer();
        assistantMessageId = lastMessage.id;
        responseContent = getAssistantMessageParts(lastMessage);
        responseText = getAssistantPartsText(responseContent);
        updateAssistantMessageContent(assistantMessageId, responseContent, RUNNING_STATUS);
        return true;
      };

      const ensureAssistantMessage = () => {
        if (assistantMessageId) {
          if (messagesRef.current.at(-1)?.id === assistantMessageId) return;
          // Answering a blocking picker appends a user message mid-run. The active
          // assistant message now sits above it, so continuing to write into it would
          // render later tool calls before the answer that triggered them. Close it
          // and open a new one underneath instead.
          completeActiveAssistantMessage(COMPLETE_STATUS, getCurrentResponseContent(), {
            collapseAssistantContent: false,
          });
        }
        if (resumeLastAssistantMessage()) return;
        createAssistantMessage();
      };

      const getCurrentResponseContent = (): readonly ThreadAssistantMessagePart[] =>
        responseContent ?? [{ type: 'text', text: responseText }];

      const completeActiveAssistantMessage = (
        status: MessageStatus,
        content: ThreadMessageLike['content'] = getCurrentResponseContent(),
        options: CompleteAssistantMessageOptions = {}
      ) => {
        if (!assistantMessageId) return;

        const currentAssistantMessageId = assistantMessageId;
        assistantMessageId = null;
        responseText = '';
        responseContent = undefined;
        completeAssistantMessageContent(
          currentAssistantMessageId,
          options.collapseAssistantContent === false
            ? content
            : completeAssistantContent(content, status, Date.now() - runStartedAt),
          status
        );
      };

      const setAssistantText = (text: string) => {
        ensureAssistantMessage();
        if (responseContent) {
          const textToAppend = text.startsWith(responseText)
            ? text.slice(responseText.length)
            : text;
          responseContent = textToAppend
            ? mergeAssistantParts(responseContent, [{ type: 'text', text: textToAppend }])
            : responseContent;
          responseText = getAssistantPartsText(responseContent);
          updateAssistantMessageContent(assistantMessageId!, responseContent, RUNNING_STATUS);
          return;
        }

        responseText = text;
        responseContent = undefined;
        updateAssistantMessageText(assistantMessageId!, responseText, RUNNING_STATUS);
      };

      const appendAssistantText = (text: string) => {
        ensureAssistantMessage();
        if (responseContent) {
          responseContent = mergeAssistantParts(responseContent, [{ type: 'text', text }]);
          responseText = getAssistantPartsText(responseContent);
          updateAssistantMessageContent(assistantMessageId!, responseContent, RUNNING_STATUS);
          return;
        }

        responseText += text;
        responseContent = undefined;
        updateAssistantMessageText(assistantMessageId!, responseText, RUNNING_STATUS);
      };

      const appendAssistantParts = (parts: readonly ThreadAssistantMessagePart[]) => {
        if (!parts.length) return;

        ensureAssistantMessage();
        responseContent = mergeAssistantParts(responseContent ?? [], parts);
        responseText = getAssistantPartsText(responseContent);
        updateAssistantMessageContent(assistantMessageId!, responseContent, RUNNING_STATUS);
      };

      const prepareForUserInput = () => {
        completeActiveAssistantMessage(COMPLETE_STATUS, getCurrentResponseContent(), {
          collapseAssistantContent: false,
        });
        runPausedAt ??= Date.now();
      };

      try {
        const beforeRunResult = await onBeforeRun?.({
          prompt,
          signal: runController.signal,
          prepareForUserInput,
          isCurrentRun,
        });

        if (beforeRunResult === 'cancel' || runController.signal.aborted || !isCurrentRun()) {
          return;
        }

        createAssistantMessage();
        setIsRunning(true);

        const result = await onRun({
          prompt,
          signal: runController.signal,
          appendAssistantText,
          appendAssistantParts,
          prepareForUserInput,
          setAssistantText,
          isCurrentRun,
        });

        if (runController.signal.aborted || !isCurrentRun()) {
          completeActiveAssistantMessage(CANCELLED_STATUS);
          return;
        }

        if (result?.content !== undefined || result?.text !== undefined) {
          // Answering a picker completes the active message, so there may be none to
          // write into and the returned result would be dropped. The error path below
          // calls this for the same reason.
          ensureAssistantMessage();
        }
        completeActiveAssistantMessage(
          result?.status ?? COMPLETE_STATUS,
          result?.content ??
            (result?.text !== undefined
              ? [{ type: 'text', text: result.text }]
              : getCurrentResponseContent())
        );
      } catch (error: unknown) {
        if (runController.signal.aborted || isAbortError(error)) {
          completeActiveAssistantMessage(CANCELLED_STATUS);
          return;
        }

        const errorMessage = error instanceof Error ? error.message : 'Unknown Error';
        ensureAssistantMessage();
        const currentResponseContent = getCurrentResponseContent();
        const failureMessage = `NeMo Assistant stopped before completing: ${errorMessage}`;
        const incompleteContent = mergeAssistantParts(currentResponseContent, [
          {
            type: 'text',
            text: `${hasVisibleAssistantContent(currentResponseContent) ? '\n\n' : ''}${failureMessage}`,
          },
        ]);
        completeActiveAssistantMessage(
          {
            type: 'incomplete',
            reason: 'error',
            error: errorMessage,
          },
          incompleteContent,
          { collapseAssistantContent: false }
        );
        onError?.(error instanceof Error ? error : new Error(errorMessage));
      } finally {
        if (abortControllerRef.current === runController) {
          abortControllerRef.current = null;
          setIsRunning(false);
        }
      }
    },
    [
      completeAssistantMessageContent,
      onBeforeRun,
      onError,
      onRun,
      setThreadMessages,
      updateAssistantMessageContent,
      updateAssistantMessageText,
    ]
  );

  const handleNewMessage = useCallback(
    async (message: AppendMessage): Promise<void> => {
      const text = getMessageText(message).trim();
      if (!text) return;

      const userMessage = appendMessageToThreadMessage({
        ...message,
        content: [{ type: 'text', text }],
      });
      const nextMessages = [...messagesRef.current, userMessage];
      setThreadMessages(nextMessages);
      await runCompletion(nextMessages);
    },
    [runCompletion, setThreadMessages]
  );

  const appendUserMessage = useCallback(
    (message: string) => {
      const text = message.trim();
      if (!text) return;

      const userMessage = createTextMessage('user', text);
      setThreadMessages([...messagesRef.current, userMessage]);
    },
    [setThreadMessages]
  );

  const submitPrompt = useCallback(
    async (prompt: string): Promise<void> => {
      const text = prompt.trim();
      if (!text) return;

      const userMessage = createTextMessage('user', text);
      const nextMessages = [...messagesRef.current, userMessage];
      setThreadMessages(nextMessages);
      await runCompletion(nextMessages);
    },
    [runCompletion, setThreadMessages]
  );

  const handleCancel = useCallback(async () => {
    abortControllerRef.current?.abort();
    setIsRunning(false);
    setThreadMessages(
      messagesRef.current.map((message) =>
        message.role === 'assistant' && message.status?.type === 'running'
          ? { ...message, status: CANCELLED_STATUS }
          : message
      )
    );
  }, [setThreadMessages]);

  const handleReset = useCallback(() => {
    abortControllerRef.current?.abort();
    setIsRunning(false);
    setThreadMessages([]);
  }, [setThreadMessages]);

  const replaceMessages = useCallback(
    (nextMessages: readonly ThreadMessageLike[]) => {
      abortControllerRef.current?.abort();
      setIsRunning(false);
      setThreadMessages(nextMessages);
    },
    [setThreadMessages]
  );

  const runtime = useExternalStoreRuntime<ThreadMessageLike>({
    messages,
    setMessages: setThreadMessages,
    isRunning,
    onNew: handleNewMessage,
    onCancel: handleCancel,
    convertMessage,
    unstable_capabilities: {
      copy: true,
    },
  });

  return {
    appendUserMessage,
    handleReset,
    isRunning,
    messages,
    replaceMessages,
    runtime,
    submitPrompt,
  };
};
