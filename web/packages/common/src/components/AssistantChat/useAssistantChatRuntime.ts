// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type AppendMessage,
  type MessageStatus,
  type ThreadAssistantMessagePart,
  type ThreadMessageLike,
  SimpleImageAttachmentAdapter,
  useExternalStoreRuntime,
} from '@assistant-ui/react';
import {
  computeCompletionMetrics,
  getReasoningDelta,
  getUsageCompletionTokens,
} from '@nemo/common/src/components/AssistantChat/completionMetrics';
import {
  appendCompletionDeltaParts,
  getWorkDetailsMessageParts,
} from '@nemo/common/src/components/AssistantChat/completionSections';
import {
  getCompletionImages,
  getCompletionText,
  isAbortError,
  isChatCompletionStream,
} from '@nemo/common/src/components/AssistantChat/completionUtils';
import {
  CANCELLED_STATUS,
  COMPLETE_STATUS,
  RUNNING_STATUS,
} from '@nemo/common/src/components/AssistantChat/constants';
import {
  appendMessageToThreadMessage,
  createTextMessage,
  getEditedMessageIndex,
  getOpenAIMessages,
  getUserMessageContent,
} from '@nemo/common/src/components/AssistantChat/messageUtils';
import { groupConsecutiveAssistantSubtleToolCalls } from '@nemo/common/src/components/AssistantChat/toolParts';
import type { AssistantChatProps } from '@nemo/common/src/components/AssistantChat/types';
import { useChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import { useCallback, useEffect, useRef, useState } from 'react';

const imageAttachmentAdapter = new SimpleImageAttachmentAdapter();

type ImagePart = { type: 'image'; image: string };

const getMessageContent = (
  parts: readonly ThreadAssistantMessagePart[],
  images: readonly ImagePart[],
  status: MessageStatus
): ThreadMessageLike['content'] => {
  const content =
    status.type === 'complete'
      ? getWorkDetailsMessageParts(parts)
      : groupConsecutiveAssistantSubtleToolCalls(parts);
  return [...(content.length ? content : [{ type: 'text' as const, text: '' }]), ...images];
};

type UseAssistantChatRuntimeOptions = Pick<
  AssistantChatProps,
  | 'baseURL'
  | 'broadcast'
  | 'stopCount'
  | 'disabled'
  | 'initialMessages'
  | 'model'
  | 'onError'
  | 'onMessageComplete'
  | 'onRunningChange'
  | 'onEmptyChange'
  | 'promptData'
  | 'tools'
  | 'workspace'
  | 'enableImageAttachments'
>;

export const useAssistantChatRuntime = ({
  model,
  workspace,
  baseURL,
  promptData,
  tools,
  disabled = false,
  initialMessages = [],
  onError,
  onMessageComplete,
  onRunningChange,
  onEmptyChange,
  broadcast,
  stopCount,
  enableImageAttachments = true,
}: UseAssistantChatRuntimeOptions) => {
  const [messages, setMessages] = useState<readonly ThreadMessageLike[]>(initialMessages);
  const [isRunning, setIsRunning] = useState(false);
  const messagesRef = useRef<readonly ThreadMessageLike[]>(initialMessages);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { mutateAsync: createChatCompletion } = useChatCompletion();

  const setThreadMessages = useCallback((nextMessages: readonly ThreadMessageLike[]) => {
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
  }, []);

  const updateAssistantMessage = useCallback(
    (
      assistantMessageId: string,
      parts: readonly ThreadAssistantMessagePart[],
      images: readonly ImagePart[],
      status: MessageStatus
    ) => {
      setThreadMessages(
        messagesRef.current.map((message) =>
          message.id === assistantMessageId
            ? { ...message, content: getMessageContent(parts, images, status), status }
            : message
        )
      );
    },
    [setThreadMessages]
  );

  const runCompletion = useCallback(
    async (conversationMessages: readonly ThreadMessageLike[]) => {
      if (disabled) return;

      abortControllerRef.current?.abort();
      const runController = new AbortController();
      abortControllerRef.current = runController;
      const isCurrentRun = () => abortControllerRef.current === runController;

      const assistantMessage = createTextMessage('assistant', '', RUNNING_STATUS);
      setThreadMessages([...conversationMessages, assistantMessage]);
      setIsRunning(true);
      let responseText = '';
      let responseParts: readonly ThreadAssistantMessagePart[] = [];
      const responseImages: ImagePart[] = [];
      // Timing for per-message metrics (TTFT, total, tokens/sec). Emitted via
      // onMessageComplete so callers (e.g. Studio's Chat route) can render a
      // stats badge without owning the runtime.
      // TODO: provide time metric from backend inference gateway (issue 219)
      const startMs = performance.now();
      let ttftMs = 0;
      let firstTokenMs = 0;
      let chunkCount = 0;
      let responseModel: string | undefined;
      let generatedChunkCount = 0;
      let reasoningChars = 0;
      let usageTokens: number | undefined;

      try {
        const result = await createChatCompletion({
          model,
          workspace,
          baseURL,
          messages: getOpenAIMessages(conversationMessages, promptData?.system_prompt),
          max_tokens: promptData?.inference_params?.max_tokens,
          temperature: promptData?.inference_params?.temperature,
          stream: true,
          stream_options: { include_usage: true },
          tools: tools?.length ? tools : undefined,
          signal: runController.signal,
        });

        if (runController.signal.aborted || !isCurrentRun()) {
          updateAssistantMessage(
            assistantMessage.id!,
            responseParts,
            responseImages,
            CANCELLED_STATUS
          );
          return;
        }

        if (isChatCompletionStream(result)) {
          const streamController = result.controller;
          runController.signal.addEventListener('abort', () => streamController.abort(), {
            once: true,
          });
          if (runController.signal.aborted) streamController.abort();
          for await (const chunk of result) {
            if (runController.signal.aborted || !isCurrentRun()) break;
            responseModel ??= chunk.model || undefined;
            usageTokens = getUsageCompletionTokens(chunk) ?? usageTokens;
            const delta = chunk.choices[0]?.delta.content ?? '';
            const reasoningDelta = getReasoningDelta(chunk);
            const toolCalls = chunk.choices[0]?.delta.tool_calls ?? [];
            const images = getCompletionImages(chunk);
            if (!reasoningDelta && !delta && !toolCalls.length && !images.length) continue;

            if (firstTokenMs === 0) firstTokenMs = Math.round(performance.now() - startMs);
            generatedChunkCount += 1;
            reasoningChars += reasoningDelta.length;
            if (delta || images.length > 0) {
              if (ttftMs === 0) ttftMs = Math.round(performance.now() - startMs);
              responseText += delta;
              responseImages.push(...images);
              chunkCount += 1;
            }
            responseParts = appendCompletionDeltaParts(responseParts, {
              reasoning: reasoningDelta,
              text: delta,
              toolCalls,
            });
            updateAssistantMessage(
              assistantMessage.id!,
              responseParts,
              responseImages,
              RUNNING_STATUS
            );
          }
          updateAssistantMessage(
            assistantMessage.id!,
            responseParts,
            responseImages,
            runController.signal.aborted ? CANCELLED_STATUS : COMPLETE_STATUS
          );
          if (!runController.signal.aborted && onMessageComplete) {
            const totalMs = Math.round(performance.now() - startMs);
            const { completionTokens, tokensPerSec } = computeCompletionMetrics({
              totalMs,
              firstTokenMs,
              usageTokens,
              outputChars: responseText.length + reasoningChars,
              chunkCount: generatedChunkCount,
            });
            onMessageComplete({
              assistantMessageId: assistantMessage.id!,
              text: responseText,
              ttftMs,
              totalMs,
              chunkCount,
              tokensPerSec,
              completionTokens,
              responseModel,
            });
          }
        } else {
          const text = getCompletionText(result);
          updateAssistantMessage(
            assistantMessage.id!,
            appendCompletionDeltaParts([], {
              text,
              toolCalls: result.choices[0]?.message.tool_calls?.map((toolCall, index) => ({
                ...toolCall,
                index,
              })),
            }),
            getCompletionImages(result),
            COMPLETE_STATUS
          );
          if (onMessageComplete) {
            const totalMs = Math.round(performance.now() - startMs);
            const { completionTokens, tokensPerSec } = computeCompletionMetrics({
              totalMs,
              firstTokenMs: 0,
              usageTokens: getUsageCompletionTokens(result),
              outputChars: text.length,
              chunkCount: 1,
            });
            onMessageComplete({
              assistantMessageId: assistantMessage.id!,
              text,
              ttftMs: totalMs,
              totalMs,
              chunkCount: 1,
              tokensPerSec,
              completionTokens,
              responseModel: result.model || undefined,
            });
          }
        }
      } catch (error: unknown) {
        if (runController.signal.aborted || isAbortError(error)) {
          updateAssistantMessage(
            assistantMessage.id!,
            responseParts,
            responseImages,
            CANCELLED_STATUS
          );
          return;
        }

        const errorMessage = error instanceof Error ? error.message : 'Unknown Error';
        const status: MessageStatus = {
          type: 'incomplete',
          reason: 'error',
          error: errorMessage,
        };
        updateAssistantMessage(
          assistantMessage.id!,
          [{ type: 'text', text: errorMessage }],
          responseImages,
          status
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
      baseURL,
      createChatCompletion,
      disabled,
      model,
      onError,
      onMessageComplete,
      promptData?.inference_params?.max_tokens,
      promptData?.inference_params?.temperature,
      promptData?.system_prompt,
      setThreadMessages,
      tools,
      updateAssistantMessage,
      workspace,
    ]
  );

  const handleNewMessage = useCallback(
    async (message: AppendMessage) => {
      const content = getUserMessageContent(message);
      if (content.length === 0) return;

      const userMessage = appendMessageToThreadMessage({ ...message, content });
      const nextMessages = [...messagesRef.current, userMessage];
      setThreadMessages(nextMessages);
      await runCompletion(nextMessages);
    },
    [runCompletion, setThreadMessages]
  );

  const handleReload = useCallback(async () => {
    const lastAssistantIndex = messagesRef.current.findLastIndex(
      (message) => message.role === 'assistant'
    );
    const nextMessages =
      lastAssistantIndex === -1
        ? messagesRef.current
        : messagesRef.current.slice(0, lastAssistantIndex);

    setThreadMessages(nextMessages);
    await runCompletion(nextMessages);
  }, [runCompletion, setThreadMessages]);

  const handleEdit = useCallback(
    async (message: AppendMessage) => {
      const sourceIndex = getEditedMessageIndex(messagesRef.current, message);
      const previousMessages =
        sourceIndex === -1 ? messagesRef.current : messagesRef.current.slice(0, sourceIndex);
      const content = getUserMessageContent(message);
      if (content.length === 0) return;

      const nextMessages = [
        ...previousMessages,
        appendMessageToThreadMessage({ ...message, content }),
      ];
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

  // External broadcast — when the caller bumps `broadcast.seq`, append the
  // payload text as a new user message and run a completion. The ref is seeded
  // with whatever seq is present at mount, so an AssistantChat that mounts
  // mid-flight (with a non-null broadcast prop) doesn't re-fire the last
  // broadcast it sees on first render. Subsequent changes fire.
  const broadcastSeenSeqRef = useRef<number | undefined>(broadcast?.seq);
  useEffect(() => {
    if (!broadcast) return;
    if (broadcast.seq === broadcastSeenSeqRef.current) return;
    broadcastSeenSeqRef.current = broadcast.seq;
    const text = broadcast.text.trim();
    if (!text || disabled) return;
    const synthetic: AppendMessage = {
      role: 'user',
      content: [{ type: 'text', text }],
    } as unknown as AppendMessage;
    void handleNewMessage(synthetic);
  }, [broadcast, disabled, handleNewMessage]);

  // External cancel — same sequence pattern.
  const stopSeenCountRef = useRef<number | undefined>(stopCount);
  useEffect(() => {
    if (stopCount === undefined) return;
    if (stopCount === stopSeenCountRef.current) return;
    stopSeenCountRef.current = stopCount;
    void handleCancel();
  }, [stopCount, handleCancel]);

  // Abort any in-flight request on unmount (panel removed or remounted).
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  // Surface running-state transitions so a parent can aggregate Stop logic
  // across multiple AssistantChat instances.
  useEffect(() => {
    onRunningChange?.(isRunning);
  }, [isRunning, onRunningChange]);

  // Surface empty/non-empty transitions so callers can derive seed-chip
  // visibility from whether the thread has any messages.
  const isEmpty = messages.length === 0;
  useEffect(() => {
    onEmptyChange?.(isEmpty);
  }, [isEmpty, onEmptyChange]);

  const runtime = useExternalStoreRuntime<ThreadMessageLike>({
    messages,
    setMessages: setThreadMessages,
    isDisabled: disabled,
    isRunning,
    onNew: handleNewMessage,
    onEdit: handleEdit,
    onReload: async () => handleReload(),
    onCancel: handleCancel,
    convertMessage: (message) => message,
    adapters: enableImageAttachments ? { attachments: imageAttachmentAdapter } : undefined,
    unstable_capabilities: {
      copy: true,
    },
  });

  return {
    handleReset,
    runtime,
  };
};
