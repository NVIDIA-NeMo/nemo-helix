// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ThreadMessageLike,
  ThreadPrimitive,
  ToolCallMessagePartComponent,
} from '@assistant-ui/react';
import type { MessageContentProps } from '@nemo/common/src/components/Chat/MessageContent';
import type { PromptData } from '@nemo/sdk/generated/platform/schema';
import type { ChatCompletionTool } from 'openai/resources/index.mjs';
import type { ComponentProps, ReactNode } from 'react';

export const ComposerMode = {
  PER_PANEL: 'per-panel',
  BROADCAST_ALL: 'broadcast-all',
} as const;
export type ComposerMode = (typeof ComposerMode)[keyof typeof ComposerMode];

export interface AssistantChatThreadAttributes {
  ThreadViewport?: ComponentProps<typeof ThreadPrimitive.Viewport>;
}

export type AssistantChatMessageContentProps = Pick<MessageContentProps, 'markdownLinkComponent'>;

export interface MessageRenderProps {
  messageContentProps?: AssistantChatMessageContentProps;
  toolCallPartComponent?: ToolCallMessagePartComponent;
}

export interface AssistantChatThreadProps {
  animateAssistantMessageCompletion?: boolean;
  disabled?: boolean;
  placeholder: string;
  onReset: () => void;
  showRunningIndicator?: boolean;
  attributes?: AssistantChatThreadAttributes;
  composerMode?: ComposerMode;
  slotComposerStart?: ReactNode;
  emptyState?: {
    slotHeading?: string;
    slotSubheading?: string;
    slotAction?: ReactNode;
  };
  contentClassName?: string;
  composerContainerClassName?: string;
  hideAssistantMessageActions?: boolean;
  toolCallPartComponent?: ToolCallMessagePartComponent;
  viewportClassName?: string;
  composerOverride?: ReactNode;
  messageContentProps?: AssistantChatMessageContentProps;
  enableImageAttachments?: boolean;
  minInputRows?: number;
}

export interface AssistantChatProps {
  /**
   * The model name to route through inference gateway.
   */
  model: string;
  /**
   * Workspace used to build the default inference gateway URL.
   */
  workspace?: string;
  /**
   * Explicit OpenAI-compatible chat completions base URL. When omitted, `useChatCompletion`
   * resolves inference gateway routing from workspace and model.
   */
  baseURL?: string;
  /**
   * Optional prompt data used for system prompt and inference parameter defaults.
   */
  promptData?: PromptData;
  /**
   * Optional OpenAI-compatible tools for the request.
   */
  tools?: ChatCompletionTool[];
  /**
   * Display name used in the composer placeholder.
   */
  assistantName?: string;
  placeholder?: string;
  disabled?: boolean;
  showRunningIndicator?: boolean;
  attributes?: AssistantChatThreadAttributes;
  className?: string;
  initialMessages?: readonly ThreadMessageLike[];
  onError?: (error: Error) => void;
  /**
   * Called once per assistant message after the stream completes (or after
   * the non-stream completion lands). Surfaces per-message timing so callers
   * can render their own latency/throughput UI without owning the runtime.
   * Not invoked on cancellation or error.
   */
  onMessageComplete?: (info: AssistantMessageCompletion) => void;
  /**
   * Fires whenever the runtime's "is currently streaming" state changes.
   * Lets a parent (e.g. a page that hosts many AssistantChats) aggregate the
   * running state across instances — used by the Chat route to drive a global
   * Stop button in Compare mode.
   */
  onRunningChange?: (isRunning: boolean) => void;
  /**
   * Fires whenever the thread transitions between empty and non-empty. Lets a
   * parent derive seed-chip visibility from whether any messages exist.
   */
  onEmptyChange?: (isEmpty: boolean) => void;
  /**
   * Controls whether the internal composer is shown and how input is driven.
   * In `broadcast-all` mode the composer is suppressed; a page-level composer
   * drives every AssistantChat in parallel.
   * @default ComposerMode.PER_PANEL
   */
  composerMode?: ComposerMode;
  /**
   * External broadcast trigger. Whenever `seq` changes (excluding initial
   * mount), the runtime appends `text` as a new user message and runs a
   * completion — same code path as a user typing into the composer.
   */
  broadcast?: BroadcastSignal;
  /**
   * Monotonic counter — when it changes, the runtime aborts any in-flight
   * stream. Lets a parent cancel many AssistantChats at once.
   */
  stopCount?: number;
  /**
   * Content rendered immediately above the composer, inside the same outer
   * frame. Use for seed-prompt chips or any prefatory hint that should read
   * as part of the composer affordance rather than a separate block.
   */
  slotComposerStart?: ReactNode;
  emptyState?: {
    slotHeading?: string;
    slotSubheading?: string;
    slotAction?: ReactNode;
  };
  /** Overrides used when rendering Markdown inside chat messages. */
  messageContentProps?: AssistantChatMessageContentProps;
  composerOverride?: ReactNode;
  /**
   * @default true
   */
  enableImageAttachments?: boolean;
}

export interface BroadcastSignal {
  /** Monotonically increasing sequence — on change, runtime fires a send. */
  seq: number;
  /** Text to inject as the user's next message. */
  text: string;
}

export interface AssistantMessageCompletion {
  assistantMessageId: string;
  text: string;
  /** ms from request start to first delta (0 if non-stream). */
  ttftMs: number;
  /** ms from request start to final delta. */
  totalMs: number;
  /** Number of delta chunks (1 for non-stream). */
  chunkCount: number;
  /** Approximate; chars/4 fallback when the API doesn't return a usage block. */
  completionTokens: number;
  /** Completion tokens per second of streaming wall-time (excludes TTFT). */
  tokensPerSec: number;
}
