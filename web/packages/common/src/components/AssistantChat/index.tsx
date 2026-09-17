// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AssistantRuntimeProvider } from '@assistant-ui/react';
import { modelSupportsImageAttachments } from '@nemo/common/src/components/AssistantChat/messageUtils';
import cn from 'classnames';
import { type FC, useMemo } from 'react';

import { AssistantChatThread } from './AssistantChatThread';
import type { AssistantChatProps } from './types';
import { useAssistantChatRuntime } from './useAssistantChatRuntime';

export type { AssistantChatProps } from './types';
export { ComposerMode } from './types';

export const AssistantChat: FC<AssistantChatProps> = ({
  model,
  workspace,
  baseURL,
  promptData,
  tools,
  extraHeaders,
  assistantName,
  placeholder,
  disabled = false,
  showRunningIndicator,
  attributes,
  className,
  initialMessages = [],
  onError,
  onMessageComplete,
  onRunningChange,
  onEmptyChange,
  composerMode,
  broadcast,
  stopCount,
  slotComposerStart,
  emptyState,
  messageContentProps,
  composerOverride,
  enableImageAttachments = true,
}) => {
  // Gate image attachments on both the caller's opt-in and a naive model
  // capability check, so a text-only model never offers an image affordance.
  const imageAttachmentsEnabled = enableImageAttachments && modelSupportsImageAttachments(model);

  const { handleReset, runtime } = useAssistantChatRuntime({
    model,
    workspace,
    baseURL,
    promptData,
    tools,
    extraHeaders,
    disabled,
    initialMessages,
    onError,
    onMessageComplete,
    onRunningChange,
    onEmptyChange,
    broadcast,
    stopCount,
    enableImageAttachments: imageAttachmentsEnabled,
  });

  const composerPlaceholder = useMemo(
    () => placeholder || `Message ${assistantName || model || 'Your Assistant'}`,
    [assistantName, model, placeholder]
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className={cn('h-full w-full', className)}>
        <AssistantChatThread
          disabled={disabled}
          placeholder={composerPlaceholder}
          onReset={handleReset}
          showRunningIndicator={showRunningIndicator}
          attributes={attributes}
          composerMode={composerMode}
          slotComposerStart={slotComposerStart}
          emptyState={emptyState}
          messageContentProps={messageContentProps}
          composerOverride={composerOverride}
          enableImageAttachments={imageAttachmentsEnabled}
        />
      </div>
    </AssistantRuntimeProvider>
  );
};
