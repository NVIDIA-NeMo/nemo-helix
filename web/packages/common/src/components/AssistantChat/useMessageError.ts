// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type MessageStatus, useAuiState } from '@assistant-ui/react';
import { GENERIC_CHAT_ERROR } from '@nemo/common/src/components/AssistantChat/describeChatError';
import { getFailedReplyExplanation } from '@nemo/common/src/components/AssistantChat/messageUtils';

const isFailedReply = (status: MessageStatus | undefined): boolean =>
  status?.type === 'incomplete' && status.reason === 'error';

/**
 * The raw error of the current message when its reply failed, otherwise `undefined`. A failed
 * reply that carries no error text still counts as failed and reports the generic line, so the
 * bar, banner and its details row always agree on whether a reply failed.
 */
export const useMessageError = (): string | undefined =>
  useAuiState((state) => {
    const status = state.message.status;
    return status?.type === 'incomplete' && status.reason === 'error'
      ? String(status.error ?? '') || GENERIC_CHAT_ERROR
      : undefined;
  });

export const useFailedReplyExplanation = (): string | undefined =>
  useAuiState((state) => {
    if (!isFailedReply(state.message.status)) {
      return undefined;
    }
    return getFailedReplyExplanation(state.message.parts);
  });

export const useIsFailedReplyExplanationPart = (): boolean =>
  useAuiState(
    (state) =>
      isFailedReply(state.message.status) &&
      state.part.type === 'text' &&
      state.part === state.message.parts.findLast((part) => part.type === 'text')
  );
