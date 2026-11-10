// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type MessageStatus, useAuiState } from '@assistant-ui/react';
import {
  getFailedReplyExplanation,
  getFailedReplyExplanationPart,
} from '@nemo/common/src/components/AssistantChat/messageUtils';

const isFailedReply = (
  status: MessageStatus | undefined
): status is Extract<MessageStatus, { type: 'incomplete' }> =>
  status?.type === 'incomplete' && status.reason === 'error';

export const useIsFailedReply = (): boolean =>
  useAuiState((state) => isFailedReply(state.message.status));

export const useMessageError = (): string | undefined =>
  useAuiState((state) => {
    const status = state.message.status;
    if (!isFailedReply(status)) {
      return undefined;
    }
    return String(status.error ?? '') || undefined;
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
      state.part === getFailedReplyExplanationPart(state.message.parts)
  );
