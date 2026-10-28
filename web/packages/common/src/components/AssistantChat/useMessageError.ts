// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuiState } from '@assistant-ui/react';
import { GENERIC_CHAT_ERROR } from '@nemo/common/src/components/AssistantChat/describeChatError';

/**
 * The raw error of the current message when its reply failed, otherwise `undefined`. A failed
 * reply that carries no error text still counts as failed and reports the generic line, so the
 * bar, banner and copy action always agree on whether a reply failed.
 */
export const useMessageError = (): string | undefined =>
  useAuiState((state) => {
    const status = state.message.status;
    return status?.type === 'incomplete' && status.reason === 'error'
      ? String(status.error ?? '') || GENERIC_CHAT_ERROR
      : undefined;
  });
