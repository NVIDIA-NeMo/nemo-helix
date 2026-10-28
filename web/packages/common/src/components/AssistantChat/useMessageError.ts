// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuiState } from '@assistant-ui/react';

/** The raw error of the current message when its reply failed, otherwise `undefined`. */
export const useMessageError = (): string | undefined =>
  useAuiState((state) => {
    const status = state.message.status;
    return status?.type === 'incomplete' && status.reason === 'error'
      ? String(status.error ?? '')
      : undefined;
  });
