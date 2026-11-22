// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DescribeChatError } from '@nemo/common/src/components/AssistantChat/describeChatError';

const ASSISTANT_RUN_INTERRUPTED = 'Connection to NeMo Assistant was interrupted.';
const ASSISTANT_RUN_INTERRUPTED_HINT =
  'Your response may still be processing. Check History to see the result.';

export const ASSISTANT_RUN_INTERRUPTED_ERROR = `${ASSISTANT_RUN_INTERRUPTED} ${ASSISTANT_RUN_INTERRUPTED_HINT}`;

export const describeAssistantRunError: DescribeChatError = (message) => {
  if (!message.startsWith(ASSISTANT_RUN_INTERRUPTED)) {
    return undefined;
  }
  return { title: ASSISTANT_RUN_INTERRUPTED, hint: ASSISTANT_RUN_INTERRUPTED_HINT };
};
