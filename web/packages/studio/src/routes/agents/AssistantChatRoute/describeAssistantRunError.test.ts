// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { humanizeChatError } from '@nemo/common/src/components/AssistantChat/describeChatError';
import {
  ASSISTANT_RUN_INTERRUPTED_ERROR,
  describeAssistantRunError,
} from '@studio/routes/agents/AssistantChatRoute/describeAssistantRunError';

describe('describeAssistantRunError', () => {
  it('explains an interrupted run with the History hint', () => {
    expect(humanizeChatError(ASSISTANT_RUN_INTERRUPTED_ERROR, describeAssistantRunError)).toBe(
      ASSISTANT_RUN_INTERRUPTED_ERROR
    );
  });

  it('leaves other failures to the generic line', () => {
    expect(describeAssistantRunError('500 Internal Server Error')).toBeUndefined();
  });
});
