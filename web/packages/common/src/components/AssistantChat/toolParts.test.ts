// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAssistantCollapsedStudioDetailsPart } from '@nemo/common/src/components/AssistantChat/toolParts';

describe('createAssistantCollapsedStudioDetailsPart', () => {
  it('keeps the error state and result of each collapsed tool call', () => {
    const part = createAssistantCollapsedStudioDetailsPart({
      label: 'worked for 42s',
      parts: [
        {
          type: 'tool-call',
          args: { id: 'job-1' },
          argsText: '{"id":"job-1"}',
          isError: true,
          result: 'job not found',
          toolCallId: 'call_job',
          toolName: 'get_job',
        },
      ],
    });

    expect(part).toMatchObject({
      args: {
        parts: [
          expect.objectContaining({
            isError: true,
            result: 'job not found',
            toolName: 'get_job',
          }),
        ],
      },
    });
  });
});
