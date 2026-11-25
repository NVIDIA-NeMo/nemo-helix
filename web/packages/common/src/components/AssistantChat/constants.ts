// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MessageStatus } from '@assistant-ui/react';

export const RUNNING_STATUS: MessageStatus = { type: 'running' };
export const COMPLETE_STATUS: MessageStatus = { type: 'complete', reason: 'stop' };
export const CANCELLED_STATUS: MessageStatus = { type: 'incomplete', reason: 'cancelled' };

/** A reply the model stopped short of finishing; `error` carries the notice shown under it. */
const stoppedEarly = (
  reason: 'length' | 'content-filter' | 'other',
  notice: string
): MessageStatus => ({ type: 'incomplete', reason, error: notice });

export const TRUNCATED_STATUS = stoppedEarly(
  'length',
  'Response truncated by max tokens. Increase the limit to let the model finish.'
);
export const CONTENT_FILTERED_STATUS = stoppedEarly(
  'content-filter',
  "Response stopped by the provider's content filter."
);
export const ENDED_EARLY_STATUS = stoppedEarly(
  'other',
  'Response ended unexpectedly. The connection may have dropped. Try regenerating.'
);
export const EMPTY_REPLY_STATUS = stoppedEarly(
  'other',
  'The model finished without returning a response.'
);
