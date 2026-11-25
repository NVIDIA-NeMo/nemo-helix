// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AssistantChatProps } from '@nemo/common/src/components/AssistantChat';
import type { ResourceRef } from '@nemo/common/src/types';
import type { ModelChatStatus } from '@nemo/common/src/utils/models';
import { DeployModelCta } from '@studio/components/ModelChat/DeployModelCta';

export function statusEmptyState(
  status: Exclude<ModelChatStatus, 'enabled'>,
  deployModelRef?: ResourceRef,
  deployModelLabel?: string
): NonNullable<AssistantChatProps['emptyState']> {
  if (status === 'pending') {
    return {
      slotHeading: 'Model Deployment in Progress',
      slotSubheading: 'Check back in a few minutes to chat with this model.',
    };
  }
  return {
    slotHeading: 'Chat Unavailable',
    slotSubheading: 'This model does not have an active deployment.',
    // Only `disabled` gets the CTA: `pending` already has a deployment coming up.
    slotAction: deployModelRef ? (
      <DeployModelCta modelRef={deployModelRef} label={deployModelLabel} />
    ) : undefined,
  };
}
