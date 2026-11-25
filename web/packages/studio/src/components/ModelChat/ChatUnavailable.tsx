// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResourceRef } from '@nemo/common/src/types';
import type { ModelChatStatus } from '@nemo/common/src/utils/models';
import { Empty } from '@studio/components/Empty';
import { DeployModelCta } from '@studio/components/ModelChat/DeployModelCta';
import { statusEmptyState } from '@studio/components/ModelChat/statusEmptyState';
import type { FC } from 'react';

export interface ChatUnavailableProps {
  status: ModelChatStatus;
  isAdapterUnserved?: boolean;
  /** The model to deploy; for an adapter, its base model. */
  deployModelRef: ResourceRef;
  deployModelLabel?: string;
}

/** Why a model or adapter can't be chatted with yet, and how to fix it. */
export const ChatUnavailable: FC<ChatUnavailableProps> = ({
  status,
  isAdapterUnserved,
  deployModelRef,
  deployModelLabel,
}) => {
  if (isAdapterUnserved || status === 'enabled') {
    return (
      <Empty
        title="Adapter is not currently served"
        description="No provider lists this adapter among its served models, so it cannot be chatted with yet. It becomes available once the deployment serving its base model has loaded the adapter."
        slotAction={<DeployModelCta modelRef={deployModelRef} label="Deploy base model" />}
      />
    );
  }

  const { slotHeading, slotSubheading, slotAction } = statusEmptyState(
    status,
    deployModelRef,
    deployModelLabel
  );
  return <Empty title={slotHeading} description={slotSubheading} slotAction={slotAction} />;
};
