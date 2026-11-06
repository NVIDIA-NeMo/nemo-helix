/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { type ModelProvider, ModelProviderStatus } from '@nemo/sdk/generated/platform/schema';
import { Banner, SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import { type FC, useMemo } from 'react';

function servedModelLabels(provider: ModelProvider): string[] {
  if (provider.served_models?.length) {
    return provider.served_models.map((m) => m.served_model_name || m.model_entity_id);
  }
  return [];
}

function isStatusMessageError(status: ModelProviderStatus | undefined): boolean {
  return (
    status === ModelProviderStatus.ERROR ||
    status === ModelProviderStatus.UNKNOWN ||
    status === ModelProviderStatus.LOST
  );
}

export interface InferenceProviderDetailsSidePanelProps {
  open: boolean;
  onClose: () => void;
  provider: ModelProvider;
}

export const InferenceProviderDetailsSidePanel: FC<InferenceProviderDetailsSidePanelProps> = ({
  open,
  onClose,
  provider,
}) => {
  const models = useMemo(() => (provider ? servedModelLabels(provider) : []), [provider]);
  const sortedModels = useMemo(() => [...models].sort(), [models]);
  const statusMessage = provider?.status_message?.trim();

  const statusMessageContent = isStatusMessageError(provider.status) ? (
    <Banner className="w-full " kind="inline" status="error">
      <Text className="whitespace-pre-wrap" kind="body/regular/sm">
        {statusMessage}
      </Text>
    </Banner>
  ) : (
    statusMessage
  );
  return (
    <SidePanel
      className="w-[600px]"
      bordered
      modal
      open={open}
      slotHeading={
        <Text className="min-w-0 truncate" kind="label/bold/lg" title={provider.name}>
          {provider.name}
        </Text>
      }
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          onClose();
        }
      }}
    >
      <Stack className="min-h-0 flex-1 gap-density-lg overflow-auto">
        <Stack className="gap-density-md">
          <KVPair
            label="Created"
            orientation="horizontal"
            size="medium"
            value={
              provider.created_at ? (
                <RelativeTime datetime={provider.created_at} focusableForTooltip={false} />
              ) : (
                '—'
              )
            }
          />
          <KVPair
            label="Host URL"
            orientation="horizontal"
            size="medium"
            truncate={false}
            value={provider.host_url || '—'}
          />
          {provider.api_key_secret_name ? (
            <KVPair
              label="API key secret"
              orientation="horizontal"
              size="medium"
              truncate={false}
              value={provider.api_key_secret_name}
            />
          ) : null}
          {provider.description?.trim() ? (
            <KVPair
              label="Description"
              orientation="horizontal"
              size="medium"
              truncate={false}
              value={provider.description.trim()}
            />
          ) : null}
          <KVPair label="Status" value={<StatusBadge status={provider.status} />} />
          <KVPair
            attributes={{ value: { className: 'whitespace-pre-wrap' } }}
            label="Status message"
            value={statusMessageContent}
          />
          <KVPair
            attributes={{ value: { className: 'whitespace-pre-wrap' } }}
            label="Served models"
            value={sortedModels.length > 0 ? sortedModels.join('\n') : '—'}
          />
        </Stack>
      </Stack>
    </SidePanel>
  );
};
